import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Upload from './Upload';
import FileItem from './FileItem';  // Adjust path as per your file structure

const API_BASE_URL = 'http://localhost:5000';  // Adjust if your Flask app is running on a different port

const Preprocess = () => {
    const [files, setFiles] = useState([]);
    const [selectedFile, setSelectedFile] = useState('');
    const [fileData, setFileData] = useState(null);
    const [filters, setFilters] = useState({ minMax: false, rayRemoval1: false, rayRemoval2: false });
    const [error, setError] = useState('');
    const [message, setMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [file, setFile] = useState([]);

    useEffect(() => {
        fetchFiles();
    }, []);

    const fetchFiles = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/list-files`);
            setFiles(response.data.files);
        } catch (error) {
            console.error('Error fetching files:', error);
            setError('Failed to fetch files. Please try again.');
        }
    };

    const handleFileChange = (event) => {
        const selectedFiles = Array.from(event.target.files);
        const validFiles = selectedFiles.filter((file) => file.name.endsWith('.npy'));
        
        if (validFiles.length === selectedFiles.length) {
            setFile(validFiles);
            setError('');
        } else {
            setFile([]);
            setError('Please select .npy files only');
        }
    };

    const handleFileUpload = async () => {
        if (file.length === 0) {
            setError('Please select .npy files');
            return;
        }
        const formData = new FormData();
        file.forEach((f) => formData.append('files[]', f));
        setIsLoading(true);
        
        try {
            const response = await axios.post(`${API_BASE_URL}/upload-multiple`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setMessage(response.data.message);
            fetchFiles();
        } catch (error) {
            console.error('Upload Error:', error);
            setError(error.response?.data?.error || 'An error occurred during upload');
        } finally {
            setIsLoading(false);
        }
    };

    const handleFileSelect = async (file) => {
        setSelectedFile(file);
        try {
            const response = await axios.get(`${API_BASE_URL}/file-data/${file}`);
            setFileData(response.data.data);
        } catch (error) {
            console.error('Error fetching file data:', error);
            setError('Failed to fetch file data. Please try again.');
        }
    };

    const handleFilterChange = (filter) => {
        setFilters((prevFilters) => ({
            ...prevFilters,
            [filter]: !prevFilters[filter],
        }));
    };

    const applyFilters = async () => {
        console.log('Applying filters:', filters);
    };

    const deleteFile = async (fileToDelete) => {
        try {
            const response = await axios.post(`${API_BASE_URL}/delete-file`, { filename: fileToDelete });
            setMessage(response.data.message);
            fetchFiles(); 
        } catch (error) {
            console.error('Delete Error:', error);
            setError(error.response?.data?.error || 'An error occurred during deletion');
        }
    };

    return (
        <div className="preprocess-container">
            <div className="main-content">
                <div className="data-visualization">
                    {selectedFile && fileData ? (
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={fileData.map((y, index) => ({ x: index, y }))}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="x" />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="placeholder">Select a file to visualize data</div>
                    )}
                </div>
                <div className="filters">
                    <h3>Filters</h3>
                    <div className="filter-options">
                        <label>
                            <input
                                type="checkbox"
                                checked={filters.minMax}
                                onChange={() => handleFilterChange('minMax')}
                            />
                            Min-Max
                        </label>
                        <label>
                            <input
                                type="checkbox"
                                checked={filters.rayRemoval1}
                                onChange={() => handleFilterChange('rayRemoval1')}
                            />
                            Ray Removal 1
                        </label>
                        <label>
                            <input
                                type="checkbox"
                                checked={filters.rayRemoval2}
                                onChange={() => handleFilterChange('rayRemoval2')}
                            />
                            Ray Removal 2
                        </label>
                    </div>
                    <div className="filter-actions">
                        <button onClick={applyFilters}>Apply</button>
                    </div>
                </div>
            </div>
            <div className="sidebar">
                <h3>Files</h3>
                <div className="file-list">
                    {files.length > 0 ? (
                        files.map((file) => (
                            <FileItem
                                key={file}
                                file={file}
                                isSelected={selectedFile === file}
                                onSelect={() => handleFileSelect(file)}
                                onDelete={() => deleteFile(file)}
                            />
                        ))
                    ) : (
                        <p>No files available</p>
                    )}
                </div>
                <div className="file-upload">
                    <Upload
                        handleFileChange={handleFileChange}
                        handleUpload={handleFileUpload}
                        file={file}
                        error={error}
                        message={message}
                        isLoading={isLoading}
                    />
                </div>
            </div>
        </div>
    );
};

export default Preprocess;




