import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Upload from './Upload';
import FileItem from './FileItem';

const API_BASE_URL = 'http://localhost:5000';

const filterFunctions = {
  minMax: (data) => {
    const min = Math.min(...data);
    const max = Math.max(...data);
    return data.map(value => (value - min) / (max - min));
  },
  rayRemoval1: (data) => {
    return data.map(value => value > 1000 ? 1000 : value);
  },
  rayRemoval2: (data) => {
    return data.map(value => value < 0 ? 0 : value);
  },
  // Add more filter functions here
};

const Preprocess = () => {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [fileData, setFileData] = useState(null);
  const [filteredData, setFilteredData] = useState(null);
  const [activeFilters, setActiveFilters] = useState({});
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [file, setFile] = useState([]);
  const [directoryStructure, setDirectoryStructure] = useState({});
  const [expandedFolders, setExpandedFolders] = useState({});
  const [outputFolder, setOutputFolder] = useState('');

  useEffect(() => {
    fetchDirectoryStructure();
  }, []);

  const fetchDirectoryStructure = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/directory-structure`);
      setDirectoryStructure(response.data.structure);
    } catch (error) {
      console.error('Error fetching directory structure:', error);
      setError('Failed to fetch directory structure. Please try again.');
    }
  };

  const handleFileSelect = async (file) => {
    setSelectedFile(file);
    try {
      const response = await axios.get(`${API_BASE_URL}/file-data/${file}`);
      setFileData(response.data.data);
      setFilteredData(null);
      setActiveFilters({});
    } catch (error) {
      console.error('Error fetching file data:', error);
      setError('Failed to fetch file data. Please try again.');
    }
  };

  const handleFilterChange = (filterName) => {
    setActiveFilters(prev => ({
      ...prev,
      [filterName]: !prev[filterName]
    }));
  };

  const applyFilters = () => {
    if (!fileData) return;

    let result = [...fileData];
    Object.entries(activeFilters).forEach(([filterName, isActive]) => {
      if (isActive && filterFunctions[filterName]) {
        result = filterFunctions[filterName](result);
      }
    });
    setFilteredData(result);
  };

  const saveFilteredData = async () => {
    if (!filteredData || !selectedFile) return;

    try {
      const response = await axios.post(`${API_BASE_URL}/save-filtered-data`, {
        filename: selectedFile,
        data: filteredData,
        outputFolder: outputFolder
      });
      setMessage(response.data.message);
      fetchDirectoryStructure();
    } catch (error) {
      console.error('Save Error:', error);
      setError(error.response?.data?.error || 'An error occurred while saving');
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
      fetchDirectoryStructure();
    } catch (error) {
      console.error('Upload Error:', error);
      setError(error.response?.data?.error || 'An error occurred during upload');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleFolder = (folderPath) => {
    setExpandedFolders(prev => ({
      ...prev,
      [folderPath]: !prev[folderPath]
    }));
  };

  const renderDirectory = (structure, path = '') => {
    return Object.entries(structure).map(([name, content]) => {
      const fullPath = path ? `${path}/${name}` : name;
      if (typeof content === 'object') {
        // It's a folder
        return (
          <div key={fullPath} className="folder">
            <div onClick={() => toggleFolder(fullPath)} className="folder-name">
              {expandedFolders[fullPath] ? '🡫' : '🡪'} 🖿 {name}
            </div>
            {expandedFolders[fullPath] && (
              <div className="folder-contents">
                {renderDirectory(content, fullPath)}
              </div>
            )}
          </div>
        );
      } else {
        // It's a file
        return (
          <div key={fullPath} className="file">
            <FileItem
            file={name}
            isSelected={selectedFile === fullPath}
            onSelect={() => handleFileSelect(fullPath)}
            onDelete={() => deleteFile(fullPath)}
            />
          </div>
        );
      }
    });
  };

  const deleteFile = async (fileToDelete) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/delete-file`, { filename: fileToDelete });
      setMessage(response.data.message);
      fetchDirectoryStructure();
    } catch (error) {
      console.error('Delete Error:', error);
      setError(error.response?.data?.error || 'An error occurred during deletion');
    }
  };

  return (
    <div className="preprocess-container">
      <div className="main-content">
        <div className="data-visualization">
          {(fileData || filteredData) && (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={(filteredData || fileData).map((y, index) => ({ x: index, y }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
        <div className="filters">
          <h3>Filters</h3>
          <div className="filter-options">
            {Object.keys(filterFunctions).map(filterName => (
              <label key={filterName}>
                <input
                  type="checkbox"
                  checked={activeFilters[filterName] || false}
                  onChange={() => handleFilterChange(filterName)}
                />
                {filterName}
              </label>
            ))}
          </div>
          <div className="filter-actions">
            <button onClick={applyFilters}>Apply Filters</button>
            {filteredData && (
              <>
                <input
                  type="text"
                  value={outputFolder}
                  onChange={(e) => setOutputFolder(e.target.value)}
                  placeholder="Output folder (optional)"
                />
                <button onClick={saveFilteredData}>Save Filtered Data</button>
              </>
            )}
          </div>
        </div>
      </div>
      <div className="sidebar">
        <h3>Files</h3>
        <div className="file-list">
          {renderDirectory(directoryStructure)}
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





