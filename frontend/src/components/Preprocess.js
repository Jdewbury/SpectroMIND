import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import FileItem from './FileItem';

const API_BASE_URL = 'http://localhost:5000';

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
  const [minWavenumber, setMinWavenumber] = useState('');
  const [maxWavenumber, setMaxWavenumber] = useState('');
  const [activeFilterTab, setActiveFilterTab] = useState('');
  const [filterCategories, setFilterCategories] = useState([]);
  const [filterOptions, setFilterOptions] = useState({});
  const [filterInputs, setFilterInputs] = useState({});
  const [filterConfig, setFilterConfig] = useState({});
  const [selectedSpectrumIndex, setSelectedSpectrumIndex] = useState(0);

  const fetchDirectoryStructure = async () => {
    let isMounted = true;

    try {
      console.log('Fetching directory structure from:', `${API_BASE_URL}/directory-structure`);
      const response = await axios.get(`${API_BASE_URL}/directory-structure`);
      console.log('Directory structure response:', response.data);
      if (isMounted) {
        setDirectoryStructure(response.data.structure || {});
      }
    } catch (error) {
      console.error('Error fetching directory structure:', error);
      if (isMounted) {
        setError('Failed to fetch directory structure. Please try again.');
      }
    }

    return () => {
      isMounted = false;
    };
  };

  useEffect(() => {
    fetchDirectoryStructure();
    fetchFilterCategories();
    fetchFilterConfig();
  }, []);

  const handleFileSelect = async (file) => {
    setSelectedFile(file);
    try {
      const response = await axios.get(`${API_BASE_URL}/file-data/${file}`);
      const data = response.data.data;
      setFileData(data);
      setFilteredData(null);
      setActiveFilters({});
      setSelectedSpectrumIndex(0);
    } catch (error) {
      console.error('Error fetching file data:', error);
      setError('Failed to fetch file data. Please try again.');
    }
  };

  const fetchFilterCategories = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/get-filters`);
      setFilterCategories(response.data);
      if (response.data.length > 0) {
        setActiveFilterTab(response.data[0]);
      }
      fetchFilterOptions(response.data);
    } catch (error) {
      console.error('Error fetching filter categories:', error);
      setError('Failed to fetch filter categories. Please try again.');
    }
  };

  const renderFilterOptions = () => {
    const options = filterOptions[activeFilterTab] || [];
    const optionsPerColumn = 2; // Adjust this number to change the number of options per column

    return (
      <div className="filter-options-container">
        <div className="filter-options-content">
          {Array.from({ length: Math.ceil(options.length / optionsPerColumn) }, (_, columnIndex) => (
            <div key={columnIndex} className="filter-column">
              {options.slice(columnIndex * optionsPerColumn, (columnIndex + 1) * optionsPerColumn).map(filterName => (
                <div key={filterName} className="filter-option">
                  <label>
                    <input
                      type="checkbox"
                      checked={activeFilters[filterName] || false}
                      onChange={() => handleFilterChange(filterName)}
                    />
                    {filterName}
                  </label>
                  {activeFilters[filterName] && filterConfig[filterName] && (
                    <div className="filter-inputs">
                      {Object.entries(filterConfig[filterName]).map(([inputName, inputConfig]) => (
                        <div key={inputName}>
                          <label>{inputConfig.label}:</label>
                          <input
                            type={inputConfig.type}
                            value={filterInputs[filterName]?.[inputName] || inputConfig.default}
                            onChange={(e) => handleFilterInputChange(filterName, inputName, e.target.value)}
                            min={inputConfig.min}
                            max={inputConfig.max}
                            step={inputConfig.step}
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderFilterTabs = () => {
    return (
      <div className="filter-tabs">
        {filterCategories.map(category => (
          <button
            key={category}
            className={`filter-tab ${activeFilterTab === category ? 'active' : ''}`}
            onClick={() => setActiveFilterTab(category)}
          >
            {category.charAt(0).toUpperCase() + category.slice(1)}
          </button>
        ))}
      </div>
    );
  };

  const fetchFilterOptions = async (categories) => {
    const options = {};
    for (const category of categories) {
      try {
        const response = await axios.get(`${API_BASE_URL}/get-filter-options?category=${category}`);
        options[category] = response.data;
      } catch (error) {
        console.error(`Error fetching filter options for ${category}:`, error);
      }
    }
    setFilterOptions(options);
  };

  const fetchFilterConfig = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/get-filter-config`);
      setFilterConfig(response.data);
    } catch (error) {
      console.error('Error fetching filter config:', error);
      setError('Failed to fetch filter configuration. Please try again.');
    }
  };

  const handleFilterChange = (filterName) => {
    setActiveFilters(prev => ({
      ...prev,
      [filterName]: !prev[filterName]
    }));
  };

  const handleFilterInputChange = (filterName, inputName, value) => {
    setFilterInputs(prev => ({
      ...prev,
      [filterName]: {
        ...prev[filterName],
        [inputName]: value
      }
    }));
  };

  const applyFilters = async () => {
    if (!fileData) return;
  
    try {
      const response = await axios.post(`${API_BASE_URL}/apply-filters`, {
        data: fileData,
        filters: Object.keys(activeFilters).filter(filter => activeFilters[filter]),
        filterInputs: Object.fromEntries(
          Object.entries(filterInputs).map(([filter, inputs]) => [
            filter,
            Object.fromEntries(
              Object.entries(inputs).map(([key, value]) => [key, Number(value)])
            )
          ])
        )
      });
      setFilteredData(response.data.filtered_data);
    } catch (error) {
      console.error('Error applying filters:', error);
      setTimedMessage(setError, `Failed to apply filters: ${error.response?.data?.error || 'Unknown error'}`, 5000);
    }
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

  const setTimedMessage = (setterFunction, message, duration = 3000) => {
    setterFunction(message);
    setTimeout(() => setterFunction(''), duration);
  };

  const handleFileUpload = async (filesToUpload) => {
    if (filesToUpload.length === 0) {
      setTimedMessage(setError, 'Please select .npy files');
      return;
    }
    const formData = new FormData();
    filesToUpload.forEach((f) => formData.append('files[]', f));
    setIsLoading(true);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/upload-multiple`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setTimedMessage(setMessage, response.data.message);
      fetchDirectoryStructure();
    } catch (error) {
      console.error('Upload Error:', error);
      setTimedMessage(setError, error.response?.data?.error || 'An error occurred during upload');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = async (event) => {
    const selectedFiles = Array.from(event.target.files);
    const validFiles = selectedFiles.filter((file) => file.name.endsWith('.npy'));
    
    if (validFiles.length === selectedFiles.length) {
      setFile(validFiles);
      setError('');
      await handleFileUpload(validFiles);
    } else {
      setFile([]);
      setError('Please select .npy files only');
    }
  };

  const toggleFolder = (folderPath) => {
    setExpandedFolders(prev => ({
      ...prev,
      [folderPath]: !prev[folderPath]
    }));
  };

  const deleteFile = async (fileToDelete) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/delete-file`, { filename: fileToDelete });
      setTimedMessage(setMessage, response.data.message);
      fetchDirectoryStructure();
    } catch (error) {
      console.error('Delete Error:', error);
      setTimedMessage(setError, error.response?.data?.error || 'An error occurred during deletion');
    }
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

  const getWavenumberRange = (length) => {
    const min = parseFloat(minWavenumber);
    const max = parseFloat(maxWavenumber);

    if (!isNaN(min) && !isNaN(max) && min < max) {
      const step = (max - min) / (length - 1);
      return Array.from({ length }, (_, index) => min + index * step);
    }

    return Array.from({ length }, (_, index) => index);
  };

  const renderSpectrum = () => {
    const dataToRender = filteredData || fileData;
    if (!dataToRender) return null;

    const spectrumsCount = dataToRender.length;
    const currentSpectrum = dataToRender[selectedSpectrumIndex];

    return (
      <div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={currentSpectrum.map((y, index) => ({
              x: Math.round(getWavenumberRange(currentSpectrum.length)[index]),
              y
            }))}
          >   
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" tick={{ fontSize: 12 }}/>
            <YAxis 
              label={{ 
                value: 'Intensity', 
                angle: -90, 
                position: 'insideLeft', 
                offset: 0, 
                style: { textAnchor: 'middle' } 
              }} 
            />
            <Tooltip />
            <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} />
          </LineChart>
        </ResponsiveContainer>
        {spectrumsCount > 1 && (
          <div className="navigation-container">
          <label className="navigation-button" onClick={() => setSelectedSpectrumIndex(prev => Math.max(0, prev - 1))}>
            Previous
          </label>
          <span>Spectrum {selectedSpectrumIndex + 1} of {spectrumsCount}</span>
          <label className="navigation-button" onClick={() => setSelectedSpectrumIndex(prev => Math.min(spectrumsCount - 1, prev + 1))}>
            Next
          </label>
        </div>
        )}
      </div>
    );
  };

  return (
    <div className="preprocess-container">
      <div className="main-content">
      <div className="data-visualization">
          {fileData || filteredData ? (
            renderSpectrum()
          ) : (
            <div className="no-data">
              <p>Please select file to visualize</p>
            </div>
          )}
        </div>
        <div className="filters">
          <h3>Filters</h3>
          {renderFilterTabs()}
          <div className="filter-content">
            {renderFilterOptions()}
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
        <div className="files-header">
          <h3>Files</h3>
          <input
            type="file"
            multiple
            onChange={handleFileChange}
            style={{ display: 'none' }}
            id="file-upload"
            accept=".npy"
          />
          <label htmlFor="file-upload" className="upload-button">
            Upload
          </label>
        </div>
        <div className="file-list">
          {Object.keys(directoryStructure).length > 0 ? (
            renderDirectory(directoryStructure)
          ) : (
            <p className="error-message">No files or directories found.</p>
          )}
        </div>
        {message && <p className="upload-message">{message}</p>}
        {error && <p className="error-message">{error}</p>}
      </div>
    </div>
  );
};

export default Preprocess;






