import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Filevisualizer = ({ onSelectFile }) => {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');

  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const response = await axios.get('/list-files');
        setFiles(response.data.files);
      } catch (error) {
        console.error('Error fetching files:', error);
      }
    };

    fetchFiles();
  }, []);

  const handleFileClick = (file) => {
    setSelectedFile(file);
    onSelectFile(file);
  };

  return (
    <div>
      <h3>Uploaded Files</h3>
      <ul>
        {files.map((file) => (
          <li key={file}>
            <button
              onClick={() => handleFileClick(file)}
              className={selectedFile === file ? 'selected' : ''}
            >
              {file}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Filevisualizer;
