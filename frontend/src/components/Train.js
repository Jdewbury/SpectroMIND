import React from 'react';

const Upload = ({ handleFileChange, handleUpload, file, error, message, isLoading }) => {
  return (
    <div>
      <h2>Upload Files</h2>
      <input type="file" multiple onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={isLoading}>
        {isLoading ? 'Uploading...' : 'Upload'}
      </button>
      {file && file.length > 0 && (
        <div>
          <p>Selected files:</p>
          <ul>
            {file.map((f, index) => (
              <li key={index}>{f.name}</li>
            ))}
          </ul>
        </div>
      )}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {message && <p style={{ color: 'green' }}>{message}</p>}
    </div>
  );
};

export default Upload;

