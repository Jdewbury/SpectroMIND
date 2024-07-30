import React from 'react';

const FileItem = ({ file, isSelected, onSelect, onDelete }) => {
  return (
    <div className="file-item">
      <button
        className={`file-button ${isSelected ? 'selected' : ''}`}
        onClick={() => onSelect(file)}
      >
        {file}
      </button>
      {isSelected && (
        <button
          className="delete-button"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(file);
          }}
        >
          x
        </button>
      )}
    </div>
  );
};

export default FileItem;


