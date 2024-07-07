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
        <xbutton
          className="delete-button"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(file);
          }}
        >
          x
        </xbutton>
      )}
    </div>
  );
};

export default FileItem;


