import React from 'react';

const FileItem = ({ file, isSelected, onSelect, onDelete }) => {
  return (
    <div className={`file-item ${isSelected ? 'selected' : ''}`}>
      <button onClick={() => onSelect(file)}>{file}</button>
      <xbutton onClick={() => onDelete(file)}>x</xbutton>
    </div>
  );
};

export default FileItem;
