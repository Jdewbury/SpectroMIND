import React, { useState } from 'react';
import axios from 'axios';

const modelConfig = {
  resnet: {
    layers: { type: 'number', default: 6, min: 1, max: 50 },
    hidden_size: { type: 'number', default: 100, min: 32, max: 1024 },
    block_size: { type: 'number', default: 2, min: 1, max: 8 },
    activation: { type: 'select', options: ['relu', 'selu', 'gelu'] },
  },
  mlp_flip: {
    depth: { type: 'number', default: 2, min: 1, max: 10 },
    token_dim: { type: 'number', default: 64, min: 16, max: 256 },
    channel_dim: { type: 'number', default: 16, min: 8, max: 128 },
    patch_size: { type: 'number', default: 50, min: 10, max: 100 },
  },
};

const optimizerConfig = {
  Adam: {
    learning_rate: { type: 'number', default: 0.001, min: 0.0001, max: 0.1, step: 0.0001 },
    weight_decay: { type: 'number', default: 0.0005, min: 0, max: 0.1, step: 0.0001 },
  },
  SGD: {
    learning_rate: { type: 'number', default: 0.01, min: 0.0001, max: 0.1, step: 0.0001 },
    momentum: { type: 'number', default: 0.9, min: 0, max: 1, step: 0.1 },
    weight_decay: { type: 'number', default: 0.0005, min: 0, max: 0.1, step: 0.0001 },
  },
  SAM: {
    base_optimizer: { type: 'select', options: ['SGD', 'Adam'] },
    rho: { type: 'number', default: 0.05, min: 0.01, max: 0.1, step: 0.01 },
  },
  ASAM: {
    base_optimizer: { type: 'select', options: ['SGD', 'Adam'] },
    rho: { type: 'number', default: 0.05, min: 0.01, max: 0.1, step: 0.01 },
  },
};

const Train = () => {
  const [model, setModel] = useState('');
  const [optimizer, setOptimizer] = useState('');
  const [parameters, setParameters] = useState({});
  const [dataFolder, setDataFolder] = useState(null);
  const [labelsFolder, setLabelsFolder] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [errorDetails, setErrorDetails] = useState('');

  const handleModelChange = (e) => {
    setModel(e.target.value);
    setParameters({});
  };

  const handleOptimizerChange = (e) => {
    setOptimizer(e.target.value);
    setParameters({});
  };

  const handleParameterChange = (paramName, value) => {
    setParameters(prevParams => ({
      ...prevParams,
      [paramName]: value
    }));
  };

  const handleDataFolderUpload = (e) => {
    setDataFolder(e.target.files);
  };

  const handleLabelsFolderUpload = (e) => {
    setLabelsFolder(e.target.files);
  };

  const handleTrainSubmit = async (e) => {
    e.preventDefault();
    setIsTraining(true);
    setTrainingStatus('Training started...');
    setErrorDetails('');

    const formData = new FormData();
    formData.append('model', model);
    formData.append('optimizer', optimizer);
    formData.append('parameters', JSON.stringify(parameters));
    
    // Append all files from dataFolder
    for (let i = 0; i < dataFolder.length; i++) {
      formData.append('dataFolder', dataFolder[i]);
    }
    
    // Append all files from labelsFolder
    for (let i = 0; i < labelsFolder.length; i++) {
      formData.append('labelsFolder', labelsFolder[i]);
    }

    try {
      const response = await axios.post('http://localhost:5000/api/train', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setTrainingStatus('Training completed successfully!');
      console.log(response.data);
    } catch (error) {
      setTrainingStatus('Error occurred during training.');
      console.error('Training error:', error);
      if (error.response && error.response.data) {
        setErrorDetails(JSON.stringify(error.response.data, null, 2));
      } else {
        setErrorDetails(error.message);
      }
    } finally {
      setIsTraining(false);
    }
  };

  const renderParameterInputs = () => {
    const modelParams = modelConfig[model] || {};
    const optimizerParams = optimizerConfig[optimizer] || {};
    const allParams = { ...modelParams, ...optimizerParams };

    return Object.entries(allParams).map(([paramName, config]) => (
      <div key={paramName}>
        <label>{paramName}:</label>
        {config.type === 'number' ? (
          <input
            type="number"
            value={parameters[paramName] || config.default}
            onChange={(e) => handleParameterChange(paramName, parseFloat(e.target.value))}
            min={config.min}
            max={config.max}
            step={config.step || 1}
          />
        ) : config.type === 'select' ? (
          <select
            value={parameters[paramName] || config.default}
            onChange={(e) => handleParameterChange(paramName, e.target.value)}
          >
            {config.options.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        ) : null}
      </div>
    ));
  };

  return (
    <div className="train-container">
      <h2>Train Model</h2>
      <form onSubmit={handleTrainSubmit}>
        <div>
          <label>Model:</label>
          <select value={model} onChange={handleModelChange}>
            <option value="">Select a model</option>
            <option value="resnet">ResNet</option>
            <option value="mlp_flip">MLP Flip</option>
          </select>
        </div>
        <div>
          <label>Optimizer:</label>
          <select value={optimizer} onChange={handleOptimizerChange}>
            <option value="">Select an optimizer</option>
            <option value="Adam">Adam</option>
            <option value="SGD">SGD</option>
            <option value="SAM">SAM</option>
            <option value="ASAM">ASAM</option>
          </select>
        </div>
        {renderParameterInputs()}
        <div>
          <label>Data Folder:</label>
          <input type="file" accept=".npy" multiple onChange={handleDataFolderUpload} />
        </div>
        <div>
          <label>Labels Folder:</label>
          <input type="file" accept=".npy" multiple onChange={handleLabelsFolderUpload} />
        </div>
        <button type="submit" disabled={isTraining}>
          {isTraining ? 'Training...' : 'Train Model'}
        </button>
      </form>
      {trainingStatus && <p>{trainingStatus}</p>}
      {errorDetails && (
        <div>
          <h3>Error Details:</h3>
          <pre>{errorDetails}</pre>
        </div>
      )}
    </div>
  );
};

export default Train;


