import React, { useState, useEffect } from 'react';
import axios from 'axios';

const generalConfig = {
  epochs: { type: 'number', default: 200, min: 1, max: 1000 },
  batch_size: { type: 'number', default: 16, min: 1, max: 256 },
  learning_rate: { type: 'number', default: 0.001, min: 0.0001, max: 0.1, step: 0.0001 },
  in_channels: { type: 'number', default: 64, min: 1, max: 256 },
  num_classes: { type: 'number', default: 5, min: 2, max: 1000 },
  input_dim: { type: 'number', default: 1000, min: 100, max: 10000 },
  label_smoothing: { type: 'number', default: 0.1, min: 0, max: 1, step: 0.01 },
  seed: { type: 'number', default: 42, min: 0, max: 9999 },
  shuffle: { type: 'boolean', default: true },
  save: { type: 'boolean', default: false },
  spectra_interval: { type: 'text', default: '400,100' },
  train_split: { type: 'number', default: 0.7, min: 0.1, max: 0.9, step: 0.01 },
  test_split: { type: 'number', default: 0.15, min: 0.1, max: 0.9, step: 0.01 },
};

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
    weight_decay: { type: 'number', default: 0.0005, min: 0, max: 0.1, step: 0.0001 },
  },
  SGD: {
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

const schedulerConfig = {
  type: 'select',
  options: ['step', 'cosine'],
  default: 'step',
};

const Train = ({ 
  isTraining, 
  setIsTraining, 
  trainingStatus, 
  setTrainingStatus, 
  trainingProgress, 
  setTrainingProgress, 
  errorDetails, 
  setErrorDetails 
}) => {
  const [model, setModel] = useState('');
  const [optimizer, setOptimizer] = useState('');
  const [parameters, setParameters] = useState({});
  const [dataFolder, setDataFolder] = useState(null);
  const [labelsFolder, setLabelsFolder] = useState(null);
  const [trainingResults, setTrainingResults] = useState(null);
  const [saveDirectory, setSaveDirectory] = useState(null);
  const [activeTab, setActiveTab] = useState('general');

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

  const validateInputs = () => {
    if (!model) return 'Please select a model.';
    if (!optimizer) return 'Please select an optimizer.';
    if (!dataFolder || dataFolder.length === 0) return 'Please upload data files.';
    if (!labelsFolder || labelsFolder.length === 0) return 'Please upload label files.';
    return null;
  };

  const handleTrainSubmit = async (e) => {
    e.preventDefault();
    setErrorDetails('');
    
    if (isTraining) {
      setErrorDetails('A training session is already in progress. Please wait for it to complete or stop it before starting a new one.');
      return;
    }

    const validationError = validateInputs();
    if (validationError) {
      setErrorDetails(validationError);
      return;
    }

    setIsTraining(true);
    setTrainingStatus('Preparing to start training...');
    setTrainingProgress(0);

    const allParams = {
      ...Object.keys(generalConfig).reduce((acc, key) => {
        acc[key] = parameters[key] !== undefined ? parameters[key] : generalConfig[key].default;
        return acc;
      }, {}),
      ...(modelConfig[model] ? Object.keys(modelConfig[model]).reduce((acc, key) => {
        acc[key] = parameters[key] !== undefined ? parameters[key] : modelConfig[model][key].default;
        return acc;
      }, {}) : {}),
      ...(optimizerConfig[optimizer] ? Object.keys(optimizerConfig[optimizer]).reduce((acc, key) => {
        acc[key] = parameters[key] !== undefined ? parameters[key] : optimizerConfig[optimizer][key].default;
        return acc;
      }, {}) : {}),
      scheduler: parameters.scheduler || schedulerConfig.default,
    };

    const formData = new FormData();
    formData.append('model', model);
    formData.append('optimizer', optimizer);
    formData.append('parameters', JSON.stringify(allParams));
    
    Array.from(dataFolder).forEach(file => {
      formData.append('dataFolder', file);
    });
    
    Array.from(labelsFolder).forEach(file => {
      formData.append('labelsFolder', file);
    });

    try {
      const response = await fetch('http://localhost:5000/api/train', {
        method: 'POST',
        body: formData,
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              handleTrainingUpdate(data);
            } catch (parseError) {
              console.error('Error parsing server response:', parseError, 'Raw data:', line.substring(6));
            }
          }
        }
      }

    } catch (error) {
      setIsTraining(false);
      setTrainingStatus('Error occurred during training.');
      console.error('Training error:', error);
      setErrorDetails(error.message || 'An unknown error occurred');
    }
  };

  const handleTrainingUpdate = (data) => {
    switch (data.status) {
      case 'error':
        setIsTraining(false);
        setTrainingStatus('Error occurred during training.');
        setErrorDetails(data.message);
        break;
      case 'progress':
        const progress = Math.round((data.epoch / data.total) * 100);
        setTrainingProgress(progress);
        setTrainingStatus(`Training progress: ${progress}%`);
        break;
      case 'stopped':
        setIsTraining(false);
        setTrainingStatus('Training stopped by user.');
        break;
      case 'completed':
        setIsTraining(false);
        setTrainingStatus('Training completed successfully.');
        setTrainingResults(data.scores);
        setSaveDirectory(data.save_directory);
        break;
      default:
        console.warn('Unknown status received:', data.status);
    }
  };

  const handleStopTraining = async () => {
    try {
      await axios.post('http://localhost:5000/api/stop-training');
      setTrainingStatus('Stopping training...');
    } catch (error) {
      console.error('Error stopping training:', error);
      setErrorDetails('Failed to stop training. ' + (error.response?.data?.error || error.message));
    }
  };

  const renderParameterInputs = (config) => {
    return Object.entries(config).map(([paramName, paramConfig]) => (
      <div key={paramName} className="parameter-input">
        <label>{paramName}:</label>
        {paramConfig.type === 'number' ? (
          <input
            type="number"
            value={parameters[paramName] !== undefined ? parameters[paramName] : paramConfig.default}
            onChange={(e) => handleParameterChange(paramName, parseFloat(e.target.value))}
            min={paramConfig.min}
            max={paramConfig.max}
            step={paramConfig.step || 1}
          />
        ) : paramConfig.type === 'select' ? (
          <select
            value={parameters[paramName] !== undefined ? parameters[paramName] : paramConfig.default}
            onChange={(e) => handleParameterChange(paramName, e.target.value)}
          >
            {paramConfig.options.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        ) : paramConfig.type === 'boolean' ? (
          <input
            type="checkbox"
            checked={parameters[paramName] !== undefined ? parameters[paramName] : paramConfig.default}
            onChange={(e) => handleParameterChange(paramName, e.target.checked)}
          />
        ) : paramConfig.type === 'text' ? (
          <input
            type="text"
            value={parameters[paramName] !== undefined ? parameters[paramName] : paramConfig.default}
            onChange={(e) => handleParameterChange(paramName, e.target.value)}
          />
        ) : null}
      </div>
    ));
  };

  return (
    <div className="train-container">
      <div className="sidebar">
        <h3>Training Input</h3>
        <form onSubmit={handleTrainSubmit}>
          <div className="input-group">
            <label>Model:</label>
            <select value={model} onChange={handleModelChange}>
              <option value="">Select a model</option>
              <option value="resnet">ResNet</option>
              <option value="mlp_flip">MLP Flip</option>
            </select>
          </div>
          <div className="input-group">
            <label>Optimizer:</label>
            <select value={optimizer} onChange={handleOptimizerChange}>
              <option value="">Select an optimizer</option>
              <option value="Adam">Adam</option>
              <option value="SGD">SGD</option>
              <option value="SAM">SAM</option>
              <option value="ASAM">ASAM</option>
            </select>
          </div>
          <div className="input-group">
            <label>Data Folder:</label>
            <input type="file" accept=".npy" multiple onChange={handleDataFolderUpload} />
          </div>
          <div className="input-group">
            <label>Labels Folder:</label>
            <input type="file" accept=".npy" multiple onChange={handleLabelsFolderUpload} />
          </div>
          <button type="submit" disabled={isTraining}>
            {isTraining ? 'Training...' : 'Train Model'}
          </button>
          {isTraining && (
            <button type="button" onClick={handleStopTraining}>
              Stop Training
            </button>
          )}
        </form>
        {trainingStatus && <p className="status-message">{trainingStatus}</p>}
        {isTraining && (
          <div className="training-progress-container">
            <div className="training-progress-bar" style={{ width: `${trainingProgress}%` }}></div>
            <div className="training-progress-text">{`Training Progress: ${Math.round(trainingProgress)}%`}</div>
          </div>
        )}
        {errorDetails && (
          <div className="error-message">
            <h3>Error</h3>
            <p>{errorDetails}</p>
          </div>
        )}
      </div>
      <div className="main-content">
        <div className="parameter-tabs">
          <span 
            className={`tab ${activeTab === 'general' ? 'active' : ''}`}
            onClick={() => setActiveTab('general')}
          >
            General
          </span>
          <span 
            className={`tab ${activeTab === 'scheduler' ? 'active' : ''}`}
            onClick={() => setActiveTab('scheduler')}
          >
            Scheduler
          </span>
          {model && (
            <span 
              className={`tab ${activeTab === 'model' ? 'active' : ''}`}
              onClick={() => setActiveTab('model')}
            >
              {model.toUpperCase()}
            </span>
          )}
          {optimizer && (
            <span 
              className={`tab ${activeTab === 'optimizer' ? 'active' : ''}`}
              onClick={() => setActiveTab('optimizer')}
            >
              {optimizer}
            </span>
          )}
        </div>
        <div className="parameter-content">
          {activeTab === 'general' && (
            <div>
              <h3>General Parameters</h3>
              {renderParameterInputs(generalConfig)}
            </div>
          )}
          {activeTab === 'scheduler' && (
            <div>
              <h3>Scheduler</h3>
              {renderParameterInputs({scheduler: schedulerConfig})}
            </div>
          )}
          {activeTab === 'model' && model && (
            <div>
              <h3>{model.toUpperCase()} Parameters</h3>
              {renderParameterInputs(modelConfig[model])}
            </div>
          )}
          {activeTab === 'optimizer' && optimizer && (
            <div>
              <h3>{optimizer} Parameters</h3>
              {renderParameterInputs(optimizerConfig[optimizer])}
            </div>
          )}
        </div>
        {trainingResults && (
          <div className="training-results">
            <h3>Training Results</h3>
            <pre>{JSON.stringify(trainingResults, null, 2)}</pre>
            {saveDirectory ? (
              <p>Results saved in: {saveDirectory}</p>
            ) : (
              <p>Results were not saved (save option was not selected).</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Train;


