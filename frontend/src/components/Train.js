import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000';

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
  const [scheduler, setScheduler] = useState('');
  const [parameters, setParameters] = useState({});
  const [dataFolder, setDataFolder] = useState(null);
  const [labelsFolder, setLabelsFolder] = useState(null);
  const [trainingResults, setTrainingResults] = useState(null);
  const [saveDirectory, setSaveDirectory] = useState(null);
  const [activeTab, setActiveTab] = useState('general');
  const [backendURL, setBackendURL] = useState(API_BASE_URL);
  const [colabURL, setColabURL] = useState('');
  const [isUsingColab, setIsUsingColab] = useState(false);
  const [config, setConfig] = useState({});

  useEffect(() => {
    fetchConfig();
  }, [backendURL]);

  const fetchConfig = async () => {
    try {
      const response = await axios.get(`${backendURL}/get-train-config`);
      setConfig(response.data);
    } catch (error) {
      console.error('Error fetching train config:', error);
      setErrorDetails('Failed to fetch training configuration. Please try again.');
    }
  };

  const handleModelChange = (e) => {
    const selectedModel = e.target.value;
    setModel(selectedModel);
    setParameters(prevParams => {
      const newParams = { ...prevParams };
      Object.keys(config.MODEL_CONFIG[selectedModel] || {}).forEach(key => {
        newParams[key] = config.MODEL_CONFIG[selectedModel][key].default;
      });
      return newParams;
    });
  };

  const handleOptimizerChange = (e) => {
    const selectedOptimizer = e.target.value;
    setOptimizer(selectedOptimizer);
    setParameters(prevParams => {
      const newParams = { ...prevParams };
      Object.keys(config.OPTIMIZER_CONFIG[selectedOptimizer] || {}).forEach(key => {
        newParams[key] = config.OPTIMIZER_CONFIG[selectedOptimizer][key].default;
      });
      return newParams;
    });
  };

  const handleSchedulerChange = (e) => {
    const selectedScheduler = e.target.value;
    setScheduler(selectedScheduler);
    setParameters(prevParams => {
      const newParams = { ...prevParams };
      Object.keys(config.SCHEDULER_CONFIG[selectedScheduler] || {}).forEach(key => {
        newParams[key] = config.SCHEDULER_CONFIG[selectedScheduler][key].default;
      });
      return newParams;
    });
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
    if (!scheduler) return 'Please select a scheduler.';
    if (!dataFolder || dataFolder.length === 0) return 'Please upload data files.';
    if (!labelsFolder || labelsFolder.length === 0) return 'Please upload label files.';
    return null;
  };

  const handleColabURLSubmit = async (e) => {
    e.preventDefault();
    try {
      await axios.get(`${colabURL}/health-check`);
      setBackendURL(colabURL);
      setIsUsingColab(true);
      fetchConfig();
    } catch (error) {
      console.error('Error connecting to Colab URL:', error);
      alert('Failed to connect to the provided Colab URL. Please check the URL and try again.');
    }
  };

  const handleSwitchToLocal = () => {
    setBackendURL(API_BASE_URL);
    setIsUsingColab(false);
    fetchConfig();
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
      ...Object.keys(config.GENERAL_CONFIG).reduce((acc, key) => {
        acc[key] = parameters[key] !== undefined ? parameters[key] : config.GENERAL_CONFIG[key].default;
        return acc;
      }, {}),
      ...(config.MODEL_CONFIG[model] ? Object.keys(config.MODEL_CONFIG[model]).reduce((acc, key) => {
        acc[key] = parameters[key] !== undefined ? parameters[key] : config.MODEL_CONFIG[model][key].default;
        return acc;
      }, {}) : {}),
      ...(config.OPTIMIZER_CONFIG[optimizer] ? Object.keys(config.OPTIMIZER_CONFIG[optimizer]).reduce((acc, key) => {
        acc[key] = parameters[key] !== undefined ? parameters[key] : config.OPTIMIZER_CONFIG[optimizer][key].default;
        return acc;
      }, {}) : {}),
      ...(config.SCHEDULER_CONFIG[scheduler] ? Object.keys(config.SCHEDULER_CONFIG[scheduler]).reduce((acc, key) => {
        acc[key] = parameters[key] !== undefined ? parameters[key] : config.SCHEDULER_CONFIG[scheduler][key].default;
        return acc;
      }, {}) : {}),
    };

    const formData = new FormData();
    formData.append('model', model);
    formData.append('optimizer', optimizer);
    formData.append('scheduler', scheduler);
    formData.append('parameters', JSON.stringify(allParams));
    
    Array.from(dataFolder).forEach(file => {
      formData.append('dataFolder', file);
    });
    
    Array.from(labelsFolder).forEach(file => {
      formData.append('labelsFolder', file);
    });

    try {
      const response = await fetch(`${backendURL}/api/train`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      if (!response.body) {
        throw new Error('No response body');
      }
  
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
        setTrainingStatus(`Training progress: ${progress}% (Loss: ${data.loss.toFixed(4)}, Accuracy: ${(data.accuracy * 100).toFixed(2)}%)`);
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
      await axios.post(`${backendURL}/api/stop-training`);
      setTrainingStatus('Stopping training...');
    } catch (error) {
      console.error('Error stopping training:', error);
      setErrorDetails('Failed to stop training. ' + (error.response?.data?.error || error.message));
    }
  };

  const renderParameterInputs = (configSection) => {
    return Object.entries(configSection).map(([paramName, paramConfig]) => (
      <div key={paramName} className="parameter-input">
        <label>{paramName}:</label>
        {paramConfig.type === 'int' || paramConfig.type === 'float' ? (
          <input
            type="number"
            value={parameters[paramName] !== undefined ? parameters[paramName] : paramConfig.default}
            onChange={(e) => handleParameterChange(paramName, paramConfig.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
            min={paramConfig.min}
            max={paramConfig.max}
            step={paramConfig.step || (paramConfig.type === 'int' ? 1 : 0.01)}
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
        <div className="backend-selector">
        {isUsingColab ? (
          <div>
            <label>Connected to Colab: {backendURL}</label>
            <button onClick={handleSwitchToLocal} className="backend" >Switch to Local Backend</button>
          </div>
        ) : (
          <form onSubmit={handleColabURLSubmit}>
            <input
              type="url"
              value={colabURL}
              onChange={(e) => setColabURL(e.target.value)}
              placeholder="Enter Colab backend URL"
              required
            />
            <button type="submit" className="backend" >Connect to Colab</button>
          </form>
        )}
      </div>
      <form onSubmit={handleTrainSubmit}>
          <div className="input-group">
            <label>Model:</label>
            <select value={model} onChange={handleModelChange}>
              <option value="">Select a model</option>
              {Object.keys(config.MODEL_CONFIG || {}).map(modelName => (
                <option key={modelName} value={modelName}>{modelName}</option>
              ))}
            </select>
          </div>
          <div className="input-group">
            <label>Optimizer:</label>
            <select value={optimizer} onChange={handleOptimizerChange}>
              <option value="">Select an optimizer</option>
              {Object.keys(config.OPTIMIZER_CONFIG || {}).map(optimizerName => (
                <option key={optimizerName} value={optimizerName}>{optimizerName}</option>
              ))}
            </select>
          </div>
          <div className="input-group">
            <label>Scheduler:</label>
            <select value={scheduler} onChange={handleSchedulerChange}>
              <option value="">Select a scheduler</option>
              {config.SCHEDULER_CONFIG && Object.keys(config.SCHEDULER_CONFIG).map(schedulerName => (
                <option key={schedulerName} value={schedulerName}>{schedulerName}</option>
              ))}
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
          {scheduler && (
            <span 
              className={`tab ${activeTab === 'scheduler' ? 'active' : ''}`}
              onClick={() => setActiveTab('scheduler')}
            >
              Scheduler
            </span>
          )}
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
          {activeTab === 'general' && config.GENERAL_CONFIG && (
            <div>
              <h3>General Parameters</h3>
              {renderParameterInputs(config.GENERAL_CONFIG)}
            </div>
          )}
          {activeTab === 'scheduler' && scheduler && config.SCHEDULER_CONFIG && config.SCHEDULER_CONFIG[scheduler] && (
            <div>
              <h3>{scheduler} Parameters</h3>
              {renderParameterInputs(config.SCHEDULER_CONFIG[scheduler])}
            </div>
          )}
          {activeTab === 'model' && model && config.MODEL_CONFIG && config.MODEL_CONFIG[model] && (
            <div>
              <h3>{model} Parameters</h3>
              {renderParameterInputs(config.MODEL_CONFIG[model])}
            </div>
          )}
          {activeTab === 'optimizer' && optimizer && config.OPTIMIZER_CONFIG && config.OPTIMIZER_CONFIG[optimizer] &&(
            <div>
              <h3>{optimizer} Parameters</h3>
              {renderParameterInputs(config.OPTIMIZER_CONFIG[optimizer])}
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


