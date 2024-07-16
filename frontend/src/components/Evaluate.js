import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Evaluate = () => {
  const [params, setParams] = useState(null);
  const [weights, setWeights] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [labels, setLabels] = useState([]);
  const [intervals, setIntervals] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event, setterFunction) => {
    const files = Array.from(event.target.files);
    setterFunction(files);
  };

  const handleIntervalChange = (event) => {
    setIntervals(event.target.value);
  };

  const handleParamChange = (event) => {
    setParams(event.target.files[0]);
  };

  const handleWeightChange = (event) => {
    setWeights(event.target.files[0]);
  };

  const allowedExtensions = ['npy', 'pth'];

  const checkFileType = (file) => {
    const extension = file.name.split('.').pop().toLowerCase();
    return allowedExtensions.includes(extension);
  };

  const handleEvaluate = async () => {
    console.log('Starting evaluation...');
    if (!params || !weights || datasets.length === 0 || labels.length === 0 || !intervals) {
      setError('Please upload all required files and specify intervals');
      return;
    }
  
    if (datasets.length !== labels.length) {
      setError('Number of dataset files and label files must match');
      return;
    }
  
    const intervalArray = intervals.split(',').map(i => parseInt(i.trim()));
    if (intervalArray.length !== datasets.length) {
      setError('Number of intervals must match number of dataset files');
      return;
    }
  
    if (!checkFileType(params) || !checkFileType(weights)) {
      setError('Invalid file type for params or weights');
      return;
    }
  
    for (let dataset of datasets) {
      if (!checkFileType(dataset)) {
        setError(`Invalid file type for dataset: ${dataset.name}`);
        return;
      }
    }
  
    for (let label of labels) {
      if (!checkFileType(label)) {
        setError(`Invalid file type for label: ${label.name}`);
        return;
      }
    }
  
    const formData = new FormData();
  
    // Read weights file as ArrayBuffer
    const weightsArrayBuffer = await weights.arrayBuffer();
    formData.append('weights', new Blob([weightsArrayBuffer]), weights.name);
  
    formData.append('params', params);
    datasets.forEach((dataset, index) => {
      formData.append(`dataset_${index}`, dataset);
    });
    labels.forEach((label, index) => {
      formData.append(`label_${index}`, label);
    });
    formData.append('intervals', intervals);
  
    setIsLoading(true);
    try {
      console.log('Sending request to server...');
      const response = await fetch('http://localhost:5000/evaluate', {
        method: 'POST',
        body: formData,
      });
  
      console.log('Response received:', response);
  
      if (response.ok) {
        const data = await response.json();
        console.log('Parsed response data:', data);
        setResults(data.results);
        setMessage('Evaluation completed successfully');
        setError('');
      } else {
        console.error('Server returned an error:', response.status, response.statusText);
        const errorData = await response.json();
        setError(`Error: ${errorData.error}`);
        setMessage('');
      }
    } catch (error) {
      console.error('Fetch error:', error);
      setError(`Error: ${error.message}`);
      setMessage('');
    } finally {
      setIsLoading(false);
    }
  };

  const renderConfusionMatrix = (confusionMatrix, classLabels) => {
    if (!confusionMatrix || confusionMatrix.length === 0) return null;

    const maxValue = Math.max(...confusionMatrix.flat());

    return (
      <div className="confusion-matrix">
        <h3>Confusion Matrix</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', margin: '20px 0' }}>
            <thead>
              <tr>
                <th style={{ padding: '8px', border: '1px solid #ddd' }}></th>
                {classLabels.map((label, index) => (
                  <th key={index} style={{ padding: '8px', border: '1px solid #ddd', writingMode: 'vertical-rl', textOrientation: 'mixed' }}>
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.map((row, i) => (
                <tr key={i}>
                  <th style={{ padding: '8px', border: '1px solid #ddd' }}>{classLabels[i]}</th>
                  {row.map((value, j) => (
                    <td
                      key={j}
                      style={{
                        padding: '8px',
                        border: '1px solid #ddd',
                        backgroundColor: `rgba(0, 0, 255, ${value / maxValue})`,
                        color: value / maxValue > 0.5 ? 'white' : 'black',
                        textAlign: 'center'
                      }}
                    >
                      {value.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <p>True labels on rows, predicted labels on columns</p>
        </div>
      </div>
    );
  };

  return (
    <div className="evaluate-container">
      <div className="sidebar">
        <h3>Evaluation Input</h3>
        <div className="input-group">
          <label>Params File (.npy)</label>
          <input type="file" accept=".npy" onChange={handleParamChange} />
        </div>
        <div className="input-group">
          <label>Weights File (.pth)</label>
          <input type="file" accept=".pth" onChange={handleWeightChange} />
        </div>
        <div className="input-group">
          <label>Dataset Files (.npy)</label>
          <input type="file" accept=".npy" multiple onChange={(e) => handleFileChange(e, setDatasets)} />
        </div>
        <div className="input-group">
          <label>Label Files (.npy)</label>
          <input type="file" accept=".npy" multiple onChange={(e) => handleFileChange(e, setLabels)} />
        </div>
        <div className="input-group">
          <label>Spectra Intervals (comma-separated)</label>
          <input type="text" value={intervals} onChange={handleIntervalChange} />
        </div>
        <button onClick={handleEvaluate} disabled={isLoading}>
          {isLoading ? 'Evaluating...' : 'Evaluate'}
        </button>
        {message && (
          <div className="upload-message">
            <p>{message}</p>
          </div>
        )}
      </div>
      <div className="main-content">
        {error && (
          <div className="error-message">
            <h3>Error</h3>
            <p>{error}</p>
          </div>
        )}
        {results && (
          <div className="results-container">
            <h2>Evaluation Results</h2>
            <div className="metrics">
              <div className="metric">
                <h3>Test Loss</h3>
                <p>{results['test-loss'].toFixed(4)}</p>
              </div>
              <div className="metric">
                <h3>Test Accuracy</h3>
                <p>{(results['test-acc'] * 100).toFixed(2)}%</p>
              </div>
              <div className="metric">
                <h3>Inference Time</h3>
                <p>{results['test-time'].toFixed(2)}s</p>
              </div>
            </div>
            {results.confusion_matrix && renderConfusionMatrix(results.confusion_matrix, results.class_labels)}
          </div>
        )}
      </div>
    </div>
  );
};

export default Evaluate;

