import React, { useState } from 'react';

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
    console.log(`Files selected (${setterFunction.name}):`, event.target.files);
    console.log(`Number of files selected (${setterFunction.name}):`, files.length);
  };

  const handleIntervalChange = (event) => {
    const value = event.target.value;
    console.log('Interval input:', value);
    setIntervals(value);
  };

  const handleParamChange = (event) => {
    const file = event.target.files[0]; 
    setParams(file);
  };

  const handleWeightChange = (event) => {
    const file = event.target.files[0];
    setWeights(file);
  };

  const allowedExtensions = ['npy', 'pth'];

  const checkFileType = (file) => {
    const extension = file.name.split('.').pop().toLowerCase();
    return allowedExtensions.includes(extension);
  };
  
  const handleEvaluate = async () => {
    // Clear previous messages
    setError('');
    setMessage('');
    setResults(null);

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
    formData.append('params', params);
    formData.append('weights', weights);
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
      } else {
        console.error('Server returned an error:', response.status, response.statusText);
        const errorData = await response.json();
        setError(`Error: ${errorData.error}`);
      }
    } catch (error) {
      console.error('Fetch error:', error);
      setError(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <h1>Evaluate Model</h1>
      <div>
        <input type="file" accept=".npy" onChange={(e) => handleParamChange(e, setParams)} />
        <label>Params File (.npy)</label>
      </div>
      <div>
        <input type="file" accept=".pth" onChange={(e) => handleWeightChange(e, setWeights)} />
        <label>Weights File (.pth)</label>
      </div>
      <div>
        <input type="file" accept=".npy" multiple onChange={(e) => handleFileChange(e, setDatasets)} />
        <label>Dataset Files (.npy)</label>
      </div>
      <div>
        <input type="file" accept=".npy" multiple onChange={(e) => handleFileChange(e, setLabels)} />
        <label>Label Files (.npy)</label>
      </div>
      <div>
        <input type="text" value={intervals} onChange={handleIntervalChange} />
        <label>Spectra Intervals (comma-separated)</label>
      </div>
      <button onClick={handleEvaluate}>Evaluate</button>
      {isLoading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {message && <p>{message}</p>}
      {results && (
        <div>
          <h2>Results</h2>
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default Evaluate;

