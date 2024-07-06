import React from 'react';

const Evaluate = ({ handleEvaluateModel, error, message, isLoading }) => {
  return (
    <div>
      <h2>Evaluate Model</h2>
      <button onClick={handleEvaluateModel} disabled={isLoading}>
        {isLoading ? 'Evaluating...' : 'Evaluate'}
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {message && <p style={{ color: 'green' }}>{message}</p>}
    </div>
  );
};

export default Evaluate;
