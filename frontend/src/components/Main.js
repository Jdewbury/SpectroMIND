import React, { useState } from 'react';
import Preprocess from './Preprocess';
import Train from './Train';
import Evaluate from './Evaluate';

const Main = () => {
  const [activeTab, setActiveTab] = useState('preprocess');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [errorDetails, setErrorDetails] = useState('');

  const renderTab = () => {
    switch (activeTab) {
      case 'preprocess':
        return <Preprocess />;
      case 'train':
        return <Train 
          isTraining={isTraining}
          setIsTraining={setIsTraining}
          trainingStatus={trainingStatus}
          setTrainingStatus={setTrainingStatus}
          trainingProgress={trainingProgress}
          setTrainingProgress={setTrainingProgress}
          errorDetails={errorDetails}
          setErrorDetails={setErrorDetails}
        />;
      case 'evaluate':
        return <Evaluate />;
      default:
        return null;
    }
  };

  return (
    <div>
      <nav className="tabs">
        <button
          className={`tab-button ${activeTab === 'preprocess' ? 'active' : ''}`}
          onClick={() => setActiveTab('preprocess')}
        >
          Preprocess
        </button>
        <button
          className={`tab-button ${activeTab === 'train' ? 'active' : ''}`}
          onClick={() => setActiveTab('train')}
        >
          Train
        </button>
        <button
          className={`tab-button ${activeTab === 'evaluate' ? 'active' : ''}`}
          onClick={() => setActiveTab('evaluate')}
        >
          Evaluate
        </button>
      </nav>
      <div className="tab-content">
        {renderTab()}
      </div>
    </div>
  );
};

export default Main;


