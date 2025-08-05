import React from 'react';


export default function Scanner({ onBack }) {
  return (
    <div className="scanner-interface">
      <h2>Room Scanner</h2>
      <div className="scanner-placeholder"></div>
      <button onClick={onBack} className="back-button">
        Back to Preferences
      </button>
    </div>
  );
}