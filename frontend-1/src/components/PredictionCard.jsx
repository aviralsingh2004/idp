import React from 'react';

function PredictionCard() {
  return (
    <div className="prediction-card" style={{ border: '1px solid #ccc', padding: 10, borderRadius: 5 }}>
      <h3>Predicted Performance</h3>
      <p>Lap Time: 1:30.123</p>
      <p>Speed: 220 km/h</p>
      <p>Position: 1</p>
    </div>
  );
}

export default PredictionCard;