import React from 'react';
import PredictionCard from './components/PredictionCard';
import FeatureImportanceChart from './components/FeatureImportanceChart';
import TelemetryTable from './components/TelemetryTable';

function App() {
  return (
    <div className="App" style={{ padding: 20 }}>
      <h1>F1 Aero Dashboard</h1>
      <PredictionCard />
      <h2>Feature Importance</h2>
      <FeatureImportanceChart />
      <h2>Telemetry Comparison vs Ideal</h2>
      <TelemetryTable />
    </div>
  );
}

export default App;