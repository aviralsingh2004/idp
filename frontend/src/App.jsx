// src/App.jsx
import React, { useEffect, useState } from "react";
import PredictionCard from "./components/PredictionCard";
import FeatureImportanceChart from "./components/FeatureImportanceChart";
import TelemetryTable from "./components/TelemetryTable";
import AerodynamicsBackground from "./components/AerodynamicsBackground";
import { getTelemetryComparison } from "./api";
import "./index.css";

function App() {
  const [telemetryRows, setTelemetryRows] = useState([]);

  useEffect(() => {
    getTelemetryComparison().then((data) => {
      if (Array.isArray(data)) setTelemetryRows(data);
    });
  }, []);

  return (
    // The main app container should be relative to position the background absolutely within it
    <div className="app-container-relative">
      <AerodynamicsBackground /> {/* Render the 3D background component here */}

      <div className="app-center-container">
        <div className="section">
          <h1>F1 Aero Dashboard</h1>
        </div>
        <div className="section">
          <PredictionCard />
        </div>
        <div className="section">
          <h2>Feature Importance</h2>
          <FeatureImportanceChart />
        </div>
        <div className="section">
          <h2>Telemetry Comparison vs Ideal</h2>
          <TelemetryTable rows={telemetryRows} />
        </div>
      </div>
    </div>
  );
}

export default App;