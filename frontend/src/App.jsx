import React from 'react'
import PredictionCard from './components/PredictionCard'
import FeatureImportanceChart from './components/FeatureImportanceChart'
import TelemetryComparison from './components/TelemetryComparision'
import SimulationDashboard   from './components/SimulationDashboard'
import AerodynamicsBackground from './components/AerodynamicsBackground'
import './index.css'

export default function App() {
  return (
    <div className="app-container-relative">
      <AerodynamicsBackground />
      <div className="app-center-container">
        <div className="section"><h1>F1 Aero Dashboard</h1></div>
        <div className="section"><PredictionCard/></div>
        <div className="section">
          <h2>Feature Importance</h2>
          <FeatureImportanceChart/>
        </div>
        <TelemetryComparison/>
        <div className="section">
          <h2>Simulation Dashboard</h2>
          <SimulationDashboard/>
        </div>
      </div>
    </div>
  )
}
