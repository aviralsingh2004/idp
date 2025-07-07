import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import PredictionCard from './components/PredictionCard'
import FeatureImportanceChart from './components/FeatureImportanceChart'
import TelemetryComparison from './components/TelemetryComparision'
import SimulationDashboard from './components/SimulationDashboard'
import AerodynamicsBackground from './components/AerodynamicsBackground'
import './index.css'
import { motion } from 'framer-motion'

function Home() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="app-container-relative" style={{ marginTop: 60 }}>
        <AerodynamicsBackground />
        <div className="app-center-container">
          <div className="section"><h1>F1 Aero Dashboard</h1></div>
          <div className="section"><PredictionCard/></div>
          <div className="section">
            <h2>Feature Importance</h2>
            <FeatureImportanceChart/>
          </div>
          <TelemetryComparison/>
        </div>
      </div>
    </motion.div>
  )
}

export default function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/simulation" element={<SimulationDashboard />} />
      </Routes>
    </Router>
  )
}
