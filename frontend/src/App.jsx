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
import Hyperspeed from './components/Hyperspeed/Hyperspeed'

function Home() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
      style={{ 
        minHeight: '100vh', 
        width: '100vw', 
        position: 'relative', 
        overflow: 'hidden', 
        marginTop: 'var(--spacing-3xl)' 
      }}
    >
      <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
        <Hyperspeed
          effectOptions={{
            onSpeedUp: () => { },
            onSlowDown: () => { },
            distortion: 'turbulentDistortion',
            length: 400,
            roadWidth: 10,
            islandWidth: 2,
            lanesPerRoad: 4,
            fov: 90,
            fovSpeedUp: 150,
            speedUp: 2,
            carLightsFade: 0.4,
            totalSideLightSticks: 20,
            lightPairsPerRoadWay: 40,
            shoulderLinesWidthPercentage: 0.05,
            brokenLinesWidthPercentage: 0.1,
            brokenLinesLengthPercentage: 0.5,
            lightStickWidth: [0.12, 0.5],
            lightStickHeight: [1.3, 1.7],
            movingAwaySpeed: [60, 80],
            movingCloserSpeed: [-120, -160],
            carLightsLength: [400 * 0.03, 400 * 0.2],
            carLightsRadius: [0.05, 0.14],
            carWidthPercentage: [0.3, 0.5],
            carShiftX: [-0.8, 0.8],
            carFloorSeparation: [0, 5],
            colors: {
              roadColor: 0x0a001a,
              islandColor: 0x181028,
              background: 0x0a001a,
              shoulderLines: 0x00e5ff,
              brokenLines: 0xff00cc,
              leftCars: [0xff00cc, 0xd856bf, 0x8f00ff],
              rightCars: [0x00e5ff, 0x0ff0fc, 0x324555],
              sticks: 0x00e5ff,
            }
          }}
        />
        <div style={{
          position: 'absolute',
          top: 'var(--spacing-3xl)',
          left: 0,
          width: '100%',
          display: 'flex',
          justifyContent: 'center',
          gap: 'var(--spacing-2xl)',
          pointerEvents: 'none',
          zIndex: 2,
          padding: '0 var(--spacing-lg)'
        }}>
          <div style={{
            background: 'var(--bg-card)',
            backdropFilter: 'blur(20px)',
            borderRadius: 'var(--border-radius-lg)',
            padding: 'var(--spacing-xl)',
            color: 'var(--text-primary)',
            maxWidth: '320px',
            boxShadow: 'var(--shadow-md), var(--shadow-glow)',
            border: '1px solid rgba(195, 0, 47, 0.3)',
            pointerEvents: 'auto'
          }}>
            <h2 style={{ 
              color: 'var(--color-accent)', 
              fontWeight: 700, 
              fontSize: '1.375rem', 
              marginBottom: 'var(--spacing-sm)' 
            }}>F1 Aero Dashboard</h2>
            <p style={{ fontSize: '1rem', color: 'var(--text-secondary)' }}>
              Explore Formula 1 aerodynamics, car telemetry, and AI-powered predictions in a visually immersive dashboard.
            </p>
          </div>
          <div style={{
            background: 'var(--bg-card)',
            backdropFilter: 'blur(20px)',
            borderRadius: 'var(--border-radius-lg)',
            padding: 'var(--spacing-xl)',
            color: 'var(--text-primary)',
            maxWidth: '320px',
            boxShadow: 'var(--shadow-md), var(--shadow-glow)',
            border: '1px solid rgba(195, 0, 47, 0.3)',
            pointerEvents: 'auto'
          }}>
            <h2 style={{ 
              color: 'var(--color-accent)', 
              fontWeight: 700, 
              fontSize: '1.375rem', 
              marginBottom: 'var(--spacing-sm)' 
            }}>AI Insights</h2>
            <p style={{ fontSize: '1rem', color: 'var(--text-secondary)' }}>
              Get real-time predictions and feature importance analysis powered by advanced machine learning models.
            </p>
          </div>
          <div style={{
            background: 'var(--bg-card)',
            backdropFilter: 'blur(20px)',
            borderRadius: 'var(--border-radius-lg)',
            padding: 'var(--spacing-xl)',
            color: 'var(--text-primary)',
            maxWidth: '320px',
            boxShadow: 'var(--shadow-md), var(--shadow-glow)',
            border: '1px solid rgba(195, 0, 47, 0.3)',
            pointerEvents: 'auto'
          }}>
            <h2 style={{ 
              color: 'var(--color-accent)', 
              fontWeight: 700, 
              fontSize: '1.375rem', 
              marginBottom: 'var(--spacing-sm)' 
            }}>Simulation & Telemetry</h2>
            <p style={{ fontSize: '1rem', color: 'var(--text-secondary)' }}>
              Simulate race conditions, compare telemetry, and visualize car performance with interactive charts.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function Dashboard() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
      style={{ marginTop: 'var(--spacing-3xl)' }}
    >
      <div className="app-container-relative">
        <AerodynamicsBackground />
        <div className="app-center-container">
          <div className="section">
            <h1>F1 Aero Dashboard</h1>
          </div>
          <div className="section">
            <PredictionCard/>
          </div>
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
        <Route path="/home" element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/simulation" element={<SimulationDashboard />} />
      </Routes>
    </Router>
  )
}
