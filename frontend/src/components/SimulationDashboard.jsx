import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend
} from 'recharts';
import Track3D from './Track3D';
import { fetchRawTelemetry, fetchTrackPositions, fetchAeroAnalysis } from '../api';

export default function SimulationDashboard() {
  const [history, setHistory] = useState([]);
  const [track, setTrack] = useState([]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [params, setParams] = useState({
    year: 2023, gp: 'Italian Grand Prix', session: 'Race', driver: 'VER', lap: 1
  });
  const [analyses, setAnalyses] = useState([]);
  const commentaryBoxRef = useRef(null);
  const timerRef = useRef(null);
  const [latestCommentary, setLatestCommentary] = useState('');
  const [lastCommentaryTime, setLastCommentaryTime] = useState(0);

  const start = async () => {
    const telemetry = await fetchRawTelemetry(params);
    const rawTrack = await fetchTrackPositions(params);
    const xs = rawTrack.map(p => p.X);
    const ys = rawTrack.map(p => p.Y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const norm = rawTrack.map(p => ({
      X: ((p.X - minX) / (maxX - minX)) * 1000,
      Y: 1000 - ((p.Y - minY) / (maxY - minY)) * 1000,
      Speed: p.Speed
    }));
    setTrack(norm);
    setHistory(telemetry);
    setStep(0);
    setPlaying(true);
  };

  const toggle = () => setPlaying(p => !p);
  const reset = () => {
    setPlaying(false);
    setStep(0);
    setHistory([]);
    setTrack([]);
    setAnalyses([]);
    setLatestCommentary('');
  };

  // Real-time simulation
  useEffect(() => {
    if (!playing || step >= history.length - 1) return;
    timerRef.current = setTimeout(() => {
      setStep(s => s + 1);
    }, 400);
    return () => clearTimeout(timerRef.current);
  }, [playing, step, history.length]);

  // AI commentary generation
  useEffect(() => {
    if (step > 0 && step % 5 === 0 && Date.now() - lastCommentaryTime > 2000) {
      const currentData = history[step];
      if (currentData) {
        // Add aerodynamics-related parameters to enhance analysis
        const analysisParams = {
          ...currentData,
          Speed_kmph: currentData.speed || 120,
          B_Ramp_Angle: Math.max(1, currentData.wingAngle / 6),
          B_Diffusor_Angle: Math.max(1, currentData.bodyFlex / 2),
          A_Car_Length: 30,
          Reynolds_Number: 2.5e7,
          Body_Surface_Ratio: 0.2,
          Greenhouse_Ratio: 1.5,
          Combined_Inclination: 3,
          Aerodynamic_Blend_Factor: Math.max(5, currentData.wingAngle * 0.8),
          Speed_Diffusor_Product: (currentData.speed || 120) * Math.max(1, currentData.bodyFlex / 2),
          Length_Width_Ratio: 1.2
        };

        fetchAeroAnalysis(analysisParams).then(res => {
          console.log("Aero analysis response:", res);
          if (res && res.analysis) {
            setLatestCommentary(res.analysis);
            setAnalyses(prev => {
              const newAnalyses = [...prev];
              newAnalyses[step] = res.analysis;
              return newAnalyses;
            });
            setLastCommentaryTime(Date.now());
          }
        }).catch(console.error);
      }
    }
  }, [step, history, lastCommentaryTime]);

  const live = history[step] || { lapTime: 0, wingAngle: 0, bodyFlex: 0 };

  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
      style={{ 
        marginTop: 'var(--spacing-3xl)', 
        padding: 'var(--spacing-xl)', 
        maxWidth: '1400px', 
        margin: '0 auto' 
      }}
    >
      <div className="simulation-dashboard-tile" style={{ 
        background: 'var(--bg-card)', 
        backdropFilter: 'blur(20px)', 
        borderRadius: 'var(--border-radius-lg)', 
        padding: 'var(--spacing-xl)', 
        border: '1px solid rgba(255, 255, 255, 0.1)' 
      }}>
        <h1 style={{ 
          textAlign: 'center', 
          marginBottom: 'var(--spacing-xl)', 
          color: 'var(--color-secondary)',
          fontSize: 'clamp(1.5rem, 4vw, 2.5rem)'
        }}>
          F1 Real-Time Simulation
        </h1>

        {/* AI Commentary Section */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 1fr', 
          gap: 'var(--spacing-lg)', 
          marginBottom: 'var(--spacing-xl)' 
        }}>
          {/* Live Commentary: show latest available analysis up to current step */}
          <div style={{
            background: 'rgba(255, 235, 59, 0.05)',
            border: '1px solid rgba(255, 235, 59, 0.2)',
            borderRadius: 'var(--border-radius-md)',
            padding: 'var(--spacing-lg)',
            maxHeight: '200px',
            overflowY: 'auto'
          }}>
            <h3 style={{ 
              color: 'var(--color-accent)', 
              marginBottom: 'var(--spacing-md)',
              fontSize: '1.1rem'
            }}>
              Live Commentary
            </h3>
            <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              {
                (() => {
                  // Find the latest available analysis up to the current step
                  let latest = null;
                  for (let i = step; i >= 0; i--) {
                    if (analyses[i]) {
                      latest = analyses[i];
                      break;
                    }
                  }
                  return latest || 'No analysis available yet. Start simulation and wait for steps.';
                })()
              }
            </p>
          </div>

          {/* Analysis History: only show steps with actual analysis */}
          <div style={{
            background: 'rgba(0, 229, 255, 0.05)',
            border: '1px solid rgba(0, 229, 255, 0.2)',
            borderRadius: 'var(--border-radius-md)',
            padding: 'var(--spacing-lg)',
            maxHeight: '200px',
            overflowY: 'auto'
          }}>
            <h3 style={{ 
              color: 'var(--color-secondary)', 
              marginBottom: 'var(--spacing-md)',
              fontSize: '1.1rem'
            }}>
              Analysis History
            </h3>
            <div>
              {
                analyses.filter(Boolean).length === 0 ? (
                  <div style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>No analysis yet. Run the simulation and wait for steps.</div>
                ) : (
                  analyses.map((analysis, i) => (
                    analysis ? (
                      <div key={i} style={{
                        padding: 'var(--spacing-sm)',
                        marginBottom: 'var(--spacing-xs)',
                        background: i === step ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                        borderRadius: 'var(--border-radius-sm)',
                        fontSize: '0.875rem',
                        color: i === step ? 'var(--text-primary)' : 'var(--text-muted)'
                      }}>
                        <strong>Step {i + 1}:</strong> {analysis.substring(0, 100) + (analysis.length > 100 ? '...' : '')}
                      </div>
                    ) : null
                  ))
                )
              }
            </div>
          </div>
        </div>

        {/* Parameters Section */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
          gap: 'var(--spacing-md)', 
          marginBottom: 'var(--spacing-lg)' 
        }}>
          <div className="form-group">
            <label>Year</label>
            <input 
              type="number" 
              min={2018} 
              max={2024} 
              value={params.year} 
              onChange={e => setParams({ ...params, year: +e.target.value })} 
            />
          </div>
          <div className="form-group">
            <label>Grand Prix</label>
            <input 
              type="text" 
              value={params.gp} 
              onChange={e => setParams({ ...params, gp: e.target.value })} 
            />
          </div>
          <div className="form-group">
            <label>Driver</label>
            <input 
              type="text" 
              value={params.driver} 
              onChange={e => setParams({ ...params, driver: e.target.value })} 
            />
          </div>
          <div className="form-group">
            <label>Session</label>
            <select 
              value={params.session} 
              onChange={e => setParams({ ...params, session: e.target.value })}
            >
              <option>Practice</option>
              <option>Qualifying</option>
              <option>Race</option>
            </select>
          </div>
          <div className="form-group">
            <label>Lap</label>
            <input 
              type="number" 
              min={1} 
              value={params.lap} 
              onChange={e => setParams({ ...params, lap: +e.target.value })} 
            />
          </div>
        </div>

        {/* Control Buttons */}
        <div style={{ 
          display: 'flex', 
          gap: 'var(--spacing-md)', 
          marginBottom: 'var(--spacing-xl)', 
          justifyContent: 'center',
          flexWrap: 'wrap' 
        }}>
          <button onClick={start} disabled={playing}>
            Start Simulation
          </button>
          <button onClick={toggle} disabled={!history.length}>
            {playing ? 'Pause' : 'Resume'}
          </button>
          <button onClick={reset} disabled={!history.length}>
            Reset
          </button>
        </div>

        {/* Live Metrics */}
        <div style={{ 
          background: 'rgba(255, 255, 255, 0.03)', 
          padding: 'var(--spacing-lg)', 
          borderRadius: 'var(--border-radius-md)', 
          marginBottom: 'var(--spacing-xl)' 
        }}>
          <h3 style={{ 
            color: 'var(--color-secondary)', 
            marginBottom: 'var(--spacing-md)',
            textAlign: 'center'
          }}>
            Live Telemetry
          </h3>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
            gap: 'var(--spacing-lg)',
            textAlign: 'center'
          }}>
            <div style={{
              padding: 'var(--spacing-md)',
              background: 'rgba(25, 118, 210, 0.1)',
              borderRadius: 'var(--border-radius-sm)',
              border: '1px solid rgba(25, 118, 210, 0.3)'
            }}>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Lap Time</div>
              <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--color-primary)' }}>
                {live.lapTime.toFixed(2)}s
              </div>
            </div>
            <div style={{
              padding: 'var(--spacing-md)',
              background: 'rgba(0, 229, 255, 0.1)',
              borderRadius: 'var(--border-radius-sm)',
              border: '1px solid rgba(0, 229, 255, 0.3)'
            }}>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Wing Angle</div>
              <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--color-secondary)' }}>
                {live.wingAngle.toFixed(1)}mm
              </div>
            </div>
            <div style={{
              padding: 'var(--spacing-md)',
              background: 'rgba(255, 152, 0, 0.1)',
              borderRadius: 'var(--border-radius-sm)',
              border: '1px solid rgba(255, 152, 0, 0.3)'
            }}>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Body Flex</div>
              <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--color-warning)' }}>
                {live.bodyFlex.toFixed(1)}mm
              </div>
            </div>
          </div>
        </div>

        {/* Visualization Section */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 400px', 
          gap: 'var(--spacing-xl)', 
          alignItems: 'start' 
        }}>
          <div>
            <h3 style={{ 
              color: 'var(--text-primary)', 
              marginBottom: 'var(--spacing-lg)',
              textAlign: 'center'
            }}>
              Performance Chart
            </h3>
            <LineChart width={800} height={400} data={history.slice(0, step + 1)}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="lapTime" 
                tick={{ fill: 'var(--text-secondary)' }} 
                label={{ value: 'Time (s)', fill: 'var(--text-secondary)', position: 'insideBottom' }} 
              />
              <YAxis 
                yAxisId="left" 
                tick={{ fill: 'var(--color-primary)' }} 
                label={{ value: 'Wing Angle (mm)', angle: -90, fill: 'var(--color-primary)' }} 
              />
              <YAxis 
                yAxisId="right" 
                orientation="right" 
                tick={{ fill: 'var(--color-secondary)' }} 
                label={{ value: 'Body Flex (mm)', angle: 90, fill: 'var(--color-secondary)' }} 
              />
              <Tooltip 
                contentStyle={{ 
                  background: 'var(--bg-card)', 
                  color: 'var(--text-primary)', 
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 'var(--border-radius-sm)'
                }} 
              />
              <Legend />
              <Line 
                yAxisId="left" 
                dataKey="wingAngle" 
                stroke="var(--color-primary)" 
                dot={false} 
                strokeWidth={2}
              />
              <Line 
                yAxisId="right" 
                dataKey="bodyFlex" 
                stroke="var(--color-secondary)" 
                dot={false} 
                strokeWidth={2}
              />
            </LineChart>
          </div>
          
          <div>
            <h3 style={{ 
              color: 'var(--text-primary)', 
              marginBottom: 'var(--spacing-lg)',
              textAlign: 'center'
            }}>
              3D Track Visualization
            </h3>
            <Track3D
              year={params.year}
              gp={params.gp}
              session={params.session}
              lap={params.lap}
              driver="VER"
              syncIndex={step}  
            />
          </div>
        </div>
      </div>
    </motion.div>
  );
}
