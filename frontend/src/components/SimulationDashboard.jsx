import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend
} from 'recharts'
import { fetchRawTelemetry } from '../api'

export default function SimulationDashboard() {
  const [history, setHistory] = useState([])
  const [step, setStep]       = useState(0)
  const [playing, setPlaying] = useState(false)
  const [params, setParams]   = useState({
    year:2023, gp:'Italian Grand Prix', session:'Race', lap:1
  })
  const timerRef = useRef(null)

  const start = async () => {
    const data = await fetchRawTelemetry(params)
    setHistory(data)
    setStep(0)
    setPlaying(true)
  }

  const toggle = () => setPlaying(p => !p)
  const reset  = () => {
    setPlaying(false)
    setStep(0)
    setHistory([])
  }

  // advance frame every 100ms
  useEffect(() => {
    if (playing && history.length) {
      timerRef.current = setInterval(() => {
        setStep(i => i+1 < history.length ? i+1 : (clearInterval(timerRef.current), i))
      }, 100)
    } else {
      clearInterval(timerRef.current)
    }
    return () => clearInterval(timerRef.current)
  }, [playing, history])

  const live = history[step] || { lapTime:0, wingAngle:0, bodyFlex:0 }

  return (
    <motion.div
      className="simulation-dashboard-tile"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, ease: 'easeOut' }}
      style={{
        width: '100%',
        maxWidth: 1000,
        margin: '56px auto 0 auto',
        background: '#232526',
        borderRadius: '18px',
        border: '1.5px solid #1976d2',
        boxShadow: '0 4px 24px 0 rgba(0,0,0,0.25)',
        padding: '40px 32px',
        color: '#fff',
        boxSizing: 'border-box',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '18px'
      }}
    >
      <div style={{ width: '100%' }}>
        <div style={{ marginBottom:10 }}>
          Year:
          <input
            type="number"
            value={params.year}
            onChange={e => setParams({ ...params, year: +e.target.value })}
            style={{ width:80, margin:'0 8px' }}
          />
          GP:
          <input
            type="text"
            value={params.gp}
            onChange={e => setParams({ ...params, gp: e.target.value })}
            style={{ width:200, margin:'0 8px' }}
          />
          Session:
          <select
            value={params.session}
            onChange={e => setParams({ ...params, session: e.target.value })}
            style={{ margin:'0 8px' }}
          >
            <option>Practice</option>
            <option>Qualifying</option>
            <option>Race</option>
          </select>
          Lap:
          <input
            type="number"
            min={1}
            value={params.lap}
            onChange={e => setParams({ ...params, lap: +e.target.value })}
            style={{ width:60, margin:'0 8px' }}
          />
        </div>

        <div style={{ marginBottom:20, display: 'flex', gap: '12px' }}>
          <button onClick={start} disabled={playing}>Start Simulation</button>
          <button onClick={toggle} disabled={!history.length}>{playing?'Pause':'Resume'}</button>
          <button onClick={reset}  disabled={!history.length}>Reset</button>
        </div>

        <h3>Live Metrics</h3>
        <p>
          Time: {live.lapTime}s | Wing Angle: {live.wingAngle}mm | Body Flex: {live.bodyFlex}mm
        </p>

        <h3>Lap {params.lap} History</h3>
        <div style={{ height: 12 }} />
        <div style={{
          background: 'rgba(30,40,60,0.7)',
          borderRadius: '18px',
          boxShadow: '0 4px 24px 0 rgba(0,0,0,0.25)',
          padding: '24px',
          margin: 'auto',
          width: 'fit-content',
          maxWidth: '100%'
        }}>
          <LineChart width={800} height={300} data={history.slice(0, step+1)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis dataKey="lapTime" label={{ value:'Time (s)', position:'insideBottom', fill:'#fff', fontSize:16, fontWeight:'bold' }} tick={{ fill:'#fff', fontSize:14 }} />
            <YAxis yAxisId="left" label={{ value:'Wing Angle (mm)', angle:-90, position:'insideLeft', fill:'#64b5f6', fontSize:16, fontWeight:'bold' }} tick={{ fill:'#64b5f6', fontSize:14 }} />
            <YAxis yAxisId="right" orientation="right"
                   label={{ value:'Body Flex (mm)', angle:90, position:'insideRight', fill:'#00e5ff', fontSize:16, fontWeight:'bold' }} tick={{ fill:'#00e5ff', fontSize:14 }} />
            <Tooltip contentStyle={{ background:'#222', border:'none', borderRadius:8, color:'#fff', fontSize:14 }} />
            <Legend verticalAlign="top" iconType="circle" wrapperStyle={{ color:'#fff', fontSize:16, fontWeight:'bold' }} />
            <Line yAxisId="left"  type="monotone" dataKey="wingAngle" dot={false} stroke="#1976d2" strokeWidth={3} strokeLinejoin="round" filter="url(#shadow)" />
            <Line yAxisId="right" type="monotone" dataKey="bodyFlex"  dot={false} stroke="#00e5ff" strokeWidth={3} strokeLinejoin="round" filter="url(#shadow)" />
            <defs>
              <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="2" stdDeviation="2" floodColor="#1976d2" floodOpacity="0.25" />
              </filter>
            </defs>
          </LineChart>
        </div>
      </div>
    </motion.div>
  )
}
