import React, { useState, useEffect, useRef } from 'react'
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
    <div style={{ padding:20 }}>

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

      <div style={{ marginBottom:20 }}>
        <button onClick={start} disabled={playing}>Start Simulation</button>
        <button onClick={toggle} disabled={!history.length}>{playing?'Pause':'Resume'}</button>
        <button onClick={reset}  disabled={!history.length}>Reset</button>
      </div>

      <h3>Live Metrics</h3>
      <p>
        Time: {live.lapTime}s | Wing Angle: {live.wingAngle}mm | Body Flex: {live.bodyFlex}mm
      </p>

      <h3>Lap {params.lap} History</h3>
      <LineChart width={800} height={300} data={history.slice(0, step+1)}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="lapTime" label={{ value:'Time (s)', position:'insideBottom' }} />
        <YAxis yAxisId="left" label={{ value:'Wing Angle (mm)', angle:-90, position:'insideLeft' }} />
        <YAxis yAxisId="right" orientation="right"
               label={{ value:'Body Flex (mm)', angle:90, position:'insideRight' }} />
        <Tooltip />
        <Legend verticalAlign="top" />
        <Line yAxisId="left"  type="monotone" dataKey="wingAngle" dot={false} />
        <Line yAxisId="right" type="monotone" dataKey="bodyFlex"  dot={false} />
      </LineChart>
    </div>
  )
}
