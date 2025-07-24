import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend
} from 'recharts';
import Track3D from './Track3D';
import { fetchRawTelemetry, fetchTrackPositions } from '../api';

export default function SimulationDashboard() {
  const [history, setHistory] = useState([]);
  const [track, setTrack] = useState([]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [params, setParams] = useState({
    year: 2023, gp: 'Italian Grand Prix', session: 'Race', driver: 'VER', lap: 1
  });
  const timerRef = useRef(null);

  const start = async () => {
    const telemetry = await fetchRawTelemetry(params);
    const rawTrack = await fetchTrackPositions(params);
    const xs = rawTrack.map(p => p.X);
    const ys = rawTrack.map(p => p.Y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const norm = rawTrack.map(p => ({
      X: ((p.X - minX) / (maxX - minX)) * 1000,
      Y: 1000 - ((p.Y - minY) / (maxY - minY)) * 1000, // flip Y
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
  };

  useEffect(() => {
    if (playing && history.length) {
      timerRef.current = setInterval(() => {
        setStep(i => i + 1 < history.length ? i + 1 : (clearInterval(timerRef.current), i));
      }, 100);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [playing, history]);

  const live = history[step] || { lapTime: 0, wingAngle: 0, bodyFlex: 0 };
  const car = track[step] || { X: 0, Y: 0 };
  const path = track.map(p => `${p.X},${p.Y}`).join(' ');

  return (
    <motion.div className="simulation-dashboard-tile" style={{ padding: '32px' }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div>
          Year:
          <input type="number" value={params.year} onChange={e => setParams({ ...params, year: +e.target.value })} />
          GP:
          <input type="text" value={params.gp} onChange={e => setParams({ ...params, gp: e.target.value })} />
          Session:
          <select value={params.session} onChange={e => setParams({ ...params, session: e.target.value })}>
            <option>Practice</option>
            <option>Qualifying</option>
            <option>Race</option>
          </select>
          Lap:
          <input type="number" min={1} value={params.lap} onChange={e => setParams({ ...params, lap: +e.target.value })} />
        </div>

        <div style={{ margin: '12px' }}>
          <button onClick={start} disabled={playing}>Start Simulation</button>
          <button onClick={toggle} disabled={!history.length}>{playing ? 'Pause' : 'Resume'}</button>
          <button onClick={reset} disabled={!history.length}>Reset</button>
        </div>

        <h3>Live Metrics</h3>
        <p>Time: {live.lapTime}s | Wing: {live.wingAngle}mm | Flex: {live.bodyFlex}mm</p>

        <h3>Track Visual</h3>
        {/* <svg viewBox="0 0 1000 1000" width="400" height="400" style={{ background: '#eee', marginBottom: 20 }}>
          <polyline points={path} fill="none" stroke="black" strokeWidth="2" />
          <circle cx={car.X} cy={car.Y} r="5" fill="red" />
        </svg> */}
        <div style={{ marginTop: 32 }}>
  <Track3D
    year={params.year}
    gp={params.gp}
    session={params.session}
    lap={params.lap}
    driver="VER"
    syncIndex={step}  
  />
</div>
        <LineChart width={800} height={300} data={history.slice(0, step + 1)}>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="lapTime" tick={{ fill: '#fff' }} label={{ value: 'Time (s)', fill: '#fff', position: 'insideBottom' }} />
          <YAxis yAxisId="left" tick={{ fill: '#64b5f6' }} label={{ value: 'Wing Angle', angle: -90, fill: '#64b5f6' }} />
          <YAxis yAxisId="right" orientation="right" tick={{ fill: '#00e5ff' }} label={{ value: 'Body Flex', angle: 90, fill: '#00e5ff' }} />
          <Tooltip contentStyle={{ background: '#222', color: '#fff' }} />
          <Legend />
          <Line yAxisId="left" dataKey="wingAngle" stroke="#1976d2" dot={false} />
          <Line yAxisId="right" dataKey="bodyFlex" stroke="#00e5ff" dot={false} />
        </LineChart>
      </div>
    </motion.div>
  );
}