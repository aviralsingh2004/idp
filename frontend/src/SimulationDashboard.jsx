import { FaClock, FaWind, FaCompressArrowsAlt } from 'react-icons/fa';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import Track3D from './Track3D';
import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useSimulation } from '../hooks/useSimulation';

const SimulationDashboard = () => {
  const { year, gp, session, lap } = useParams();
  const [params, setParams] = useState({ year: +year, gp: gp, session: session, lap: +lap });
  const [step, setStep] = useState(0);
  const { live, history, start, toggle, reset } = useSimulation(params);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setParams({ year: +year, gp: gp, session: session, lap: +lap });
  }, [year, gp, session, lap]);

  useEffect(() => {
    setPlaying(false); // Reset playing state when params change
  }, [params]);

  useEffect(() => {
    if (history.length > 0) {
      setPlaying(true);
    }
  }, [history]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (playing) {
        setStep(prev => prev + 1);
      }
    }, 100); // Simulate 100ms per step
    return () => clearInterval(interval);
  }, [playing]);

  return (
    <motion.div className="simulation-dashboard-tile" style={{ padding: '32px', display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh' }}>
      <div style={{
        background: 'rgba(34, 34, 50, 0.7)',
        borderRadius: 24,
        boxShadow: '0 8px 32px 0 #C3002F33',
        padding: '40px 48px',
        maxWidth: 900,
        width: '100%',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        border: '1.5px solid #C3002F44',
        margin: '0 auto',
        zIndex: 2
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{ marginBottom: 24, width: '100%', display: 'flex', justifyContent: 'center', gap: 24 }}>
            <div>
              <label style={{ fontWeight: 600, color: '#FFEB00', marginRight: 8 }}>Year:</label>
              <input type="number" value={params.year} onChange={e => setParams({ ...params, year: +e.target.value })} style={{ borderRadius: 8, border: '1px solid #444', padding: '4px 8px', width: 80, background: '#222', color: '#fff' }} />
            </div>
            <div>
              <label style={{ fontWeight: 600, color: '#FFEB00', marginRight: 8 }}>GP:</label>
              <input type="text" value={params.gp} onChange={e => setParams({ ...params, gp: e.target.value })} style={{ borderRadius: 8, border: '1px solid #444', padding: '4px 8px', width: 160, background: '#222', color: '#fff' }} />
            </div>
            <div>
              <label style={{ fontWeight: 600, color: '#FFEB00', marginRight: 8 }}>Session:</label>
              <select value={params.session} onChange={e => setParams({ ...params, session: e.target.value })} style={{ borderRadius: 8, border: '1px solid #444', padding: '4px 8px', background: '#222', color: '#fff' }}>
                <option>Practice</option>
                <option>Qualifying</option>
                <option>Race</option>
              </select>
            </div>
            <div>
              <label style={{ fontWeight: 600, color: '#FFEB00', marginRight: 8 }}>Lap:</label>
              <input type="number" min={1} value={params.lap} onChange={e => setParams({ ...params, lap: +e.target.value })} style={{ borderRadius: 8, border: '1px solid #444', padding: '4px 8px', width: 60, background: '#222', color: '#fff' }} />
            </div>
          </div>

          <div style={{ margin: '12px', display: 'flex', gap: 16 }}>
            <button onClick={start} disabled={playing} style={{
              background: 'linear-gradient(90deg, #FFEB00 0%, #C3002F 100%)',
              color: '#222',
              border: 'none',
              borderRadius: 10,
              padding: '8px 20px',
              fontWeight: 700,
              fontSize: 16,
              boxShadow: '0 2px 8px 0 #C3002F33',
              cursor: playing ? 'not-allowed' : 'pointer',
              opacity: playing ? 0.6 : 1,
              transition: 'all 0.2s'
            }}>Start Simulation</button>
            <button onClick={toggle} disabled={!history.length} style={{
              background: playing ? '#C3002F' : '#FFEB00',
              color: playing ? '#fff' : '#222',
              border: 'none',
              borderRadius: 10,
              padding: '8px 20px',
              fontWeight: 700,
              fontSize: 16,
              boxShadow: '0 2px 8px 0 #C3002F33',
              cursor: !history.length ? 'not-allowed' : 'pointer',
              opacity: !history.length ? 0.6 : 1,
              transition: 'all 0.2s'
            }}>{playing ? 'Pause' : 'Resume'}</button>
            <button onClick={reset} disabled={!history.length} style={{
              background: '#222',
              color: '#FFEB00',
              border: '1.5px solid #FFEB00',
              borderRadius: 10,
              padding: '8px 20px',
              fontWeight: 700,
              fontSize: 16,
              boxShadow: '0 2px 8px 0 #C3002F33',
              cursor: !history.length ? 'not-allowed' : 'pointer',
              opacity: !history.length ? 0.6 : 1,
              transition: 'all 0.2s'
            }}>Reset</button>
          </div>

          <h3 style={{ color: '#FFEB00', fontWeight: 700, fontSize: 22, marginTop: 24, marginBottom: 8 }}>Live Metrics</h3>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }} style={{ display: 'flex', gap: 32, marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 18 }}>
              <FaClock style={{ color: '#FFEB00' }} />
              <span>Time:</span>
              <span style={{ fontWeight: 700, color: '#fff', fontSize: 20 }}>{live.lapTime}s</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 18 }}>
              <FaWind style={{ color: '#00e5ff' }} />
              <span>Wing:</span>
              <span style={{ fontWeight: 700, color: '#00e5ff', fontSize: 20 }}>{live.wingAngle}mm</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 18 }}>
              <FaCompressArrowsAlt style={{ color: '#C3002F' }} />
              <span>Flex:</span>
              <span style={{ fontWeight: 700, color: '#C3002F', fontSize: 20 }}>{live.bodyFlex}mm</span>
            </div>
          </motion.div>

          <h3 style={{ color: '#FFEB00', fontWeight: 700, fontSize: 22, marginTop: 24, marginBottom: 8 }}>Track Visual</h3>
          <div style={{ marginTop: 32, marginBottom: 32, borderRadius: 16, overflow: 'hidden', boxShadow: '0 4px 24px 0 #C3002F33', background: 'rgba(10,10,30,0.5)', backdropFilter: 'blur(8px)' }}>
            <Track3D
              year={params.year}
              gp={params.gp}
              session={params.session}
              lap={params.lap}
              driver="VER"
              syncIndex={step}
            />
          </div>

          <LineChart width={800} height={300} data={history.slice(0, step + 1)} style={{ background: 'rgba(20,20,30,0.7)', borderRadius: 16, boxShadow: '0 2px 12px 0 #C3002F33', marginTop: 24 }}>
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
      </div>
    </motion.div>
  );
};

export default SimulationDashboard; 