import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

function TelemetryTable({ rows = [] }) {
  return (
    <div className="telemetry-table-tile" style={{
      background: '#232526',
      borderRadius: '18px',
      boxShadow: '0 4px 24px 0 rgba(0,0,0,0.25)',
      padding: '32px 40px',
      margin: '32px 0',
      width: '100%',
      maxWidth: '100%',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      boxSizing: 'border-box',
      border: '1.5px solid #1976d2'
    }}>
      <table style={{ borderCollapse: 'separate', borderSpacing: '0 10px', width: '100%', maxWidth: 900 }}>
        <thead>
          <tr style={{ color: '#00e5ff', fontWeight: 'bold', fontSize: 16 }}>
            <th style={{padding: '8px 16px'}}>Team</th>
            <th style={{padding: '8px 16px'}}>Turn</th>
            <th style={{padding: '8px 16px'}}>Speed Diff</th>
            <th style={{padding: '8px 16px'}}>Gear Diff</th>
            <th style={{padding: '8px 16px'}}>Throttle Diff</th>
            <th style={{padding: '8px 16px'}}>Brake Diff</th>
          </tr>
        </thead>
        <tbody>
          <AnimatePresence>
            {(rows || []).map((row, idx) => (
              <motion.tr
                key={idx}
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 24 }}
                transition={{ duration: 0.45, delay: idx * 0.06, ease: [0.22, 1, 0.36, 1] }}
                style={{ color: '#fff', fontSize: 15, textAlign: 'center', transition: 'background 0.2s', cursor: 'pointer' }}
                onMouseOver={e => e.currentTarget.style.background = 'rgba(0,229,255,0.10)'}
                onMouseOut={e => e.currentTarget.style.background = 'transparent'}
              >
                <td style={{padding: '8px 16px', borderBottom: '1px solid #1976d2'}}>{row.team}</td>
                <td style={{padding: '8px 16px', borderBottom: '1px solid #1976d2'}}>{row.turn}</td>
                <td style={{padding: '8px 16px', borderBottom: '1px solid #1976d2'}}>{row.speed_diff.toFixed(1)}</td>
                <td style={{padding: '8px 16px', borderBottom: '1px solid #1976d2'}}>{row.gear_diff.toFixed(1)}</td>
                <td style={{padding: '8px 16px', borderBottom: '1px solid #1976d2'}}>{row.throttle_diff.toFixed(2)}</td>
                <td style={{padding: '8px 16px', borderBottom: '1px solid #1976d2'}}>{row.brake_diff.toFixed(2)}</td>
              </motion.tr>
            ))}
          </AnimatePresence>
        </tbody>
      </table>
    </div>
  );
}

export default TelemetryTable;
