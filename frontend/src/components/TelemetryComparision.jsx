import React, { useState } from 'react'
import TelemetryTable from './TelemetryTable'
import { getTelemetryComparison } from '../api'

export default function TelemetryComparison() {
  const [rows, setRows] = useState([])
  const [running, setRunning] = useState(false)

  const start = async () => {
    setRunning(true)
    try {
      setRows(await getTelemetryComparison())
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="section telemetry-table-tile">
      <h2>Telemetry Comparison vs Ideal</h2>
      <div style={{ 
        display: 'flex', 
        gap: 'var(--spacing-md)', 
        marginBottom: 'var(--spacing-lg)',
        justifyContent: 'center'
      }}>
        <button onClick={start} disabled={running}>
          {running ? 'Comparing…' : 'Start Comparison'}
        </button>
        <button 
          onClick={() => setRows([])} 
          disabled={running || !rows.length}
          style={{ 
            background: rows.length > 0 ? 'var(--color-warning)' : undefined 
          }}
        >
          Clear
        </button>
      </div>
      {rows.length > 0 && <TelemetryTable rows={rows} />}
    </div>
  )
}
