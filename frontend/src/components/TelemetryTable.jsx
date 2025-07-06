import React, { useEffect, useState } from 'react';
import { getTelemetryComparison } from '../api';

function TelemetryTable({ rows = [] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Team</th><th>Turn</th><th>Speed Diff</th>
          <th>Gear Diff</th><th>Throttle Diff</th><th>Brake Diff</th>
        </tr>
      </thead>
      <tbody>
        {(rows || []).map((row, idx) => (
          <tr key={idx}>
            <td>{row.team}</td><td>{row.turn}</td>
            <td>{row.speed_diff.toFixed(1)}</td>
            <td>{row.gear_diff.toFixed(1)}</td>
            <td>{row.throttle_diff.toFixed(2)}</td>
            <td>{row.brake_diff.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default TelemetryTable;
