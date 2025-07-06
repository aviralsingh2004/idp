import React from 'react';

function TelemetryTable() {
  const telemetryData = [
    { parameter: 'Speed', ideal: '200 km/h', actual: '195 km/h' },
    { parameter: 'RPM', ideal: '12000', actual: '11500' },
    { parameter: 'Temperature', ideal: '90°C', actual: '92°C' },
  ];

  return (
    <table>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Ideal Value</th>
          <th>Actual Value</th>
        </tr>
      </thead>
      <tbody>
        {telemetryData.map((data, index) => (
          <tr key={index}>
            <td>{data.parameter}</td>
            <td>{data.ideal}</td>
            <td>{data.actual}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default TelemetryTable;