import React, { useEffect, useState } from 'react';
import axios from 'axios';

export default function TrackWithSpeedColor() {
  const [points, setPoints] = useState([]);
  const [carIndex, setCarIndex] = useState(0);

  useEffect(() => {
    axios.get('http://localhost:5000/api/track-positions', {
      params: {
        year: 2023,
        gp: 'Italian Grand Prix',
        session: 'Race',
        driver: 'VER',
        lap: 1
      }
    }).then(res => {
      const raw = res.data;
      if (!raw || raw.length === 0) return;

      const xs = raw.map(p => p.X);
      const ys = raw.map(p => p.Y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);

      const norm = raw.map(p => ({
        X: ((p.X - minX) / (maxX - minX)) * 1000,
        Y: 1000 - ((p.Y - minY) / (maxY - minY)) * 1000,
        Speed: p.Speed
      }));

      setPoints(norm);
      setCarIndex(0);
    });
  }, []);

  useEffect(() => {
    if (points.length === 0) return;
    const interval = setInterval(() => {
      setCarIndex(prev => (prev + 1) % points.length);
    }, 100); // slower animation
    return () => clearInterval(interval);
  }, [points]);

  const car = points[carIndex] || { X: 0, Y: 0, Speed: 0 };

  const getColor = (speed) => {
    if (speed > 300) return 'red';
    if (speed > 200) return 'orange';
    if (speed > 100) return 'yellow';
    return 'green';
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <h2 className="text-xl font-bold text-blue-400">Track Animation with Speed Gradient</h2>
      <svg viewBox="0 0 1000 1000" width="600" height="600" className="bg-gray-100 rounded">
        {points.slice(1).map((p, i) => (
          <line
            key={i}
            x1={points[i].X}
            y1={points[i].Y}
            x2={p.X}
            y2={p.Y}
            stroke={getColor(p.Speed)}
            strokeWidth="2"
          />
        ))}
        <circle cx={car.X} cy={car.Y} r="6" fill="blue">
          <title>
            Speed: {car.Speed.toFixed(1)} km/h\n
            Location: ({car.X.toFixed(1)}, {car.Y.toFixed(1)})
          </title>
        </circle>
      </svg>
      <div className="mt-2 text-gray-700">
        Speed: {car.Speed.toFixed(1)} km/h
      </div>
    </div>
  );
}
