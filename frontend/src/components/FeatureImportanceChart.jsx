import React, { useEffect, useState } from 'react';
import { getFeatureImportance } from '../api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from 'recharts';

export default function FeatureImportanceChart() {
  const [data, setData] = useState([]);
  useEffect(() => {
    getFeatureImportance().then(obj => {
      const arr = Object.entries(obj).map(([feat, imp]) => ({ feat, imp }));
      setData(arr);
    });
  }, []);
  return (
    <div className="feature-importance-tile" style={{
      background: 'rgba(30,40,60,0.7)',
      borderRadius: '18px',
      boxShadow: '0 4px 24px 0 rgba(0,0,0,0.25)',
      padding: '24px',
      margin: 'auto',
      width: 'fit-content',
      maxWidth: '100%'
    }}>
      <ResponsiveContainer width={600} height={300}>
        <BarChart data={data} layout="vertical">
          <defs>
            <linearGradient id="barGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#1976d2" />
              <stop offset="100%" stopColor="#00e5ff" />
            </linearGradient>
            <filter id="barShadow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="2" stdDeviation="2" floodColor="#1976d2" floodOpacity="0.25" />
            </filter>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis type="number" tick={{ fill:'#fff', fontSize:14 }} label={{ value:'Importance', position:'bottom', offset: 20, fill:'#fff', fontSize:16, fontWeight:'bold' }} />
          <YAxis dataKey="feat" type="category" width={150} tick={{ fill:'#90caf9', fontSize:14 }} label={{ value:'Feature', angle:-90, position:'insideLeft', fill:'#90caf9', fontSize:16, fontWeight:'bold' }} />
          <Tooltip contentStyle={{ background:'#222', border:'none', borderRadius:8, color:'#fff', fontSize:14 }} />
          <Legend verticalAlign="top" wrapperStyle={{ color:'#fff', fontSize:16, fontWeight:'bold' }} />
          <Bar dataKey="imp" fill="url(#barGradient)" radius={[8,8,8,8]} barSize={18} stroke="#1976d2" strokeWidth={1.5} filter="url(#barShadow)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
