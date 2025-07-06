import React, { useEffect, useState } from 'react';
import { getFeatureImportance } from '../api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function FeatureImportanceChart() {
  const [data, setData] = useState([]);
  useEffect(() => {
    getFeatureImportance().then(obj => {
      const arr = Object.entries(obj).map(([feat, imp]) => ({ feat, imp }));
      setData(arr);
    });
  }, []);
  return (
    <BarChart width={600} height={300} data={data} layout="vertical">
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis type="number" />
      <YAxis dataKey="feat" type="category" width={150}/>
      <Tooltip />
      <Bar dataKey="imp" fill="#8884d8" />
    </BarChart>
  );
}
