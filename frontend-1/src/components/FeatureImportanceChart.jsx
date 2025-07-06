import React from 'react';
import { Bar } from 'react-chartjs-2';

const FeatureImportanceChart = () => {
  const data = {
    labels: ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
    datasets: [
      {
        label: 'Feature Importance',
        data: [0.2, 0.4, 0.1, 0.15, 0.15],
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
      },
    ],
  };

  const options = {
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <div>
      <Bar data={data} options={options} />
    </div>
  );
};

export default FeatureImportanceChart;