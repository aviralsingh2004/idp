import React, { useState } from 'react';
import { predictAero } from '../api';

export default function PredictionCard() {
  const [inputs, setInputs] = useState({
    Speed_kmph: 120, B_Ramp_Angle: 5, B_Diffusor_Angle: 6,
    A_Car_Length: 30, Reynolds_Number: 2.5e7,
    Body_Surface_Ratio: 0.2, Greenhouse_Ratio: 1.5,
    Combined_Inclination: 3, Aerodynamic_Blend_Factor: 15,
    Speed_Diffusor_Product: 720, Length_Width_Ratio: 1.2
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setInputs({
      ...inputs,
      [e.target.name]: parseFloat(e.target.value)
    });
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await predictAero(inputs);
      setResult(res);
    } catch (err) {
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form
      className="aerodynamics-form prediction-card-tile"
      onSubmit={onSubmit}
    >
      <div className="form-title" style={{ fontSize: 24, fontWeight: 'bold', color: '#00e5ff', marginBottom: 12 }}>
        Predict Aerodynamics
      </div>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        width: '100%'
      }}>
        {Object.keys(inputs).map((key) => (
          <div className="form-group" key={key} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label htmlFor={key} style={{ color: '#90caf9', fontWeight: 500, marginBottom: 2 }}>{key}:</label>
            <input
              type="number"
              id={key}
              name={key}
              value={inputs[key]}
              onChange={handleChange}
              step="any"
              style={{
                padding: '8px 10px',
                borderRadius: 6,
                border: '1.5px solid #1976d2',
                background: '#232526',
                color: '#fff',
                fontSize: 15,
                outline: 'none',
                transition: 'border 0.2s',
                boxShadow: '0 1px 4px 0 rgba(0,0,0,0.10)'
              }}
            />
          </div>
        ))}
      </div>

      <button
        type="submit"
        disabled={loading}
        style={{
          background: '#1976d2',
          color: '#fff',
          fontWeight: 'bold',
          fontSize: 18,
          border: 'none',
          borderRadius: 8,
          padding: '12px 32px',
          marginTop: 12,
          cursor: loading ? 'not-allowed' : 'pointer',
          boxShadow: '0 2px 8px 0 rgba(25,118,210,0.15)',
          transition: 'background 0.2s, box-shadow 0.2s'
        }}
      >
        {loading ? 'Predicting...' : 'Predict'}
      </button>

      {result && (
        <div
          className="prediction-result"
          style={{
            marginTop: '24px',
            textAlign: 'center',
            width: '100%',
            background: 'rgba(25,118,210,0.10)',
            borderRadius: '12px',
            padding: '18px 12px',
            boxShadow: '0 2px 8px 0 rgba(25,118,210,0.10)',
            color: '#fff',
            fontSize: 17,
            border: '1.5px solid #1976d2'
          }}
        >
          <p><strong>Cd:</strong> {result.cd.toFixed(4)}</p>
          <p><strong>Downforce:</strong> {result.downforce_level}</p>
          <p><strong>Suggestion:</strong> {result.suggestion}</p>
          <hr style={{ margin: '20px 0' }} />
          <div style={{
            padding: '15px',
            borderRadius: '8px',
            textAlign: 'left',
            maxWidth: '600px',
            margin: 'auto',
            fontStyle: 'italic',
            background: 'rgba(0,229,255,0.08)',
            color: '#00e5ff',
            border: '1.5px solid #00e5ff'
          }}>
            <strong>AI Summary:</strong>
            <p>{result.analysis}</p>
          </div>
        </div>
      )}
    </form>
  );
}
