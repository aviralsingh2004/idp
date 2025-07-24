import React, { useState } from 'react';
import { predictAero } from '../api';

export default function PredictionCard() {
  const [inputs, setInputs] = useState({
    Speed_kmph: 120,
    B_Ramp_Angle: 5,
    B_Diffusor_Angle: 6,
    A_Car_Length: 30,
    Reynolds_Number: 2.5e7,
    Body_Surface_Ratio: 0.2,
    Greenhouse_Ratio: 1.5,
    Combined_Inclination: 3,
    Aerodynamic_Blend_Factor: 15,
    Speed_Diffusor_Product: 720,
    Length_Width_Ratio: 1.2
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
    <form className="aerodynamics-form prediction-card-tile" onSubmit={onSubmit}>
      <div className="form-title">Aerodynamic Prediction</div>

      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
        gap: 'var(--spacing-md)' 
      }}>
        {Object.keys(inputs).map((key) => (
          <div className="form-group" key={key}>
            <label htmlFor={key}>
              {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </label>
            <input
              type="number"
              id={key}
              name={key}
              value={inputs[key]}
              onChange={handleChange}
              step="any"
              required
            />
          </div>
        ))}
      </div>

      <button type="submit" disabled={loading} style={{ marginTop: 'var(--spacing-lg)' }}>
        {loading ? 'Analyzing...' : 'Predict Performance'}
      </button>

      {result && (
        <div className="prediction-result">
          <h3>Aerodynamic Analysis</h3>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', 
            gap: 'var(--spacing-md)', 
            marginBottom: 'var(--spacing-lg)' 
          }}>
            {result.downforce && (
              <div style={{ 
                padding: 'var(--spacing-md)', 
                background: 'rgba(76, 175, 80, 0.1)', 
                borderRadius: 'var(--border-radius-sm)',
                border: '1px solid rgba(76, 175, 80, 0.3)'
              }}>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Downforce</div>
                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--color-success)' }}>
                  {result.downforce.toFixed(2)} N
                </div>
              </div>
            )}
            
            {result.drag && (
              <div style={{ 
                padding: 'var(--spacing-md)', 
                background: 'rgba(255, 152, 0, 0.1)', 
                borderRadius: 'var(--border-radius-sm)',
                border: '1px solid rgba(255, 152, 0, 0.3)'
              }}>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Drag Force</div>
                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--color-warning)' }}>
                  {result.drag.toFixed(2)} N
                </div>
              </div>
            )}
            
            {result.drag_coefficient && (
              <div style={{ 
                padding: 'var(--spacing-md)', 
                background: 'rgba(25, 118, 210, 0.1)', 
                borderRadius: 'var(--border-radius-sm)',
                border: '1px solid rgba(25, 118, 210, 0.3)'
              }}>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Drag Coefficient</div>
                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--color-primary)' }}>
                  {result.drag_coefficient.toFixed(4)}
                </div>
              </div>
            )}
            
            {result.lift_to_drag_ratio && (
              <div style={{ 
                padding: 'var(--spacing-md)', 
                background: 'rgba(0, 229, 255, 0.1)', 
                borderRadius: 'var(--border-radius-sm)',
                border: '1px solid rgba(0, 229, 255, 0.3)'
              }}>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>L/D Ratio</div>
                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--color-secondary)' }}>
                  {result.lift_to_drag_ratio.toFixed(3)}
                </div>
              </div>
            )}
          </div>
          
          {result.analysis && (
            <div style={{ 
              padding: 'var(--spacing-lg)', 
              background: 'rgba(255, 235, 59, 0.05)', 
              borderRadius: 'var(--border-radius-md)',
              border: '1px solid rgba(255, 235, 59, 0.2)'
            }}>
              <h4 style={{ 
                color: 'var(--color-accent)', 
                marginBottom: 'var(--spacing-md)', 
                fontSize: '1.1rem',
                fontWeight: '600'
              }}>
                AI Performance Analysis
              </h4>
              <p style={{ 
                lineHeight: '1.7', 
                color: 'var(--text-secondary)',
                margin: 0
              }}>
                {result.analysis}
              </p>
            </div>
          )}
        </div>
      )}
    </form>
  );
}
