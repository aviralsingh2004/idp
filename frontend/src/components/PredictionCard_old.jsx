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
    <form className="aerodynamics-form prediction-card-tile" onSubmit={onSubmit}>
      <div className="form-title">Predict Aerodynamics</div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 'var(--spacing-md)' }}>
        {Object.keys(inputs).map((key) => (
          <div className="form-group" key={key}>
            <label htmlFor={key}>{key.replace(/_/g, ' ')}</label>
            <input
              type="number"
              id={key}
              name={key}
              value={inputs[key]}
              onChange={handleChange}
              step="any"
            />
          </div>
        ))}
      </div>

      <button type="submit" disabled={loading}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>

      {result && (
        <div className="prediction-result">
          <h3>Prediction Results</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--spacing-sm)', marginBottom: 'var(--spacing-md)' }}>
            <div><strong>Downforce:</strong> {result.downforce?.toFixed(3)} N</div>
            <div><strong>Drag:</strong> {result.drag?.toFixed(3)} N</div>
            <div><strong>Drag Coefficient:</strong> {result.drag_coefficient?.toFixed(5)}</div>
            <div><strong>L/D Ratio:</strong> {result.lift_to_drag_ratio?.toFixed(3)}</div>
          </div>
          {result.analysis && (
            <div>
              <h4 style={{ color: 'var(--color-accent)', marginBottom: 'var(--spacing-sm)' }}>AI Analysis</h4>
              <p>{result.analysis}</p>
            </div>
          )}
        </div>
      )}
    </form>
  );
}
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
