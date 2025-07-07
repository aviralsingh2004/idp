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
    <form className="aerodynamics-form" onSubmit={onSubmit}>
      <div className="form-title">Predict Aerodynamics</div>

      {Object.keys(inputs).map((key) => (
        <div className="form-group" key={key}>
          <label htmlFor={key}>{key}:</label>
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

      <button type="submit" disabled={loading}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>

      {result && (
        <div className="prediction-result" style={{ marginTop: '20px', textAlign: 'center' }}>
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
            fontStyle: 'italic'
          }}>
            <strong>AI Summary:</strong>
            <p>{result.analysis}</p>
          </div>
        </div>
      )}
    </form>
  );
}
