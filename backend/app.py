from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import predict_aero, get_feature_importance
import pandas as pd
import fastf1

app = Flask(__name__)
CORS(app)

# Load FastF1 cache
fastf1.Cache.enable_cache('./f1_cache')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Expects JSON:
    {
      "Speed_kmph": float, "B_Ramp_Angle": float, ...
      # all 11 features
    }
    """
    data = request.get_json()
    res = predict_aero(data)
    return jsonify(res)

@app.route('/api/feature-importance', methods=['GET'])
def api_feat_imp():
    fi = get_feature_importance()
    return jsonify(fi)

@app.route('/api/telemetry-comparison', methods=['GET'])
def api_telemetry():
    # Example using Monza race and ideal_values
    session = fastf1.get_session(2023, 'Monza', 'R')
    session.load(telemetry=True)
    required_cols = ['Speed','nGear','Throttle','Brake']
    ideal_values = {
      'Straight': {'Speed':320,'nGear':8,'Throttle':1.0,'Brake':0.0},
      'Low-Speed Turn': {'Speed':160,'nGear':3,'Throttle':0.5,'Brake':0.5},
      'Medium-Speed Turn': {'Speed':180,'nGear':5,'Throttle':0.7,'Brake':0.3}
    }
    telemetry_comparison = []
    laps = session.laps.pick_quicklaps()
    for drv in session.drivers:
        drv_laps = laps.pick_driver(drv)
        if drv_laps.empty: continue
        team = drv_laps.iloc[0]['Team']
        try:
            fastest = drv_laps.pick_fastest()
            tel = fastest.get_car_data().add_distance()
        except:
            continue
        if not all(c in tel.columns for c in required_cols):
            continue
        avg = {c: tel[c].mean() for c in required_cols}
        for turn, ideal in ideal_values.items():
            telemetry_comparison.append({
                'team': team,
                'turn': turn,
                'speed_diff': abs(avg['Speed']-ideal['Speed']),
                'gear_diff': abs(avg['nGear']-ideal['nGear']),
                'throttle_diff': abs(avg['Throttle']-ideal['Throttle']),
                'brake_diff': abs(avg['Brake']-ideal['Brake'])
            })
    return jsonify(telemetry_comparison)

if __name__ == '__main__':
    app.run(debug=True)
