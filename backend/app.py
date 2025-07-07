# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import fastf1

from utils import predict_aero, get_feature_importance

app = Flask(__name__)
CORS(app)

# enable FastF1 cache directory
fastf1.Cache.enable_cache('./f1_cache')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Expects JSON body with keys matching FEATURES:
      {
        "Speed_kmph": float,
        "B_Ramp_Angle": float,
        ... (all 11 features)
      }
    Returns: { cd, downforce_level, suggestion }
    """
    params = request.get_json() or {}
    result = predict_aero(params)
    return jsonify(result), 200


@app.route('/api/feature-importance', methods=['GET'])
def api_feature_importance():
    """
    Returns the RandomForest feature importances used in predict_aero.
    """
    fi = get_feature_importance()
    return jsonify(fi), 200


@app.route('/api/telemetry-comparison', methods=['GET'])
def api_telemetry_comparison():
    """
    Compares each driver's fastest lap telemetry vs. ideal values.
    Returns a list of:
      { team, turn, speed_diff, gear_diff, throttle_diff, brake_diff }
    """
    session = fastf1.get_session(2023, 'Monza', 'R')
    session.load(telemetry=True)

    required = ['Speed', 'nGear', 'Throttle', 'Brake']
    ideal = {
        'Straight':          {'Speed': 320, 'nGear': 8, 'Throttle': 1.0, 'Brake': 0.0},
        'Low-Speed Turn':    {'Speed': 160, 'nGear': 3, 'Throttle': 0.5, 'Brake': 0.5},
        'Medium-Speed Turn': {'Speed': 180, 'nGear': 5, 'Throttle': 0.7, 'Brake': 0.3},
    }

    result = []
    laps = session.laps.pick_quicklaps()
    for drv in session.drivers:
        drv_laps = laps.pick_driver(drv)
        if drv_laps.empty:
            continue
        team = drv_laps.iloc[0]['Team']
        try:
            fastest = drv_laps.pick_fastest()
            tel = fastest.get_car_data().add_distance()
        except Exception:
            continue
        if not all(c in tel.columns for c in required):
            continue
        avg = {c: tel[c].mean() for c in required}
        for turn_name, iv in ideal.items():
            result.append({
                'team':           team,
                'turn':           turn_name,
                'speed_diff':     abs(avg['Speed']    - iv['Speed']),
                'gear_diff':      abs(avg['nGear']    - iv['nGear']),
                'throttle_diff':  abs(avg['Throttle'] - iv['Throttle']),
                'brake_diff':     abs(avg['Brake']    - iv['Brake']),
            })

    return jsonify(result), 200


@app.route('/api/raw-telemetry', methods=['GET'])
def api_raw_telemetry():
    """
    Streams one lap's raw suspension or fallback telemetry:
      - lapTime: seconds since start
      - wingAngle: SuspensionTravelRL or fallback from Speed
      - bodyFlex:  SuspensionTravelRR or fallback from Brake
    Query params:
      year, gp (Grand Prix name), session, lap (1-based index)
    """
    year    = int(request.args.get('year', 2023))
    gp      = request.args.get('gp', 'Italian Grand Prix')
    session = request.args.get('session', 'Race')
    lap_no  = int(request.args.get('lap', 1))

    # load session
    sess = fastf1.get_session(year, gp, session)
    sess.load(telemetry=True)

    laps = sess.laps.pick_accurate()
    if lap_no < 1 or lap_no > len(laps):
        return jsonify([]), 200

    lap = laps.iloc[lap_no - 1]
    tel = lap.get_car_data().add_distance()
    cols = tel.columns

    # find real suspension channels if present
    rl_key = next((c for c in cols if 'SuspensionTravelRL' in c), None) \
             or next((c for c in cols if c == 'SuspensionRL'), None)
    rr_key = next((c for c in cols if 'SuspensionTravelRR' in c), None) \
             or next((c for c in cols if c == 'SuspensionRR'), None)

    use_fallback = not (rl_key and rr_key)
    if use_fallback:
        max_speed = float(tel['Speed'].max() or 1.0)

    data = []
    t, dt = 0.0, 0.1

    for _, row in tel.iterrows():
        if use_fallback:
            speed = float(row.get('Speed', 0))
            brake = float(row.get('Brake', 0))
            wing  = round((1 - speed / max_speed) * 30.0, 2)
            flex  = round(brake * 5.0, 2)
        else:
            wing  = float(row.get(rl_key, 0))
            flex  = float(row.get(rr_key, 0))

        data.append({
            'lapTime':  round(t, 3),
            'wingAngle': wing,
            'bodyFlex':  flex
        })
        t += dt

    return jsonify(data), 200


if __name__ == '__main__':
    app.run(debug=True)
