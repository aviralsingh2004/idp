from flask import Flask, request, jsonify
from flask_cors import CORS
import threading, fastf1

from utils import (
    predict_aero, get_feature_importance,
    preprocess_f1_lap, inference_loop,
    metrics_latest, metrics_history
)

app = Flask(__name__)
CORS(app)
fastf1.Cache.enable_cache('./f1_cache')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    return jsonify(predict_aero(request.get_json()))

@app.route('/api/feature-importance', methods=['GET'])
def api_feat_imp():
    return jsonify(get_feature_importance())

@app.route('/api/raw-telemetry', methods=['GET'])
def api_raw_telemetry():
    """
    Returns one lap’s telemetry, falling back to speed/brake if no suspension.
    """
    # params
    year    = int(request.args.get('year', 2023))
    gp      = request.args.get('gp', 'Italian Grand Prix')
    session = request.args.get('session', 'Race')
    lap_no  = int(request.args.get('lap', 1))

    # load
    sess = fastf1.get_session(year, gp, session)
    sess.load(telemetry=True)

    laps = sess.laps.pick_accurate()
    if lap_no < 1 or lap_no > len(laps):
        return jsonify([])

    lap = laps.iloc[lap_no - 1]
    tel = lap.get_car_data().add_distance()

    cols = tel.columns
    # try real suspension channels
    rl_key = next((c for c in cols if 'Suspension' in c and c.endswith('RL')), None)
    rr_key = next((c for c in cols if 'Suspension' in c and c.endswith('RR')), None)

    # if no suspension, prepare fallback: speed and brake
    use_fallback = not (rl_key and rr_key)
    if use_fallback:
        # compute max speed so wingAngle scales properly
        max_speed = float(tel['Speed'].max() or 1.0)

    data, t, dt = [], 0.0, 0.1
    for _, row in tel.iterrows():
        if use_fallback:
            speed = float(row.get('Speed', 0))
            brake = float(row.get('Brake', 0))
            wing = round((1 - speed / max_speed) * 30.0, 2)
            flex = round(brake * 5.0, 2)
        else:
            wing = float(row.get(rl_key, 0))
            flex = float(row.get(rr_key, 0))

        data.append({
            'lapTime':  round(t, 3),
            'wingAngle': wing,
            'bodyFlex':  flex
        })
        t += dt

    return jsonify(data)

@app.route('/api/start-sim', methods=['POST'])
def api_start_sim():
    """
    Starts inference_loop in background.
    Called when user presses 'Start Simulation'.
    """
    threading.Thread(target=inference_loop, daemon=True).start()
    return jsonify({"status":"Simulation started"}), 200

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    return jsonify(metrics_latest)

@app.route('/api/metrics/history', methods=['GET'])
def api_metrics_history():
    lap = int(request.args.get('lap',1))
    return jsonify(metrics_history.get(lap, []))

if __name__ == '__main__':
    app.run(debug=True)
