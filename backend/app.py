from flask import Flask, request, jsonify
from flask_cors import CORS
import fastf1
import pandas as pd
import json
from groq import Groq
from utils import predict_aero, get_feature_importance
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()
if not os.environ.get('GROQ_API_KEY'):
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in the .env file.")

groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
MODEL = "llama3-70b-8192"

app = Flask(__name__)
CORS(app)
fastf1.Cache.enable_cache('./f1_cache')

@app.route("/api/predict", methods=["POST"])
def api_predict():
    params = request.get_json() or {}
    basic = predict_aero(params)

    system_prompt = (
        "You are an expert F1 aerodynamics analyst. "
        "You will be provided with aerodynamic prediction results in JSON. "
        "Return a JSON response in the format: {\"analysis\": \"...\"}, where the value is a paragraph summarizing the aerodynamic impact and car behavior."
    )
    user_prompt = (
        f"Prediction result: {json.dumps(basic)}\n"
        "Return a JSON object as specified above."
    )

    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        raw_output = resp.choices[0].message.content
        print("Groq raw output:", raw_output)

        summary_text = json.loads(raw_output).get("analysis") if isinstance(raw_output, str) else raw_output
    except Exception as e:
        print("Groq error:", str(e))
        summary_text = "Error generating summary: " + str(e)

    return jsonify({
        **basic,
        "analysis": summary_text
    }), 200

@app.route('/api/feature-importance', methods=['GET'])
def api_feature_importance():
    fi = get_feature_importance()
    return jsonify(fi), 200
import traceback

@app.route('/api/track-positions', methods=['GET'])
def api_track_positions():
    try:
        year    = int(request.args.get('year', 2023))
        gp      = request.args.get('gp', 'Italian Grand Prix')
        session = request.args.get('session', 'Race')
        driver  = request.args.get('driver', 'VER')
        lap_no  = int(request.args.get('lap', 1))

        sess = fastf1.get_session(year, gp, session)
        sess.load(telemetry=True)

        laps = sess.laps.pick_drivers([driver])
        if lap_no < 1 or lap_no > len(laps):
            return jsonify([]), 200

        lap = laps.iloc[lap_no - 1]

        # Fetch position and speed data separately
        pos_data = lap.get_pos_data()
        car_data = lap.get_car_data()

        import pandas as pd

        # Merge by timestamp
        merged = pd.merge_asof(pos_data, car_data, on='Time')

        output = [
            {
                'X': float(row['X']),
                'Y': float(row['Y']),
                'Speed': float(row['Speed']) if not pd.isna(row['Speed']) else 0.0
            }
            for _, row in merged.iterrows()
            if not pd.isna(row['X']) and not pd.isna(row['Y'])
        ]

        return jsonify(output), 200

    except Exception as e:
        import traceback
        print("\n[TRACK POSITION ERROR]")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/telemetry-comparison', methods=['GET'])
def api_telemetry_comparison():
    session = fastf1.get_session(2023, 'Monza', 'R')
    session.load(telemetry=True)
    required = ['Speed', 'nGear', 'Throttle', 'Brake']
    ideal = {
        'Straight': {'Speed': 320, 'nGear': 8, 'Throttle': 1.0, 'Brake': 0.0},
        'Low-Speed Turn': {'Speed': 160, 'nGear': 3, 'Throttle': 0.5, 'Brake': 0.5},
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
                'team': team,
                'turn': turn_name,
                'speed_diff': abs(avg['Speed'] - iv['Speed']),
                'gear_diff': abs(avg['nGear'] - iv['nGear']),
                'throttle_diff': abs(avg['Throttle'] - iv['Throttle']),
                'brake_diff': abs(avg['Brake'] - iv['Brake']),
            })

    return jsonify(result), 200

@app.route('/api/raw-telemetry', methods=['GET'])
def api_raw_telemetry():
    year = int(request.args.get('year', 2023))
    gp = request.args.get('gp', 'Italian Grand Prix')
    session = request.args.get('session', 'Race')
    lap_no = int(request.args.get('lap', 1))

    sess = fastf1.get_session(year, gp, session)
    sess.load(telemetry=True)

    laps = sess.laps.pick_accurate()
    if lap_no < 1 or lap_no > len(laps):
        return jsonify([]), 200

    lap = laps.iloc[lap_no - 1]
    tel = lap.get_car_data().add_distance()
    cols = tel.columns

    rl_key = next((c for c in cols if 'SuspensionTravelRL' in c), None) or next((c for c in cols if c == 'SuspensionRL'), None)
    rr_key = next((c for c in cols if 'SuspensionTravelRR' in c), None) or next((c for c in cols if c == 'SuspensionRR'), None)

    use_fallback = not (rl_key and rr_key)
    if use_fallback:
        max_speed = float(tel['Speed'].max() or 1.0)

    data = []
    t, dt = 0.0, 0.1
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
            'lapTime': round(t, 3),
            'wingAngle': wing,
            'bodyFlex': flex
        })
        t += dt

    return jsonify(data), 200

if __name__ == '__main__':
    app.run(debug=True)
