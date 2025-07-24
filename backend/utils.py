import time
import torch
import torch.nn as nn
import fastf1
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ── Your existing Scikit-Learn models & scaler ───────────────────
scaler    = joblib.load('model/scaler.pkl')
rf_cd     = joblib.load('model/rf_model.pkl')
grid_cd   = joblib.load('model/grid_model.pkl')
rf_df_cls = joblib.load('model/rf_cls.pkl')

FEATURES = [
    'Speed_kmph','B_Ramp_Angle','B_Diffusor_Angle','A_Car_Length','Reynolds_Number',
    'Body_Surface_Ratio','Greenhouse_Ratio','Combined_Inclination',
    'Aerodynamic_Blend_Factor','Speed_Diffusor_Product','Length_Width_Ratio'
]

def predict_aero(params: dict):
    df = pd.DataFrame([params], columns=FEATURES)
    Xs = scaler.transform(df)
    cd_pred = float(grid_cd.predict(Xs)[0])
    lvl = int(round(rf_df_cls.predict(Xs)[0]))
    speed = float(params.get('Speed_kmph', 0))  # default = 0
    sug = "Keep wing angle low for less drag (straight)" if speed > 300 else "Keep wing angle high for more downforce (corners)"
    return {'cd': cd_pred, 'downforce_level': ['Low','Medium','High'][lvl], 'suggestion': sug}

def get_feature_importance():
    return dict(zip(FEATURES, rf_cd.feature_importances_))

# ── Autoencoder + simulation state ──────────────────────────────
metrics_latest  = {'lapTime':0.0,'wingAngle':0.0,'bodyFlex':0.0}
metrics_history = {}  # lap → list of entries

def preprocess_f1_lap(year:int, gp:str, sess:str='Race') -> pd.DataFrame:
    fastf1.Cache.enable_cache('f1_cache')
    session = fastf1.get_session(year, gp, sess)
    session.load()
    lap0 = session.laps.pick_accurate().pick_fastest()
    tel = lap0.get_car_data().add_distance()
    df = tel[['Distance','Speed','Throttle','Brake']].copy()
    scaler_local = StandardScaler().fit(df.values)
    df[df.columns] = scaler_local.transform(df.values)
    return df

class Autoencoder(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden, dim), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))

def inference_loop(poll_interval=0.1):
    global metrics_latest, metrics_history
    while True:
        try:
            df = preprocess_f1_lap(2023, 'Italian Grand Prix', 'Race')
            model = Autoencoder(df.shape[1])
            model.load_state_dict(torch.load('model/aero_model.pth'))
            model.eval()
            hist, t = [], 0.0
            for _, row in df.iterrows():
                x = torch.tensor(row.values, dtype=torch.float32).unsqueeze(0)
                mse = float(((model(x)-x)**2).mean())
                t += poll_interval
                entry = {
                  'lapTime': round(t,3),
                  'wingAngle': round(mse*30,2),
                  'bodyFlex':  round(mse*10,2)
                }
                metrics_latest = entry
                hist.append(entry)
                time.sleep(poll_interval)
            metrics_history[1] = hist
        except Exception as e:
            print("Inference error:", e)
            time.sleep(1)
