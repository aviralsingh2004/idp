import joblib
import numpy as np
import pandas as pd

# Load models & scaler at import time
scaler      = joblib.load('model/scaler.pkl')
rf_cd       = joblib.load('model/rf_model.pkl')
grid_cd     = joblib.load('model/grid_model.pkl')
rf_df_cls   = joblib.load('model/rf_cls.pkl')

FEATURES = [
    'Speed_kmph','B_Ramp_Angle','B_Diffusor_Angle','A_Car_Length','Reynolds_Number',
    'Body_Surface_Ratio','Greenhouse_Ratio','Combined_Inclination',
    'Aerodynamic_Blend_Factor','Speed_Diffusor_Product','Length_Width_Ratio'
]

def predict_aero(params: dict):
    """
    params must include keys matching FEATURES
    Returns: dict with cd, downforce_level, suggestion
    """
    # create DF
    df = pd.DataFrame([params], columns=FEATURES)
    Xs = scaler.transform(df)
    cd_pred = float(grid_cd.predict(Xs)[0])
    df_level = int(np.round(rf_df_cls.predict(Xs)[0]))
    levels = ['Low','Medium','High']
    
    # suggestion logic
    speed = params['Speed_kmph']
    if speed > 300:
        sug = "Keep wing angle low for less drag (straight)"
    elif speed > 150:
        sug = "Moderate wing angle, optimize for balance"
    else:
        sug = "Increase wing angle for maximum downforce (turns)"
    
    return {'cd': cd_pred, 'downforce_level': levels[df_level], 'suggestion': sug}

def get_feature_importance():
    fi = rf_cd.feature_importances_
    return dict(zip(FEATURES, fi))
