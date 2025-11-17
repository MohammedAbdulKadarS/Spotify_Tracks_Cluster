import streamlit as st
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "kmeans_model.joblib"
SCALER_PATH = "scaler_model.joblib"
ALL_FEATURES = [
    'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
    'loudness', 'speechiness', 'tempo', 'valence', 'popularity'
]

cluster_description = {
    0: "Trending/Popular Songs",
    1: "Acoustic/Chill/Lo-fi",
    2: "High Energy/Dance/Party",
    3: "Outlier/Noise or Rare"
}

st.title("🎵 Cluster Prediction (KMeans): Two Feature Input 🎵")
st.write("Pick any 2 music features, enter their values, and hit **PREDICT** to see which cluster/mood you belong to! Only those features will be used for prediction.")

kmeans = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with st.form("predict_form", clear_on_submit=True):
    f1 = st.selectbox("First Feature", ALL_FEATURES, index=0)
    f2 = st.selectbox("Second Feature", [f for f in ALL_FEATURES if f != f1], index=1)
    v1 = st.number_input(f"Value for {f1}", value=0.5, step=0.01)
    v2 = st.number_input(f"Value for {f2}", value=0.5, step=0.01)
    predict_btn = st.form_submit_button("PREDICT")

if predict_btn:
    # Only use the selected two features. All other features are dropped from the input!
    X_user = pd.DataFrame({f1: [v1], f2: [v2]})

    # Make a blank dataframe with all columns, fill in only the selected features, others set as np.nan
    row = pd.DataFrame({f: [np.nan] for f in ALL_FEATURES})
    row[f1] = v1
    row[f2] = v2

    # Select only the columns that are not nan
    real_features = [f1, f2]
    # NOTE: Since scaler & model expect full feature set, set nan for missing features (or zero/default if your scaler was fit that way)
    input_row = row.fillna(0) # or .fillna(row.mean()) if you want to use mean value for missing
    
    # Scale and predict
    scaled = scaler.transform(input_row)
    pred_cluster = int(kmeans.predict(scaled)[0])
    desc = cluster_description.get(pred_cluster, "Unknown Type")
    
    st.success(f"Your input matches **Cluster {pred_cluster}: {desc}** based only on features: {f1}, {f2}.")

