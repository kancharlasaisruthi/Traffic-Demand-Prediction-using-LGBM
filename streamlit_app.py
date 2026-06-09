import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pygeohash as pgh
from pathlib import Path

BASE_DIR = Path(__file__).parent

@st.cache_data
def load_artifacts():
    preprocessor_path = BASE_DIR / "preprocessor.pkl"
    model_path = BASE_DIR / "lgb_model.pkl"
    train_csv = BASE_DIR / "dataset" / "training.csv"

    with open(preprocessor_path, "rb") as f:
        prep_obj = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)


    df_train = pd.read_csv(train_csv)

    if "geohash" not in df_train.columns and "geohash6" in df_train.columns:
        df_train = df_train.rename(columns={"geohash6": "geohash"})


    if "timestamp" not in df_train.columns and "time_stamp" in df_train.columns:
        df_train = df_train.rename(columns={"time_stamp": "timestamp"})
    df_train["timestamp"] = df_train["timestamp"].astype(str).str.replace('.', ':', regex=False)
    df_train[["Hour", "Minute"]] = df_train["timestamp"].str.split(':', expand=True)[[0,1]].astype(int)
    df_train["time_slot"] = df_train["Hour"] * 4 + (df_train["Minute"] // 15)
    df_train["geo_5"] = df_train["geohash"].astype(str).str[:5]


    geo_freq = df_train["geo_5"].value_counts().to_dict()
    geo_time_mean = (
        df_train.groupby(["geo_5", "time_slot"])["demand"].mean().reset_index()
        .set_index(["geo_5", "time_slot"]) ["demand"].to_dict()
    )
    hour_mean = df_train.groupby("Hour")["demand"].mean().to_dict()
    global_mean = df_train["demand"].mean()

    return prep_obj, model, geo_freq, geo_time_mean, hour_mean, global_mean


def featurize_input(geohash, time_str, day, geo_freq, geo_time_mean, hour_mean, global_mean):

    t = str(time_str).strip()
    if ":" in t:
        parts = t.split(":")
        hour = int(parts[0])
        minute = int(parts[1])
    else:
       
        hour = int(t)
        minute = 0

    time_slot = hour * 4 + (minute // 15)

    rush_hour = int(((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 20)))

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)

    lat, lon = pgh.decode(geohash)
    geo_4 = geohash[:4]
    geo_5 = geohash[:5]
    geo_6 = geohash[:6]

    geo_freq_val = int(geo_freq.get(geo_5, 0))

    geo_time_key = (geo_5, time_slot)
    geo_time_val = geo_time_mean.get(geo_time_key, global_mean)

    hour_avg = hour_mean.get(hour, global_mean)

    row = {
        "day": int(day),
        "rush_hour": rush_hour,
        "geo_4": geo_4,
        "geo_5": geo_5,
        "geo_6": geo_6,
        "geo_freq": geo_freq_val,
        "Latitude": lat,
        "Longitude": lon,
        "Hour": hour,
        "Minute": minute,
        "time_slot": time_slot,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "minute_sin": minute_sin,
        "minute_cos": minute_cos,
        "geo_time_demand_mean": geo_time_val,
        "hour_avg_demand": hour_avg,
    }

    return pd.DataFrame([row])


def main():
    st.title("Traffic Demand Predictor")

    st.markdown("Enter `geohash` (6-char), `time` (HH:MM) and `day` to predict demand.")

    geohash = st.text_input("Geohash (6 chars)")
    time_str = st.text_input("Time (HH:MM)")
    day = st.number_input("Day (integer)", min_value=0, value=1)

    if st.button("Predict"):
        if not geohash or not time_str:
            st.error("Please provide geohash and time.")
            return

        try:
            prep_obj, model, geo_freq, geo_time_mean, hour_mean, global_mean = load_artifacts()
        except Exception as e:
            st.error(f"Failed loading artifacts: {e}")
            return

        X_df = featurize_input(geohash.strip(), time_str.strip(), day, geo_freq, geo_time_mean, hour_mean, global_mean)

        # ensure columns order matches training preprocessor expectations
        num_features = prep_obj.get("num_features")
        cat_features = prep_obj.get("cat_features")
        all_features = num_features + cat_features

        X_df = X_df.reindex(columns=all_features)

        preprocessor = prep_obj.get("preprocessor")
        X_trans = preprocessor.transform(X_df)

        pred = model.predict(X_trans)

        st.success(f"Predicted demand: {float(pred[0]):.4f}")


if __name__ == "__main__":
    main()
