# ================================
# Imports & Setup
# ================================

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

# ================================
# Load & Prepare Data
# ================================


@st.cache_data
def load_data():
    db_url = (
        "postgres://ufnbfacj9c7u80:"
        "pa129f8c5adad53ef2c90db10cce0c899f8c7bdad022cca4e85a8729b19aad68d"
        "@ceq2kf3e33g245.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d9f89h4ju1lleh"
    ).replace("postgres://", "postgresql://")
    engine = create_engine(db_url)

    query = """
    SELECT SERIALNO, VALP, TEN, HINCP, FINCP, BDS, RMSP, NP, YRBLT,
           ADJINC, REGION, DIVISION, PUMA, AGS
    FROM acs_pums
    WHERE TEN = 1 AND VALP > 0
      AND HINCP IS NOT NULL AND FINCP IS NOT NULL
      AND BDS IS NOT NULL AND RMSP IS NOT NULL
      AND NP IS NOT NULL AND YRBLT IS NOT NULL
      AND REGION IS NOT NULL AND PUMA IS NOT NULL
      AND DIVISION IS NOT NULL AND AGS IS NOT NULL;
    """
    df_pums = pd.read_sql(query, engine)

    # Manual PUMA to County
    puma_to_county = {
        3902: "Orange",
        900: "San Francisco",
        6101: "Los Angeles",
        3502: "Sacramento",
        300: "San Diego",
    }

    df_pums["puma"] = df_pums["puma"].astype(int)
    df_pums["County"] = df_pums["puma"].map(puma_to_county).astype(str).str.title()

    # Load Crime Data
    crime_csv = "/Users/mahekpatel/Downloads/Crimes_and_Clearances_with_Arson-1985-2023_by month.csv"
    df_crime = pd.read_csv(crime_csv, low_memory=False)

    df_crime["Month"] = df_crime["Month"].astype(str).str.zfill(2)
    df_crime["Year"] = df_crime["Year"].astype(str)
    df_crime["CrimeMonth"] = pd.to_datetime(df_crime["Year"] + "-" + df_crime["Month"])

    df_crime["County"] = (
        df_crime["County"]
        .str.replace(" County", "", regex=False)
        .str.strip()
        .str.title()
    )

    # Aggregate annual crime for 2023
    df_crime_annual = (
        df_crime[df_crime["Year"] == "2023"]
        .groupby("County")
        .agg({"Violent_sum": "sum", "Property_sum": "sum"})
        .reset_index()
    )

    df_final = df_pums.merge(df_crime_annual, how="left", on="County")

    # === Scale VALP to real $
    df_final["valp"] = (
        pd.to_numeric(df_final["valp"], errors="coerce")
        .fillna(1)
        .clip(lower=1, upper=5_000)
        * 1_000
    )

    # Fix features
    safe_features = [
        "hincp",
        "fincp",
        "bds",
        "rmsp",
        "np",
        "yrblt",
        "Violent_sum",
        "Property_sum",
    ]
    for col in safe_features:
        df_final[col] = pd.to_numeric(df_final[col], errors="coerce").fillna(0)

    return df_final, sorted(df_final["County"].dropna().unique())


# ================================
# Train TensorFlow Model w/ Progress
# ================================


class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, status_text, total_epochs):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        pct = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(pct)
        self.status_text.text(f"Training... {epoch+1}/{self.total_epochs} epochs")


@st.cache_resource
def train_model(df_final):
    features = [
        "hincp",
        "fincp",
        "bds",
        "rmsp",
        "np",
        "yrblt",
        "Violent_sum",
        "Property_sum",
    ]
    target = ["valp"]

    X = df_final[features].values
    y = np.log(df_final[target].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = keras.Sequential(
        [
            keras.Input(shape=(len(features),)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Streamlit progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,
        callbacks=[
            StreamlitProgressCallback(progress_bar, status_text, total_epochs=50)
        ],
        verbose=0,
    )

    status_text.text("Training complete.")
    return model, scaler


# ================================
# Streamlit GUI
# ================================

import streamlit as st

# ================================
# STREAMLIT APP GUI SECTION
# ================================

st.title("Housing Market Investment Predictor")

df_final, counties = load_data()
model, scaler = train_model(df_final)

# --- APP LAYOUT ---
st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    county = st.selectbox("Select County:", options=counties)
    zipcode = st.text_input(
        "ZIP Code (optional)", placeholder="Enter ZIP code (e.g., 90001)"
    )
    hincp = st.number_input(
        "Household Income (HINCP)", min_value=0, value=100000, format="%d", step=1000
    )
    fincp = st.number_input(
        "Family Income (FINCP)", min_value=0, value=120000, format="%d", step=1000
    )
    purchase_price = st.number_input(
        "Proposed Purchase Price ($)", min_value=0, value=500000, format="%d", step=1000
    )

with col2:
    bds = st.number_input("Bedrooms (BDS)", min_value=0, value=3)
    rmsp = st.number_input("Total Rooms (RMSP)", min_value=0, value=5)
    npersons = st.number_input("Number of People (NP)", min_value=1, value=3)
    yrblt = st.number_input(
        "Year Built (YRBLT)", min_value=1800, max_value=2025, value=2000
    )

# --- BUTTONS ---
col3, col4 = st.columns(2)

predict_clicked = col3.button("Predict Value")
reset_clicked = col4.button("Reset Inputs")

# --- RESET LOGIC ---
if reset_clicked:
    st.experimental_rerun()

# --- PREDICTION LOGIC ---
if predict_clicked:
    violent = df_final[df_final["County"] == county]["Violent_sum"].values[0]
    property_crime = df_final[df_final["County"] == county]["Property_sum"].values[0]

    # If you add ZIP → override violent/property_crime here later

    user_input = np.array(
        [[hincp, fincp, bds, rmsp, npersons, yrblt, violent, property_crime]]
    )
    input_scaled = scaler.transform(user_input)

    pred_log = model.predict(input_scaled)[0][0]
    pred_val = np.exp(pred_log) - 1
    score = (pred_val - purchase_price) / purchase_price

    st.success(f"Predicted Market Value: ${pred_val:,.0f}")
    st.info(f"Investment Score: {score:.2%} (positive means good)")

    # --- Basic Crime Info ---
    st.subheader("Local Crime Insights")
    st.write(f"County annual violent crime incidents: {violent:,.0f}")
    st.write(f"County annual property crime incidents: {property_crime:,.0f}")
    st.write("More ZIP-specific crime insights coming soon!")
