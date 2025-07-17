#!/usr/bin/env python3
"""
Production Model Trainer for Housing Prediction - OVERFITTING-PROOF VERSION
Run this once to train and save all models for instant loading
BULLETPROOF TENSORFLOW: Extreme Overfitting Prevention
"""

import keras
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import tensorflow as tf
from sqlalchemy import create_engine, text
import os
from datetime import datetime
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Set TensorFlow to be less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


def load_crime_data():
    """Load crime data for feature engineering"""
    crime_data = {
        "County": [
            "Los Angeles",
            "San Francisco",
            "San Diego",
            "Sacramento",
            "Orange",
            "Story",
            "Riverside",
            "Alameda",
            "Santa Clara",
        ],
        "State": ["CA", "CA", "CA", "CA", "CA", "IA", "CA", "CA", "CA"],
        "Violent_sum": [45821, 6854, 8245, 3421, 4832, 234, 3892, 2156, 1876],
        "Property_sum": [125643, 28745, 22156, 12845, 18932, 1456, 15234, 8934, 7234],
        "Population": [
            10014009,
            873965,
            3298634,
            1585055,
            3175692,
            97117,
            2418185,
            1682353,
            1927852,
        ],
        "Median_Income": [
            70381,
            119136,
            87067,
            75237,
            103518,
            58289,
            71941,
            112017,
            140258,
        ],
        "Education_College_Pct": [32.1, 58.3, 41.2, 35.8, 42.7, 65.2, 28.9, 52.1, 67.8],
    }
    df_crime = pd.DataFrame(crime_data)

    # Calculate rates per 100k
    df_crime["Violent_rate"] = (
        df_crime["Violent_sum"] / df_crime["Population"]
    ) * 100000
    df_crime["Property_rate"] = (
        df_crime["Property_sum"] / df_crime["Population"]
    ) * 100000

    # Safety score (0-100, higher = safer)
    max_violent = df_crime["Violent_rate"].max()
    max_property = df_crime["Property_rate"].max()
    df_crime["Safety_Score"] = 100 - (
        (df_crime["Violent_rate"] / max_violent * 40)
        + (df_crime["Property_rate"] / max_property * 60)
    )

    return df_crime


def load_and_prepare_data():
    """Load data from database and prepare for training"""

    print("Loading data from database...")

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

    df_raw = pd.read_sql(query, engine)
    print(f"Loaded {len(df_raw):,} records from database")

    # Apply feature engineering
    df = engineer_comprehensive_features(df_raw)

    return df


def engineer_comprehensive_features(df):
    """Apply the same feature engineering as the main app"""

    print("🔧 Engineering features...")

    # Convert column names to lowercase for consistency
    df.columns = df.columns.str.lower()

    # Property value scaling (VALP codes to real dollars)
    df["valp"] = (
        pd.to_numeric(df["valp"], errors="coerce").fillna(1).clip(lower=1, upper=5_000)
        * 1_000
    )

    # Income adjustments with inflation correction
    if "adjinc" in df.columns:
        df["hincp_real"] = df["hincp"] * df["adjinc"] / 1_000_000
        df["fincp_real"] = df["fincp"] * df["adjinc"] / 1_000_000
    else:
        df["hincp_real"] = pd.to_numeric(df["hincp"], errors="coerce").fillna(0)
        df["fincp_real"] = pd.to_numeric(df["fincp"], errors="coerce").fillna(0)

    # Derived housing characteristics
    df["house_age"] = 2023 - pd.to_numeric(df["yrblt"], errors="coerce").fillna(2000)
    df["rooms_per_person"] = pd.to_numeric(df["rmsp"], errors="coerce") / pd.to_numeric(
        df["np"], errors="coerce"
    ).clip(lower=1)
    df["income_to_value_ratio"] = df["hincp_real"] / df["valp"].clip(lower=1)

    # Add PUMA to County mapping
    puma_to_county = {
        300: "San Diego",
        301: "San Diego",
        302: "San Diego",
        303: "San Diego",
        900: "San Francisco",
        901: "San Francisco",
        902: "San Francisco",
        3502: "Sacramento",
        3503: "Sacramento",
        3504: "Sacramento",
        3505: "Sacramento",
        3902: "Orange",
        3903: "Orange",
        3904: "Orange",
        3905: "Orange",
        3906: "Orange",
        6101: "Los Angeles",
        6102: "Los Angeles",
        6103: "Los Angeles",
        6104: "Los Angeles",
        6105: "Los Angeles",
        6106: "Los Angeles",
        6107: "Los Angeles",
        6108: "Los Angeles",
        7307: "Riverside",
        7308: "Riverside",
        7309: "Riverside",
        7310: "Riverside",
        4901: "Alameda",
        4902: "Alameda",
        4903: "Alameda",
        5901: "Santa Clara",
        5902: "Santa Clara",
        5903: "Santa Clara",
        5904: "Santa Clara",
        1901: "Story",
        1902: "Story",
        1903: "Story",
    }

    df["puma"] = df["puma"].astype(int)
    df["county"] = df["puma"].map(puma_to_county).fillna("Unknown")

    # Add crime data
    df_crime = load_crime_data()
    df = df.merge(df_crime, left_on="county", right_on="County", how="left")

    # Handle crime features - use lowercase names
    crime_features_map = {
        "Violent_sum": "violent_sum",
        "Property_sum": "property_sum",
        "Violent_rate": "violent_rate",
        "Property_rate": "property_rate",
        "Safety_Score": "safety_score",
    }

    for old_name, new_name in crime_features_map.items():
        if old_name in df.columns:
            df[new_name] = pd.to_numeric(df[old_name], errors="coerce").fillna(0)
        else:
            df[new_name] = 0

    # Log transform target (as per methodology)
    df["valp_log"] = np.log(df["valp"] + 1)

    print("Feature engineering complete")
    return df


def create_bulletproof_tensorflow_model(n_features):
    """Create the most conservative TensorFlow model possible"""

    keras = tf.keras
    layers = tf.keras.layers

    print("    BULLETPROOF ARCHITECTURE:")
    print("    - Single layer: 6 neurons only")
    print("    - Massive regularization: L1 + L2")
    print("    - Extreme dropout: 0.8")
    print("    - Ultra-low learning rate: 0.00005")

    # MINIMAL model - almost as simple as linear regression
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            # Single tiny hidden layer with extreme regularization
            layers.Dense(
                6,  # Only 6 neurons - extremely small
                activation="relu",
                kernel_regularizer=keras.regularizers.l1_l2(
                    l1=0.01, l2=0.01
                ),  # Both L1 and L2
                bias_regularizer=keras.regularizers.l2(0.01),  # Regularize bias too
                activity_regularizer=keras.regularizers.l1(
                    0.01
                ),  # Regularize activations
            ),
            layers.Dropout(0.8),  # Drop 80% of neurons
            # Output layer with regularization
            layers.Dense(
                1,
                kernel_regularizer=keras.regularizers.l2(0.01),
                bias_regularizer=keras.regularizers.l2(0.01),
            ),
        ]
    )

    # Extremely conservative optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.00005,  # Ultra-low learning rate
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def train_all_models(df):
    """Train all models with bulletproof TensorFlow"""

    print("Training all 5 models with BULLETPROOF TensorFlow...")

    # Define features (same as your app)
    base_features = ["hincp_real", "fincp_real", "bds", "rmsp", "np", "house_age"]
    crime_features = ["violent_rate", "property_rate", "safety_score"]
    derived_features = ["rooms_per_person", "income_to_value_ratio"]

    features = base_features + crime_features + derived_features
    target = "valp_log"

    # Prepare data
    X = df[features].fillna(0)
    y = df[target]

    print(f"Training on {len(X):,} records with {len(features)} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    results = {}

    # 1. Linear Regression
    print("  Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = lr

    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="r2")

    results["Linear Regression"] = {
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
        "CV_R2_Mean": cv_scores.mean(),
        "CV_R2_Std": cv_scores.std(),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Generalization_Gap": r2_score(y_train, y_pred_train)
        - r2_score(y_test, y_pred_test),
    }

    # 2. Ridge Regression
    print("  Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    models["Ridge Regression"] = ridge

    y_pred_train = ridge.predict(X_train)
    y_pred_test = ridge.predict(X_test)
    cv_scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring="r2")

    results["Ridge Regression"] = {
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
        "CV_R2_Mean": cv_scores.mean(),
        "CV_R2_Std": cv_scores.std(),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Generalization_Gap": r2_score(y_train, y_pred_train)
        - r2_score(y_test, y_pred_test),
    }

    # 3. Random Forest
    print("  Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")

    results["Random Forest"] = {
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
        "CV_R2_Mean": cv_scores.mean(),
        "CV_R2_Std": cv_scores.std(),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Generalization_Gap": r2_score(y_train, y_pred_train)
        - r2_score(y_test, y_pred_test),
    }

    # 4. XGBoost
    print("  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
    )
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = xgb_model

    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="r2")

    results["XGBoost"] = {
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
        "CV_R2_Mean": cv_scores.mean(),
        "CV_R2_Std": cv_scores.std(),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Generalization_Gap": r2_score(y_train, y_pred_train)
        - r2_score(y_test, y_pred_test),
    }

    # 5. BULLETPROOF TensorFlow - Minimal and Heavily Regularized
    print("  Training BULLETPROOF TensorFlow (Anti-Overfitting)...")

    # Create bulletproof model
    tf_model = create_bulletproof_tensorflow_model(len(features))

    # Large validation split for monitoring
    X_train_tf, X_val_tf, y_train_tf, y_val_tf = train_test_split(
        X_train_scaled, y_train, test_size=0.3, random_state=42  # 30% validation
    )

    # Ultra-conservative callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,  # Very patient
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0001,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,  # Aggressive LR reduction
        patience=15,
        min_lr=0.000001,
        verbose=1,
    )

    print("    BULLETPROOF TRAINING SETTINGS:")
    print("    - 30% validation split (huge monitoring)")
    print("    - 80% dropout rate (extreme regularization)")
    print("    - Learning rate: 0.00005 (ultra-conservative)")
    print("    - Early stopping: 30 patience")
    print("    - L1+L2 regularization: 0.01 (massive penalty)")

    # Train with maximum conservatism
    history = tf_model.fit(
        X_train_tf,
        y_train_tf,
        validation_data=(X_val_tf, y_val_tf),
        epochs=500,  # Many epochs but will stop early
        batch_size=128,  # Large batches for stability
        callbacks=[early_stopping, reduce_lr],
        verbose=1,  # Show progress to monitor
    )

    models["TensorFlow"] = tf_model

    # Evaluation
    y_pred_train_tf = tf_model.predict(X_train_scaled, verbose=0).flatten()
    y_pred_test_tf = tf_model.predict(X_test_scaled, verbose=0).flatten()

    train_loss = tf_model.evaluate(X_train_scaled, y_train, verbose=0)[0]
    test_loss = tf_model.evaluate(X_test_scaled, y_test, verbose=0)[0]
    overfitting_ratio = test_loss / train_loss

    train_r2 = r2_score(y_train, y_pred_train_tf)
    test_r2 = r2_score(y_test, y_pred_test_tf)
    gen_gap = train_r2 - test_r2

    results["TensorFlow"] = {
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_tf)),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_tf)),
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "CV_R2_Mean": np.nan,
        "CV_R2_Std": np.nan,
        "Test_MAE": mean_absolute_error(y_test, y_pred_test_tf),
        "Generalization_Gap": gen_gap,
        "Training_History": history.history,
        "Train_Loss": train_loss,
        "Test_Loss": test_loss,
        "Overfitting_Ratio": overfitting_ratio,
        "Final_Epoch": len(history.history["loss"]),
    }

    print(f"\n  BULLETPROOF TENSORFLOW RESULTS:")
    print(f"    Train R²: {train_r2:.4f}")
    print(f"    Test R²: {test_r2:.4f}")
    print(f"    Generalization Gap: {gen_gap:.4f}")
    print(f"    Overfitting Ratio: {overfitting_ratio:.3f}")
    print(f"    Final Epoch: {len(history.history['loss'])}")

    # Compare with Linear Regression (sanity check)
    lr_r2 = results["Linear Regression"]["Test_R2"]
    print(f"\n  SANITY CHECK vs Linear Regression:")
    print(f"    Linear R²: {lr_r2:.4f}")
    print(f"    TensorFlow R²: {test_r2:.4f}")
    print(f"    Difference: {test_r2 - lr_r2:.4f}")

    if abs(test_r2 - lr_r2) < 0.01:
        print(
            f"    PERFECT: TensorFlow behaves like Linear Regression (no overfitting)"
        )
    elif test_r2 > lr_r2 + 0.02:
        print(f"    Still slightly better than linear - check predictions")
    else:
        print(f"    GOOD: Reasonable performance vs linear model")

    # Final overfitting assessment
    if gen_gap < 0.01:
        print(f"    BULLETPROOF SUCCESS: Gap = {gen_gap:.4f} < 0.01")
    elif gen_gap < 0.03:
        print(f"    EXCELLENT: Gap = {gen_gap:.4f} < 0.03")
    else:
        print(f"    Still some overfitting: Gap = {gen_gap:.4f}")

    print("All models trained successfully!")
    return models, scaler, results, features


def save_models(models, scaler, results, features):
    """Save all models and metadata to disk"""

    print("\nSaving models to disk...")

    # Create models directory
    os.makedirs("saved_models", exist_ok=True)

    # Save each traditional ML model
    traditional_models = [
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "XGBoost",
    ]

    for name in traditional_models:
        if name in models:
            filename = name.lower().replace(" ", "_") + "_model.pkl"
            filepath = os.path.join("saved_models", filename)
            joblib.dump(models[name], filepath)
            print(f"  Saved {name}")

    # Save TensorFlow model
    if "TensorFlow" in models:
        tf_path = os.path.join("saved_models", "tensorflow_model.keras")
        models["TensorFlow"].save(tf_path)
        print(f"  Saved BULLETPROOF TensorFlow")

    # Save scaler
    scaler_path = os.path.join("saved_models", "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler")

    # Save features list
    features_path = os.path.join("saved_models", "features.pkl")
    with open(features_path, "wb") as f:
        pickle.dump(features, f)
    print(f"  Saved features list")

    # Save results and metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "model_performance": results,
        "features": features,
        "n_features": len(features),
        "n_models": len(models),
        "tensorflow_type": "BULLETPROOF_ANTI_OVERFITTING",
    }

    metadata_path = os.path.join("saved_models", "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"  Saved metadata")

    print("\nBULLETPROOF MODEL PERFORMANCE SUMMARY:")
    print("=" * 90)
    print(f"{'Model':<20} | {'R²':<8} | {'RMSE':<8} | {'Gap':<8} | {'Status':<20}")
    print("-" * 90)

    for name, metrics in results.items():
        r2 = metrics["Test_R2"]
        rmse = metrics["Test_RMSE"]
        gen_gap = metrics["Generalization_Gap"]

        # Determine overfitting status
        if gen_gap < 0.01:
            status = "BULLETPROOF"
        elif gen_gap < 0.03:
            status = "EXCELLENT"
        elif gen_gap < 0.05:
            status = "GOOD"
        elif gen_gap < 0.1:
            status = "MODERATE"
        else:
            status = "OVERFITTING"

        print(
            f"{name:<20} | {r2:<8.4f} | {rmse:<8.4f} | {gen_gap:<8.4f} | {status:<20}"
        )

    # Special TensorFlow analysis
    if "TensorFlow" in results:
        tf_gap = results["TensorFlow"]["Generalization_Gap"]
        tf_ratio = results["TensorFlow"]["Overfitting_Ratio"]
        print(f"\n BULLETPROOF TENSORFLOW ANALYSIS:")
        print(f"   Generalization Gap: {tf_gap:.4f} (Target: < 0.03)")
        print(f"   Overfitting Ratio: {tf_ratio:.3f} (Target: < 1.2)")
        print(f"   Architecture: 6 neurons only (minimal complexity)")
        print(f"   Regularization: Extreme (L1+L2+Dropout 0.8)")

        if tf_gap < 0.03 and tf_ratio < 1.2:
            print(f"   MISSION ACCOMPLISHED: Overfitting eliminated!")
        else:
            print(f"   Still needs work - consider even simpler model")


def test_prediction_sanity(models, scaler, features):
    """Test that TensorFlow gives reasonable predictions"""

    print("\nPREDICTION SANITY TEST:")
    print("=" * 50)

    # Create a reasonable test case
    test_input = [
        75000,  # hincp_real
        85000,  # fincp_real
        3,  # bds
        6,  # rmsp
        3,  # np
        20,  # house_age
        400,  # violent_rate
        2500,  # property_rate
        60,  # safety_score
        2.0,  # rooms_per_person
        0.00015,  # income_to_value_ratio
    ]

    test_df = pd.DataFrame([test_input], columns=features)
    test_scaled = scaler.transform(test_df)

    print("Test case: 3BR house, $75K income, 20 years old")
    print("-" * 50)

    predictions = {}
    for name, model in models.items():
        try:
            if name == "TensorFlow":
                pred_log = model.predict(test_scaled, verbose=0)[0][0]
            else:
                pred_log = model.predict(test_df)[0]

            pred_value = np.exp(pred_log) - 1
            predictions[name] = pred_value
            print(f"{name:<20}: ${pred_value:,.0f}")

        except Exception as e:
            print(f"{name:<20}: ERROR - {e}")

    # Check if TensorFlow is reasonable
    if "TensorFlow" in predictions and "Linear Regression" in predictions:
        tf_pred = predictions["TensorFlow"]
        lr_pred = predictions["Linear Regression"]
        ratio = tf_pred / lr_pred

        print(f"\n🔍 TENSORFLOW vs LINEAR COMPARISON:")
        print(f"TensorFlow: ${tf_pred:,.0f}")
        print(f"Linear Reg: ${lr_pred:,.0f}")
        print(f"Ratio: {ratio:.2f}")

        if 0.5 <= ratio <= 2.0:
            print(f"REASONABLE: TensorFlow within 2x of Linear")
        else:
            print(f"EXTREME: TensorFlow {ratio:.1f}x different from Linear!")

    return predictions


if __name__ == "__main__":
    print("BULLETPROOF HOUSING PREDICTION MODEL TRAINER")
    print("=" * 60)
    print("MISSION: Eliminate TensorFlow overfitting completely")
    print("STRATEGY: Minimal architecture + Extreme regularization")
    print("TARGET: Generalization gap < 0.03")
    print()

    start_time = datetime.now()

    try:
        # 1. Load and prepare data
        print("Phase 1: Data Loading & Preparation")
        df = load_and_prepare_data()
        print(f"Data preparation complete: {len(df):,} records ready")

        # 2. Train all models with bulletproof TensorFlow
        print("\nPhase 2: BULLETPROOF Model Training")
        models, scaler, results, features = train_all_models(df)

        # 3. Test prediction sanity
        print("\nPhase 3: Prediction Sanity Check")
        test_predictions = test_prediction_sanity(models, scaler, features)

        # 4. Save everything
        print("\nPhase 4: Saving Models")
        save_models(models, scaler, results, features)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\nBULLETPROOF TRAINING COMPLETED in {duration:.1f} seconds")
        print("=" * 60)

        # Final overfitting assessment
        tf_metrics = results.get("TensorFlow", {})
        gen_gap = tf_metrics.get("Generalization_Gap", 1.0)
        overfitting_ratio = tf_metrics.get("Overfitting_Ratio", 2.0)

        print(f"   FINAL BULLETPROOF ASSESSMENT:")
        print(f"   TensorFlow Generalization Gap: {gen_gap:.4f}")
        print(f"   TensorFlow Overfitting Ratio: {overfitting_ratio:.3f}")

        if gen_gap < 0.01:
            print(f"   PERFECT: Bulletproof success! Gap < 0.01")
        elif gen_gap < 0.03:
            print(f"   EXCELLENT: Overfitting controlled! Gap < 0.03")
        elif gen_gap < 0.05:
            print(f"   GOOD: Reasonable control. Gap < 0.05")
        else:
            print(f"   NEEDS WORK: Still overfitting. Gap = {gen_gap:.4f}")

        # Check prediction sanity
        if "TensorFlow" in test_predictions and "Linear Regression" in test_predictions:
            tf_pred = test_predictions["TensorFlow"]
            lr_pred = test_predictions["Linear Regression"]
            pred_ratio = tf_pred / lr_pred

            print(f"\nPREDICTION SANITY CHECK:")
            print(f"   TensorFlow prediction: ${tf_pred:,.0f}")
            print(f"   Linear Regression: ${lr_pred:,.0f}")
            print(f"   Ratio: {pred_ratio:.2f}")

            if 0.5 <= pred_ratio <= 2.0:
                print(f"   SANE: Predictions within reasonable range")
            else:
                print(f"   INSANE: Predictions still extreme!")

        print(f"\nNEXT STEPS:")
        print(f"1. Check the bulletproof results above")
        print(f"2. TensorFlow should now predict reasonable values")
        print(f"3. Generalization gap should be < 0.03")
        print(f"4. Run your Streamlit app to test real predictions")
        print(f"5. If still overfitting, use Linear/Ridge instead")

        print(f"\nFILES SAVED:")
        print(f"All 5 models saved to 'saved_models/' directory")
        print(f"BULLETPROOF TensorFlow with extreme regularization")
        print(f"Ready for instant loading in your Streamlit app")

        # Architecture summary
        print(f"\nBULLETPROOF TENSORFLOW ARCHITECTURE:")
        print(f"   Input → 6 neurons (relu) → Output")
        print(f"   L1+L2 regularization: 0.01 (extreme)")
        print(f"   Dropout: 0.8 (drops 80% of neurons)")
        print(f"   Learning rate: 0.00005 (ultra-low)")
        print(f"   Early stopping: 30 patience")
        print(f"   Validation split: 30% (massive monitoring)")

        if gen_gap < 0.03:
            print(f"\nMISSION ACCOMPLISHED!")
            print(f"   Overfitting has been eliminated!")
            print(f"   TensorFlow is now bulletproof!")
        else:
            print(f"\nFALLBACK RECOMMENDATION:")
            print(f"   If TensorFlow still overfits, use Ridge Regression")
            print(f"   Ridge is linear and cannot overfit significantly")

    except Exception as e:
        print(f"Error during bulletproof training: {e}")
        print("Please check your database connection and requirements.")
        import traceback

        traceback.print_exc()
