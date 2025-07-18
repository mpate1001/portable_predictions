# ================================
# FIXED DATABASE CONNECTION SETUP
# ================================

import numpy as np
import pandas as pd
import pickle
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Database connection
import psycopg2
from sqlalchemy import create_engine, text
import urllib.parse

# Core ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# For advanced feature engineering
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression

print("🏠 Housing Price Prediction Model Trainer (Database Version)")
print("=" * 60)

# Database credentials
DB_CONFIG = {
    'host': 'ceq2kf3e33g245.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com',
    'database': 'd9f89h4ju1lleh',
    'user': 'ufnbfacj9c7u80',
    'password': 'pa129f8c5adad53ef2c90db10cce0c899f8c7bdad022cca4e85a8729b19aad68d',
    'port': 5432
}

def create_db_connection():
    """Create database connection using SQLAlchemy"""
    try:
        # URL encode the password to handle special characters
        password = urllib.parse.quote_plus(DB_CONFIG['password'])
        
        # Create connection string
        connection_string = f"postgresql://{DB_CONFIG['user']}:{password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        
        # Create engine
        engine = create_engine(connection_string)
        
        print("✅ Database connection established successfully")
        return engine
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def test_db_connection():
    """Test database connection and show available tables"""
    try:
        engine = create_db_connection()
        if engine is None:
            return False
            
        # Test connection with a simple query - FIXED VERSION
        with engine.connect() as conn:
            # Use text() wrapper for raw SQL queries in newer SQLAlchemy versions
            result = conn.execute(text("SELECT current_database(), current_user;"))
            db_info = result.fetchone()
            print(f"✅ Connected to database: {db_info[0]} as user: {db_info[1]}")
            
            # Check available tables - FIXED VERSION
            tables_result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """))
            
            tables = [row[0] for row in tables_result.fetchall()]
            print(f"📊 Available tables: {tables}")
            
        return True
        
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

def load_and_preprocess_data():
    """Load and preprocess housing and crime data from database"""
    
    print("📊 Loading datasets from database...")
    
    # Create database connection
    engine = create_db_connection()
    if engine is None:
        print("❌ Cannot connect to database!")
        return None, None
    
    try:
        # Load housing data from database
        print("   Loading housing data...")
        housing_query = "SELECT * FROM acs_housing_vw;"
        housing_df = pd.read_sql_query(housing_query, engine)
        print(f"✅ Loaded {len(housing_df):,} housing records from database")
        
        # Load crime data from database  
        print("   Loading crime data...")
        crime_query = "SELECT * FROM crime_data;"
        crime_df = pd.read_sql_query(crime_query, engine)
        print(f"✅ Loaded {len(crime_df):,} crime records from database")
        
    except Exception as e:
        print(f"❌ Error loading data from database: {e}")
        return None, None
    
    finally:
        engine.dispose()
    
    print("\n🔧 Preprocessing data...")
    
    # Clean housing data
    housing_df = housing_df.copy()
    
    # Remove invalid property values
    initial_count = len(housing_df)
    housing_df = housing_df[
        (housing_df['valp'] > 0) & 
        (housing_df['valp'] < 5000000) &  # Remove extreme outliers
        (housing_df['hincp'] >= 0) &
        (housing_df['zip'].notna()) &
        (housing_df['county'].notna())
    ]
    print(f"   Filtered {initial_count - len(housing_df):,} invalid records")
    
    # Clean county names in housing data (remove "County" suffix if present)
    housing_df['county_clean'] = housing_df['county'].str.replace(' County', '').str.strip()
    
    # Process crime data
    crime_df = crime_df.copy()
    
    # Clean county names in crime data
    crime_df['county_clean'] = crime_df['county'].str.replace(' County', '').str.strip()
    
    # Get latest crime data by county
    latest_crime = crime_df.loc[crime_df.groupby('county_clean')['year'].idxmax()].copy()
    
    # Calculate crime rates and safety scores
    if len(latest_crime) > 0:
        # Rename columns for consistency
        latest_crime = latest_crime.rename(columns={
            'violent_sum': 'violent_crime',
            'property_sum': 'property_crime'
        })
        
        # Handle case where columns might have different names
        if 'violent_sum' not in latest_crime.columns and 'Violent_sum' in latest_crime.columns:
            latest_crime = latest_crime.rename(columns={
                'Violent_sum': 'violent_crime',
                'Property_sum': 'property_crime'
            })
        
        # Calculate safety scores (higher is safer)
        max_violent = latest_crime['violent_crime'].quantile(0.95)
        max_property = latest_crime['property_crime'].quantile(0.95)
        
        latest_crime['violent_rate'] = latest_crime['violent_crime']
        latest_crime['property_rate'] = latest_crime['property_crime']
        latest_crime['safety_score'] = 100 - (
            (latest_crime['violent_crime'] / max(max_violent, 1) * 40) + 
            (latest_crime['property_crime'] / max(max_property, 1) * 60)
        ).clip(0, 100)
        
        print(f"   Processed crime data for {len(latest_crime)} counties")
    
    # Merge housing and crime data
    print("\n🔗 Merging datasets...")
    
    # Merge on county first
    merged_df = housing_df.merge(
        latest_crime[['county_clean', 'violent_rate', 'property_rate', 'safety_score']], 
        on='county_clean', 
        how='left'
    )
    
    # Fill missing crime data with median values
    for col in ['violent_rate', 'property_rate', 'safety_score']:
        median_val = merged_df[col].median()
        merged_df[col] = merged_df[col].fillna(median_val)
        print(f"   Filled {col} missing values with median: {median_val:.1f}")
    
    print(f"✅ Final dataset: {len(merged_df):,} records")
    print(f"   Counties covered: {merged_df['county_clean'].nunique()}")
    print(f"   ZIP codes covered: {merged_df['zip'].nunique()}")
    
    return merged_df, latest_crime

def engineer_features(df):
    """Create additional features for better predictions"""
    
    print("\n⚙️ Engineering features...")
    
    df = df.copy()
    
    # Basic derived features
    df['rooms_per_person'] = df['nr'] / np.maximum(df['np'], 1)
    df['income_to_value_ratio'] = df['hincp'] / np.maximum(df['valp'], 1)
    df['age_category'] = pd.cut(df['house_age'], bins=[0, 10, 30, 50, 100], labels=['New', 'Recent', 'Mature', 'Old'])
    
    # Income features
    df['log_income'] = np.log1p(df['hincp'])
    df['income_per_person'] = df['hincp'] / np.maximum(df['np'], 1)
    
    # Property features
    df['bedrooms_per_person'] = df['bds'] / np.maximum(df['np'], 1)
    df['is_large_household'] = (df['np'] > 4).astype(int)
    
    # Location-based features
    df['high_crime_area'] = (df['safety_score'] < 50).astype(int)
    df['safety_tier'] = pd.cut(df['safety_score'], bins=[0, 40, 60, 80, 100], labels=['High Risk', 'Moderate', 'Safe', 'Very Safe'])
    
    # Encode categorical features
    le_age = LabelEncoder()
    le_safety = LabelEncoder()
    
    df['age_category_encoded'] = le_age.fit_transform(df['age_category'].astype(str))
    df['safety_tier_encoded'] = le_safety.fit_transform(df['safety_tier'].astype(str))
    
    print(f"   Created additional engineered features")
    
    return df

def prepare_model_data(df):
    """Prepare final dataset for modeling"""
    
    print("\n📋 Preparing modeling dataset...")
    
    # Select features for modeling
    feature_columns = [
        'hincp',           # Household income
        'fincp',           # Family income  
        'bds',             # Bedrooms
        'nr',              # Number of rooms
        'np',              # Number of people
        'house_age',       # Age of house
        'violent_rate',    # Violent crime rate
        'property_rate',   # Property crime rate
        'safety_score',    # Safety score
        'rooms_per_person', # Derived feature
        'income_to_value_ratio', # Derived feature
    ]
    
    # Ensure all features exist
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        return None, None, None
    
    # Prepare feature matrix and target
    X = df[feature_columns].copy()
    y = np.log1p(df['valp'])  # Log transform target for better distribution
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    print(f"   Features: {list(X.columns)}")
    print(f"   Samples: {len(X):,}")
    print(f"   Target range: ${np.exp(y.min()):.0f} - ${np.exp(y.max()):,.0f}")
    
    return X, y, feature_columns

def train_models(X, y, features):
    """Train multiple models and evaluate performance"""
    
    print("\n🤖 Training models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        try:
            # Use scaled data for linear models, original for tree-based
            if name in ['Linear Regression', 'Ridge Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled if name in ['Linear Regression', 'Ridge Regression'] else X_train, 
                y_train, cv=5, scoring='r2'
            )
            
            results[name] = {
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'CV_R2_Mean': cv_scores.mean(),
                'CV_R2_Std': cv_scores.std(),
                'Generalization_Gap': train_r2 - test_r2
            }
            
            trained_models[name] = model
            
            print(f"      R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            continue
    
    return trained_models, scaler, results, X_test, y_test

def save_models(models, scaler, results, X_test, y_test, features):
    """Save trained models and metadata"""
    
    print("\n💾 Saving models...")
    
    # Create directory
    os.makedirs('saved_models', exist_ok=True)
    
    # Save individual models
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '_model.pkl'
        filepath = os.path.join('saved_models', filename)
        joblib.dump(model, filepath)
        print(f"   ✅ Saved {name}")
    
    # Save scaler
    scaler_path = os.path.join('saved_models', 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Saved scaler")
    
    # Save features list
    features_path = os.path.join('saved_models', 'features.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"   ✅ Saved features")
    
    # Save test data for diagnostics
    X_test_path = os.path.join('saved_models', 'X_test.pkl')
    y_test_path = os.path.join('saved_models', 'y_test.pkl')
    joblib.dump(X_test, X_test_path)
    joblib.dump(y_test, y_test_path)
    print(f"   ✅ Saved test data for diagnostics")
    
    # Save metadata and results
    metadata = {
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_performance': results,
        'feature_names': features,
        'data_summary': {
            'n_samples': len(X_test) * 5,  # Approximate total samples
            'n_features': len(features),
            'target_transform': 'log1p'
        },
        'database_info': {
            'host': DB_CONFIG['host'],
            'database': DB_CONFIG['database'],
            'user': DB_CONFIG['user'],
            'connection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    metadata_path = os.path.join('saved_models', 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✅ Saved metadata with database info")
    
    return metadata

def display_results(results):
    """Display training results summary"""
    
    print("\n" + "=" * 60)
    print("📊 MODEL TRAINING RESULTS (Database Version)")
    print("=" * 60)
    
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    # Find best model
    if 'Test_R2' in results_df.columns:
        best_model = results_df['Test_R2'].idxmax()
        best_r2 = results_df.loc[best_model, 'Test_R2']
        print(f"\n🏆 Best Model: {best_model} (R² = {best_r2:.4f})")
    
    print("\n✅ All models saved to 'saved_models/' directory")
    print("🚀 Ready to run Streamlit app!")
    print("📊 Models trained on live database data")

def main():
    """Main training pipeline with database connection"""
    
    print("Starting database-connected model training pipeline...\n")
    
    # Test database connection first
    if not test_db_connection():
        print("❌ Cannot proceed without database connection!")
        return
    
    print("\n" + "=" * 60)
    
    # Load and preprocess data
    df, crime_df = load_and_preprocess_data()
    if df is None:
        return
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare model data
    X, y, features = prepare_model_data(df)
    if X is None:
        return
    
    # Train models
    models, scaler, results, X_test, y_test = train_models(X, y, features)
    
    if not models:
        print("❌ No models were successfully trained!")
        return
    
    # Save everything
    metadata = save_models(models, scaler, results, X_test, y_test, features)
    
    # Display results
    display_results(results)
    
    print(f"\n🎉 Database-connected training completed successfully!")
    print(f"   Database: {DB_CONFIG['database']} on {DB_CONFIG['host']}")
    print(f"   Models saved: {len(models)}")
    print(f"   Best performing model ready for predictions")
    print(f"   Run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    # Install required packages reminder
    required_packages = ['psycopg2-binary', 'sqlalchemy']
    print("📦 Required packages for database connection:")
    for package in required_packages:
        print(f"   pip install {package}")
    print()
    
    main()