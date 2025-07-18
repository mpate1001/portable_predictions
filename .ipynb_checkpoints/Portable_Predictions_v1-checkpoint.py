# ================================
# Portable Predictions: Enhanced Housing Investment Streamlit App
# Compatible with both CSV and Database-trained models
# Authors: Joe Bryant, Mahek Patel, Nathan Deering
# ================================

# Core Libraries
import os
import pickle
import joblib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit Mapping
import folium
from streamlit_folium import folium_static

# Machine Learning Evaluation
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Database Support
try:
    import psycopg2
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import URL  # Required if using SQLAlchemy's URL.create()
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# SHAP Support
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Portable Predictions: Housing Investment Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1.3rem;
    color: #555;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    text-align: center;
}
.recommendation-card {
    padding: 2rem;
    border-radius: 1rem;
    text-align: center;
    margin: 1rem 0;
    border: 2px solid;
}
</style>
""", unsafe_allow_html=True)

# ================================
# DATABASE CONNECTION (Optional)
# ================================

@st.cache_resource
def create_db_connection():
    """Create database connection using Streamlit Secrets"""
    try:
        cfg = st.secrets["postgres"]

        # Safely build the connection URL
        url = URL.create(
            "postgresql+psycopg2",
            username=cfg["user"],
            password=cfg["password"],
            host=cfg["host"],
            port=cfg.get("port", 5432),
            database=cfg["database"],
            query={"sslmode": "require"},
        )

        engine = create_engine(url, pool_pre_ping=True)

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return engine
    except Exception as e:
        st.warning(f"Database connection failed: {e}. Using CSV fallback.")
        return None
# ================================
# DATA LOADING FUNCTIONS
# ================================

@st.cache_data
def load_datasets():
    """Load housing and crime datasets from database or CSV files"""
    
    datasets = {}
    data_source = "Unknown"
    
    # Try database first
    if DATABASE_AVAILABLE:
        engine = create_db_connection()
        if engine:
            try:
                st.info("Loading data from database...")
                
                # Load housing data
                housing_query = "SELECT * FROM acs_housing_vw;"
                housing_df = pd.read_sql_query(housing_query, engine)
                
                # Load crime data
                crime_query = "SELECT * FROM crime_data;"
                crime_df = pd.read_sql_query(crime_query, engine)
                
                data_source = "Database"
                st.success(f"Loaded from database: {len(housing_df):,} housing records, {len(crime_df):,} crime records")
                
            except Exception as e:
                st.warning(f"Database loading failed: {e}. Trying CSV files...")
                engine = None
            finally:
                if engine:
                    engine.dispose()
    
    # Fallback to CSV files
    if 'housing_df' not in locals():
        try:
            st.info("Loading data from CSV files...")
            
            # Load housing data
            housing_df = pd.read_csv('acs_housing_vw.csv')
            
            # Load crime data
            crime_df = pd.read_csv('crime_data.csv')
            
            data_source = "CSV Files"
            st.success(f"Loaded from CSV: {len(housing_df):,} housing records, {len(crime_df):,} crime records")
            
        except Exception as e:
            st.error(f"Error loading datasets: {e}")
            return {}
    
    # Process housing data
    housing_df = housing_df.copy()
    
    # Clean county names
    housing_df['county_clean'] = housing_df['county'].str.replace(' County', '').str.strip()
    
    # Remove invalid property values
    initial_count = len(housing_df)
    housing_df = housing_df[
        (housing_df['valp'] > 0) & 
        (housing_df['valp'] < 5000000) &
        (housing_df['hincp'] >= 0) &
        (housing_df['zip'].notna()) &
        (housing_df['county'].notna())
    ]
    
    if initial_count != len(housing_df):
        st.info(f"Filtered {initial_count - len(housing_df):,} invalid housing records")
    
    datasets['housing'] = housing_df
    
    # Process crime data
    crime_df = crime_df.copy()
    crime_df['county_clean'] = crime_df['county'].str.replace(' County', '').str.strip()
    
    # Handle different column naming conventions
    crime_columns_map = {
        'Violent_sum': 'violent_crime',
        'Property_sum': 'property_crime',
        'violent_sum': 'violent_crime',
        'property_sum': 'property_crime'
    }
    
    for old_col, new_col in crime_columns_map.items():
        if old_col in crime_df.columns:
            crime_df[new_col] = crime_df[old_col]
    
    # Ensure required columns exist
    if 'violent_crime' not in crime_df.columns:
        crime_df['violent_crime'] = crime_df.get('violent_rate', 200)
    if 'property_crime' not in crime_df.columns:
        crime_df['property_crime'] = crime_df.get('property_rate', 1000)
    
    # Get latest crime data by county
    if 'year' in crime_df.columns:
        latest_crime = crime_df.loc[crime_df.groupby('county_clean')['year'].idxmax()].copy()
    else:
        latest_crime = crime_df.drop_duplicates('county_clean').copy()
    
    # Calculate safety scores
    max_violent = latest_crime['violent_crime'].quantile(0.95)
    max_property = latest_crime['property_crime'].quantile(0.95)
    
    latest_crime['violent_rate'] = latest_crime['violent_crime']
    latest_crime['property_rate'] = latest_crime['property_crime']
    latest_crime['safety_score'] = 100 - (
        (latest_crime['violent_crime'] / max(max_violent, 1) * 40) + 
        (latest_crime['property_crime'] / max(max_property, 1) * 60)
    ).clip(0, 100)
    
    datasets['crime'] = latest_crime
    
    # Get available counties and ZIP codes
    counties = sorted(housing_df['county_clean'].unique())
    zip_county_map = housing_df[['zip', 'county_clean']].drop_duplicates()
    
    datasets['counties'] = counties
    datasets['zip_county_map'] = zip_county_map
    datasets['data_source'] = data_source
    
    return datasets

@st.cache_resource
def load_trained_models():
    """Load pre-trained models and metadata"""
    
    models_dir = 'saved_models'
    
    if not os.path.exists(models_dir):
        st.error("No saved models found! Please run the model trainer first.")
        return None
    
    try:
        st.info("⚡ Loading pre-trained models...")
        
        # Load models
        models = {}
        model_files = {
            'Linear Regression': 'linear_regression_model.pkl',
            'Ridge Regression': 'ridge_regression_model.pkl', 
            'Random Forest': 'random_forest_model.pkl',
            'XGBoost': 'xgboost_model.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                models[name] = joblib.load(filepath)
            else:
                st.warning(f"{filename} not found, skipping {name}")
        
        # Load scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        # Load features
        features_path = os.path.join(models_dir, 'features.pkl')
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        
        # Load test data for diagnostics
        X_test_path = os.path.join(models_dir, 'X_test.pkl')
        y_test_path = os.path.join(models_dir, 'y_test.pkl')
        X_test = joblib.load(X_test_path) if os.path.exists(X_test_path) else None
        y_test = joblib.load(y_test_path) if os.path.exists(y_test_path) else None
        
        # Load metadata
        metadata_path = os.path.join(models_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Display data source info
        if 'database_info' in metadata:
            data_source = "Database"
            st.success(f"Loaded {len(models)} database-trained models")
        else:
            data_source = "CSV"
            st.success(f"Loaded {len(models)} CSV-trained models")
        
        return {
            'models': models,
            'scaler': scaler,
            'features': features,
            'X_test': X_test,
            'y_test': y_test,
            'metadata': metadata,
            'model_source': data_source
        }
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ================================
# ANALYSIS FUNCTIONS
# ================================

def get_zip_codes_by_county(county, zip_county_map):
    """Get ZIP codes for a specific county"""
    county_zips = zip_county_map[zip_county_map['county_clean'] == county]['zip'].unique()
    return sorted(county_zips.tolist())

def get_county_crime_data(county, crime_data):
    """Get crime statistics for a county"""
    county_crime = crime_data[crime_data['county_clean'] == county]
    
    if len(county_crime) > 0:
        return county_crime.iloc[0].to_dict()
    else:
        # Default values
        return {
            'violent_rate': 200,
            'property_rate': 1000,
            'safety_score': 60,
            'violent_crime': 200,
            'property_crime': 1000
        }

def calculate_investment_score(predictions, purchase_price, crime_data, market_context):
    """Calculate comprehensive investment score"""
    
    pred_values = list(predictions.values())
    avg_prediction = np.mean(pred_values)
    prediction_std = np.std(pred_values)
    
    # Price score based on predicted vs purchase price
    price_differential = (avg_prediction - purchase_price) / purchase_price
    price_score = np.tanh(price_differential * 2) * 50 + 50
    
    # Safety score from crime data
    safety_score = crime_data.get('safety_score', 60)
    
    # Model consensus score
    consensus_score = max(0, 100 - (prediction_std / avg_prediction * 100)) if avg_prediction > 0 else 0
    
    # Market context score
    context_score = 50
    if market_context:
        if market_context.get('is_urban', False):
            context_score += 15
        if market_context.get('has_coordinates', False):
            context_score += 10
    
    # Final weighted score
    final_score = (
        price_score * 0.35 +
        safety_score * 0.25 +
        consensus_score * 0.20 +
        context_score * 0.20
    )
    
    final_score = max(0, min(100, final_score))
    
    return final_score, {
        'price_score': price_score,
        'safety_score': safety_score,
        'consensus_score': consensus_score,
        'context_score': context_score,
        'avg_prediction': avg_prediction,
        'prediction_std': prediction_std,
        'price_differential': price_differential
    }

# ================================
# SHAP ANALYSIS
# ================================

@st.cache_resource
def generate_shap_analysis(_models, _X_test, _features):
    """Generate SHAP analysis for interpretability"""
    
    if not SHAP_AVAILABLE or 'XGBoost' not in _models or _X_test is None:
        return {}
    
    try:
        model = _models['XGBoost']
        
        # Use sample of test data for faster SHAP computation
        X_sample = _X_test.sample(n=min(200, len(_X_test)), random_state=42)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'X_sample': X_sample,
            'expected_value': explainer.expected_value,
            'feature_importance': pd.DataFrame({
                'Feature': _features,
                'Mean_SHAP': np.abs(shap_values).mean(0)
            }).sort_values('Mean_SHAP', ascending=False)
        }
    except Exception as e:
        st.error(f"SHAP analysis failed: {e}")
        return {}

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_folium_map(county, zip_code, housing_data, crime_data):
    """Create Folium map with crime heat overlay and ZIP code focus"""
    
    # Get county data
    county_housing = housing_data[housing_data['county_clean'] == county]
    county_crime = get_county_crime_data(county, crime_data)
    
    # Initialize map center and zoom
    if zip_code and zip_code != "No ZIP codes available":
        # Focus on selected ZIP code
        zip_data = county_housing[county_housing['zip'] == int(zip_code)]
        if len(zip_data) > 0 and 'latitude' in zip_data.columns and 'longitude' in zip_data.columns:
            center_lat = zip_data['latitude'].median()
            center_lon = zip_data['longitude'].median()
            zoom_start = 13
        else:
            # Fallback to county center
            if len(county_housing) > 0 and 'latitude' in county_housing.columns:
                center_lat = county_housing['latitude'].median()
                center_lon = county_housing['longitude'].median()
                zoom_start = 10
            else:
                center_lat, center_lon = 36.7783, -119.4179
                zoom_start = 6
    else:
        # County-wide view
        if len(county_housing) > 0 and 'latitude' in county_housing.columns:
            center_lat = county_housing['latitude'].median()
            center_lon = county_housing['longitude'].median()
            zoom_start = 10
        else:
            # Default to California center
            center_lat, center_lon = 36.7783, -119.4179
            zoom_start = 6
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    
    # Add crime-based color coding
    safety_score = county_crime['safety_score']
    
    # Determine color based on safety
    if safety_score >= 80:
        color = 'green'
    elif safety_score >= 60:
        color = 'orange'
    else:
        color = 'red'
    
    # Check if we have coordinate data
    if 'latitude' in county_housing.columns and 'longitude' in county_housing.columns:
        # Group data by ZIP code for cleaner display
        zip_groups = county_housing.groupby('zip').agg({
            'latitude': 'median',
            'longitude': 'median',
            'primary_city': 'first' if 'primary_city' in county_housing.columns else lambda x: 'Unknown',
            'valp': ['mean', 'count']
        }).reset_index()
        
        # Flatten column names
        zip_groups.columns = ['zip', 'latitude', 'longitude', 'primary_city', 'avg_price', 'property_count']
        
        # Add ZIP code markers with labels
        for _, row in zip_groups.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                
                # Highlight selected ZIP code
                if zip_code and str(row['zip']) == str(zip_code):
                    marker_color = 'blue'
                    marker_size = 15
                    icon_color = 'white'
                    popup_prefix = "SELECTED ZIP: "
                else:
                    marker_color = color
                    marker_size = 10
                    icon_color = 'white'
                    popup_prefix = "ZIP: "
                
                # Create popup with detailed information
                popup_text = f"""
                {popup_prefix}{row['zip']}<br>
                City: {row['primary_city']}<br>
                Properties: {row['property_count']} listings<br>
                Avg Price: ${row['avg_price']:,.0f}<br>
                Safety Score: {safety_score:.1f}/100<br>
                Violent Crime: {county_crime['violent_rate']:.0f}<br>
                Property Crime: {county_crime['property_rate']:.0f}
                """
                
                # Add marker with ZIP code as icon
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"ZIP: {row['zip']} | {row['primary_city']}",
                    icon=folium.DivIcon(
                        html=f"""
                        <div style="
                            background-color: {marker_color};
                            border: 2px solid white;
                            border-radius: 8px;
                            color: {icon_color};
                            font-size: 12px;
                            font-weight: bold;
                            text-align: center;
                            padding: 4px 6px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                            min-width: 50px;
                        ">
                            {row['zip']}
                        </div>
                        """,
                        icon_size=(60, 30),
                        icon_anchor=(30, 15)
                    )
                ).add_to(m)
    else:
        # Add text overlay if no coordinates
        no_coords_html = """
        <div style="
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255,255,255,0.9);
            border: 2px solid #ccc;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            font-size: 16px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            z-index: 1000;
        ">
            <strong>Map View Not Available</strong><br>
            No coordinate data for this county.<br>
            Please use the analysis tools below.
        </div>
        """
        m.get_root().html.add_child(folium.Element(no_coords_html))
    
    # Add county info box
    county_info = f"""
    <div style="
        position: fixed;
        top: 10px;
        left: 50px;
        background: white;
        border: 2px solid {color};
        border-radius: 8px;
        padding: 10px;
        font-size: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        z-index: 1000;
    ">
        <strong>{county} County</strong><br>
        🛡️ Safety Score: {safety_score:.1f}/100<br>
        Properties: {len(county_housing):,} listings
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(county_info))
    
    # Add legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        background: white;
        border: 2px solid #ccc;
        border-radius: 8px;
        padding: 10px;
        font-size: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        z-index: 1000;
    ">
        <strong>Map Legend</strong><br>
        <span style="color: green;">●</span> Safe Areas (80+ score)<br>
        <span style="color: orange;">●</span> Moderate Areas (60-79)<br>
        <span style="color: red;">●</span> High Risk Areas (<60)<br>
        <span style="color: blue;">●</span> Selected ZIP Code<br>
        <small>Click ZIP codes for details</small>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def display_model_diagnostics(models, X_test, y_test):
    """Display comprehensive model diagnostic plots"""
    
    if y_test is None or X_test is None:
        st.warning("Test data not available for diagnostics")
        return
    
    st.header("Model Diagnostics")
    
    # Use XGBoost as primary model for diagnostics
    if 'XGBoost' in models:
        model = models['XGBoost']
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted (Log scale)
            fig_ap = px.scatter(
                x=y_test, y=y_pred,
                labels={'x': 'Actual Log(Value+1)', 'y': 'Predicted Log(Value+1)'},
                title="Actual vs Predicted (Log Scale)"
            )
            # Add perfect prediction line
            min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig_ap.add_shape(
                type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                line=dict(color="red", width=2, dash="dash")
            )
            st.plotly_chart(fig_ap, use_container_width=True)
            
        with col2:
            # Residuals plot
            residuals = y_test - y_pred
            fig_res = px.scatter(
                x=y_pred, y=residuals,
                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                title="Residuals Plot"
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)
        
        # Histogram of residuals
        fig_hist = px.histogram(
            residuals, nbins=30,
            title="Histogram of Residuals",
            labels={'value': 'Residuals', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

def display_shap_analysis(shap_results, user_input, features):
    """Display SHAP interpretability plots"""
    
    if not shap_results or not SHAP_AVAILABLE:
        st.warning("SHAP analysis not available")
        return
    
    st.header("Model Interpretability (SHAP)")
    
    # Feature importance
    importance_df = shap_results['feature_importance']
    fig_imp = px.bar(
        importance_df.head(10),
        x='Mean_SHAP', y='Feature',
        orientation='h',
        title='Top 10 Feature Importance (SHAP)',
        labels={'Mean_SHAP': 'Mean |SHAP Value|'}
    )
    fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # SHAP summary plot
    if len(shap_results) > 0:
        st.subheader("SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_results['shap_values'], 
            shap_results['X_sample'], 
            plot_type="dot", 
            show=False
        )
        st.pyplot(fig, bbox_inches='tight')

        # SHAP dependence plots
        st.subheader("SHAP Dependence Plots")
        top_features = importance_df['Feature'].head(5).tolist()
        selected_feature = st.selectbox("Select feature for dependence plot:", top_features)
        
        if selected_feature:
            fig_dep, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(
                selected_feature, 
                shap_results['shap_values'], 
                shap_results['X_sample'], 
                show=False
            )
            st.pyplot(fig_dep, bbox_inches='tight')

# ================================
# MAIN SECTIONS - PART 1
# ================================

def display_property_analysis(datasets, model_data):
    """Main property analysis interface"""
    
    # Main interface
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("Property Analysis")
        
        # Location selection
        st.subheader("Location")
        location_col1, location_col2 = st.columns(2)
        
        with location_col1:
            selected_county = st.selectbox(
                "Select County",
                options=datasets['counties'],
                help="Choose county for analysis"
            )
        
        with location_col2:
            county_zips = get_zip_codes_by_county(selected_county, datasets['zip_county_map'])
            selected_zip = st.selectbox(
                "Select ZIP Code",
                options=county_zips,
                help="ZIP codes in selected county - map will auto-zoom to selection",
                key="zip_selector"
            )
            
        # Display ZIP code details (outside of columns to avoid nesting)
        if selected_zip and selected_zip != "No ZIP codes available":
            zip_data = datasets['housing'][datasets['housing']['zip'] == int(selected_zip)]
            if len(zip_data) > 0:
                avg_price = zip_data['valp'].mean()
                property_count = len(zip_data)
                primary_city = zip_data['primary_city'].iloc[0] if 'primary_city' in zip_data.columns else 'Unknown'
                
                st.info(f"**ZIP {selected_zip}** | {primary_city} | {property_count} properties | Avg: ${avg_price:,.0f}")
           
        
        # Property characteristics
        st.subheader("Property Details")
        prop_col1, prop_col2, prop_col3 = st.columns(3)
        
        with prop_col1:
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
            total_rooms = st.number_input("Total Rooms", min_value=1, max_value=20, value=6)
        
        with prop_col2:
            num_people = st.number_input("Household Size", min_value=1, max_value=15, value=3)
            year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)
        
        with prop_col3:
            household_income = st.number_input("Household Income ($)", min_value=0, value=75000, step=1000)
            family_income = st.number_input("Family Income ($)", min_value=0, value=85000, step=1000)
        
        purchase_price = st.number_input(
            "Purchase Price ($)", 
            min_value=50000, 
            value=500000, 
            step=10000
        )
    
    with col2:
        st.header("Analysis Options")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Full Investment Analysis", "Model Comparison", "SHAP Interpretability", "Model Diagnostics"]
        )
        
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)
        
        # Display county crime info
        if selected_county:
            crime_data = get_county_crime_data(selected_county, datasets['crime'])
            
            st.markdown("### Area Context")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Safety Score", f"{crime_data['safety_score']:.1f}/100")
                st.metric("Violent Crime", f"{crime_data['violent_rate']:.0f}")
            with col_b:
                st.metric("Property Crime", f"{crime_data['property_rate']:.0f}")
                st.metric("ZIP Code", selected_zip)
    
    # Enhanced Crime heat map with ZIP code focus
    st.subheader("Interactive Crime & Property Map")
    
    # ZIP code statistics before map (to avoid nested columns)
    if selected_zip and selected_zip != "No ZIP codes available":
        zip_data = datasets['housing'][datasets['housing']['zip'] == int(selected_zip)]
        if len(zip_data) > 0:
            # Create metrics row for selected ZIP
            zip_col1, zip_col2, zip_col3, zip_col4, zip_col5 = st.columns(5)
            
            zip_stats = {
                'median_price': zip_data['valp'].median(),
                'avg_bedrooms': zip_data['bds'].mean(),
                'avg_house_age': zip_data['house_age'].mean(),
                'avg_income': zip_data['hincp'].mean(),
                'property_count': len(zip_data)
            }
            
            with zip_col1:
                st.metric("Properties", f"{zip_stats['property_count']}")
            with zip_col2:
                st.metric("Median Price", f"${zip_stats['median_price']:,.0f}")
            with zip_col3:
                st.metric("Avg Bedrooms", f"{zip_stats['avg_bedrooms']:.1f}")
            with zip_col4:
                st.metric("House Age", f"{zip_stats['avg_house_age']:.0f}y")
            with zip_col5:
                st.metric("Avg Income", f"${zip_stats['avg_income']:,.0f}")
    
    # Map display
    map_info_col1, map_info_col2 = st.columns([3, 1])
    
    with map_info_col1:
        if selected_county:
            folium_map = create_folium_map(selected_county, selected_zip, datasets['housing'], datasets['crime'])
            folium_static(folium_map, width=700, height=500)
        else:
            st.info("Please select a county to view the map.")
    
    with map_info_col2:
        st.markdown("### Map Features")
        st.markdown("""
        **Auto-Zoom**: Map focuses on selected ZIP code
        
        **ZIP Labels**: ZIP codes displayed as text markers
        
        **Color Coding**:
        - 🟢 Safe areas (80+ score)
        - 🟡 Moderate (60-79 score)  
        - 🔴 High risk (<60 score)
        - 🔵 Selected ZIP code
        
        **Interactive**: Click ZIP codes for detailed property information
        """)
        
        if selected_county:
            crime_data = get_county_crime_data(selected_county, datasets['crime'])
            st.markdown("---")
            st.markdown("### County Safety")
            st.metric("Safety Score", f"{crime_data['safety_score']:.1f}/100")
            st.metric("Violent Crime", f"{crime_data['violent_rate']:.0f}")
            st.metric("Property Crime", f"{crime_data['property_rate']:.0f}")
    
    # Analysis execution
    if run_analysis and selected_county:
        # Get crime data
        crime_data = get_county_crime_data(selected_county, datasets['crime'])
        
        # Calculate derived features
        house_age = 2024 - year_built
        rooms_per_person = total_rooms / max(num_people, 1)
        income_to_value_ratio = household_income / max(purchase_price, 1)
        
        # Prepare input vector
        user_input = [
            household_income,              # hincp
            family_income,                 # fincp
            bedrooms,                      # bds
            total_rooms,                   # nr (total rooms)
            num_people,                    # np
            house_age,                     # house_age
            crime_data['violent_rate'],    # violent_rate
            crime_data['property_rate'],   # property_rate
            crime_data['safety_score'],    # safety_score
            rooms_per_person,              # rooms_per_person
            income_to_value_ratio          # income_to_value_ratio
        ]
        
        # Make predictions
        predictions = {}
        input_df = pd.DataFrame([user_input], columns=model_data['features'])
        
        for name, model in model_data['models'].items():
            try:
                # Use scaled data for linear models
                if name in ['Linear Regression', 'Ridge Regression'] and model_data['scaler']:
                    input_scaled = model_data['scaler'].transform(input_df)
                    pred_log = model.predict(input_scaled)[0]
                else:
                    pred_log = model.predict(input_df)[0]
                
                # Convert from log space
                pred_value = np.exp(pred_log) - 1
                predictions[name] = max(0, pred_value)  # Ensure positive
                
            except Exception as e:
                st.error(f"Error with {name}: {e}")
                continue
        
        # Display results based on analysis type
        if analysis_type == "Full Investment Analysis":
            display_full_investment_analysis(predictions, purchase_price, crime_data, selected_county)
        elif analysis_type == "Model Comparison":
            display_model_performance(model_data['metadata'])
            display_prediction_comparison(predictions, purchase_price)
        elif analysis_type == "SHAP Interpretability":
            shap_results = generate_shap_analysis(model_data['models'], model_data['X_test'], model_data['features'])
            display_shap_analysis(shap_results, input_df, model_data['features'])
        elif analysis_type == "Model Diagnostics":
            display_model_diagnostics(model_data['models'], model_data['X_test'], model_data['y_test'])
            
def display_data_explorer(datasets):
    """Comprehensive data exploration dashboard"""
    
    st.header("Dataset Overview & Market Analysis")
    
    # High-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", f"{len(datasets['housing']):,}")
    with col2:
        st.metric("Counties", f"{len(datasets['counties'])}")
    with col3:
        st.metric("ZIP Codes", f"{datasets['housing']['zip'].nunique():,}")
    with col4:
        avg_price = datasets['housing']['valp'].mean()
        st.metric("Avg Property Value", f"${avg_price:,.0f}")
    
    # Data source indicator
    st.info(f"**Data Source**: {datasets.get('data_source', 'Unknown')}")
    
    # Geographic coverage
    st.subheader("Geographic Coverage")
    
    # Check if coordinate data exists
    has_coordinates = 'latitude' in datasets['housing'].columns and 'longitude' in datasets['housing'].columns
    if has_coordinates:
        coord_coverage = datasets['housing'].dropna(subset=['latitude', 'longitude']).shape[0]
        coverage_pct = (coord_coverage / len(datasets['housing'])) * 100
    else:
        coord_coverage = 0
        coverage_pct = 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Properties with Coordinates", f"{coord_coverage:,}", f"{coverage_pct:.1f}%")
    with col2:
        st.metric("Crime Data Coverage", f"{len(datasets['crime'])} counties")
    with col3:
        price_range = f"${datasets['housing']['valp'].min():,.0f} - ${datasets['housing']['valp'].max():,.0f}"
        st.metric("Price Range", price_range)
    
    # Price distribution by county
    st.subheader("Property Values by County")
    
    county_stats = datasets['housing'].groupby('county_clean').agg({
        'valp': ['count', 'mean', 'median', 'std'],
        'hincp': 'mean'
    }).round(0)
    
    county_stats.columns = ['Count', 'Mean_Price', 'Median_Price', 'Std_Price', 'Avg_Income']
    county_stats = county_stats.sort_values('Mean_Price', ascending=False)
    
    st.dataframe(county_stats, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        fig = px.histogram(
            datasets['housing'], 
            x='valp', 
            nbins=50,
            title='Property Value Distribution',
            labels={'valp': 'Property Value ($)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Income vs Property Value
        sample_data = datasets['housing'].sample(n=min(1000, len(datasets['housing'])))
        fig = px.scatter(
            sample_data,
            x='hincp', y='valp',
            title='Income vs Property Value',
            labels={'hincp': 'Household Income ($)', 'valp': 'Property Value ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Property characteristics analysis
    st.subheader("Property Characteristics")
    
    char_col1, char_col2 = st.columns(2)
    
    with char_col1:
        # Bedrooms distribution
        fig = px.histogram(
            datasets['housing'],
            x='bds',
            title='Bedrooms Distribution',
            labels={'bds': 'Number of Bedrooms', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with char_col2:
        # House age distribution
        fig = px.histogram(
            datasets['housing'],
            x='house_age',
            title='House Age Distribution',
            labels={'house_age': 'House Age (Years)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Crime analysis
    st.subheader("Crime & Safety Analysis")
    
    crime_col1, crime_col2 = st.columns(2)
    
    with crime_col1:
        crime_stats = datasets['crime'][['violent_rate', 'property_rate', 'safety_score']].describe()
        st.dataframe(crime_stats.T, use_container_width=True)
    
    with crime_col2:
        # Safety score distribution
        fig = px.histogram(
            datasets['crime'],
            x='safety_score',
            title='Safety Score Distribution by County',
            labels={'safety_score': 'Safety Score', 'count': 'Number of Counties'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # County-level crime comparison
    st.subheader("Top Safest Counties")
    crime_viz = datasets['crime'].nlargest(10, 'safety_score')[['county_clean', 'safety_score', 'violent_rate', 'property_rate']]
    
    fig = px.bar(
        crime_viz,
        x='county_clean', y='safety_score',
        title='Top 10 Safest Counties by Safety Score',
        labels={'county_clean': 'County', 'safety_score': 'Safety Score'},
        color='safety_score',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data quality assessment
    st.subheader("Data Quality Assessment")
    
    quality_col1, quality_col2 = st.columns(2)
    
    with quality_col1:
        st.write("**Missing Data Analysis**")
        missing_data = datasets['housing'].isnull().sum()
        missing_pct = (missing_data / len(datasets['housing']) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing data found!")
    
    with quality_col2:
        st.write("**Data Types & Coverage**")
        coverage_info = pd.DataFrame({
            'Metric': [
                'Total Records',
                'Complete Property Values',
                'Complete Income Data',
                'Complete Geographic Data',
                'Crime Data Coverage'
            ],
            'Count': [
                len(datasets['housing']),
                datasets['housing']['valp'].notna().sum(),
                datasets['housing']['hincp'].notna().sum(),
                coord_coverage,
                len(datasets['crime'])
            ]
        })
        st.dataframe(coverage_info, use_container_width=True)
        
def display_model_insights(model_data):
    """Enhanced model performance and insights"""
    
    st.header("Model Performance & Technical Insights")
    
    if not model_data or 'metadata' not in model_data:
        st.warning("Model data not available")
        return
    
    # Training info
    metadata = model_data['metadata']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Trained", len(model_data['models']))
    with col2:
        st.metric("Features Used", len(model_data['features']))
    with col3:
        trained_at = metadata.get('trained_at', 'Unknown')
        st.metric("Last Trained", trained_at.split()[0] if ' ' in trained_at else trained_at)
    with col4:
        data_summary = metadata.get('data_summary', {})
        st.metric("Training Samples", f"{data_summary.get('n_samples', 'N/A'):,}")
    
    # Display model source
    model_source = model_data.get('model_source', 'Unknown')
    if 'database_info' in metadata:
        db_info = metadata['database_info']
        st.info(f"**Models trained from Database**: {db_info.get('database', 'Unknown')} on {db_info.get('host', 'Unknown')}")
    else:
        st.info(f"**Models trained from**: {model_source}")
    
    # Feature importance (if available)
    st.subheader("Feature Importance Analysis")
    
    if 'XGBoost' in model_data['models'] and model_data['X_test'] is not None:
        model = model_data['models']['XGBoost']
        
        # Get feature importances
        importance_scores = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': model_data['features'],
            'Importance': importance_scores
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_importance.tail(10),
            x='Importance', y='Feature',
            orientation='h',
            title='Top 10 Most Important Features (XGBoost)',
            color='Importance',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.dataframe(feature_importance.sort_values('Importance', ascending=False), use_container_width=True)
    
    # Model comparison details
    if 'model_performance' in metadata:
        st.subheader("Detailed Model Performance Comparison")
        
        results_df = pd.DataFrame(metadata['model_performance']).T
        
        # Enhanced performance visualization
        if 'Test_R2' in results_df.columns and 'CV_R2_Mean' in results_df.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Test R²',
                x=results_df.index,
                y=results_df['Test_R2'],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='CV R² Mean',
                x=results_df.index,
                y=results_df['CV_R2_Mean'],
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title='Model Performance Comparison (R² Scores)',
                xaxis_title='Model',
                yaxis_title='R² Score',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # RMSE comparison
        if 'Test_RMSE' in results_df.columns:
            fig_rmse = px.bar(
                x=results_df.index,
                y=results_df['Test_RMSE'],
                title='Model RMSE Comparison (Lower is Better)',
                labels={'x': 'Model', 'y': 'RMSE'},
                color=results_df['Test_RMSE'],
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Full metrics table
        st.subheader("Complete Performance Metrics")
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Best model highlight
        if 'Test_R2' in results_df.columns:
            best_model = results_df['Test_R2'].idxmax()
            best_r2 = results_df.loc[best_model, 'Test_R2']
            best_rmse = results_df.loc[best_model, 'Test_RMSE'] if 'Test_RMSE' in results_df.columns else 'N/A'
            
            st.success(f"**Best Performing Model**: {best_model}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best R² Score", f"{best_r2:.4f}")
            with col2:
                if best_rmse != 'N/A':
                    st.metric("Corresponding RMSE", f"{best_rmse:.4f}")
    
    # Technical details
    st.subheader("Technical Configuration")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.write("**Model Architecture**")
        model_info = []
        for name, model in model_data['models'].items():
            model_type = type(model).__name__
            model_info.append({'Model': name, 'Type': model_type})
        
        model_df = pd.DataFrame(model_info)
        st.dataframe(model_df, use_container_width=True)
    
    with tech_col2:
        st.write("**Data Processing**")
        processing_info = pd.DataFrame({
            'Step': [
                'Target Transformation',
                'Feature Scaling',
                'Train-Test Split',
                'Cross Validation'
            ],
            'Configuration': [
                data_summary.get('target_transform', 'log1p'),
                'StandardScaler for Linear Models',
                '80-20 Split',
                '5-Fold CV'
            ]
        })
        st.dataframe(processing_info, use_container_width=True)
    
    # Model predictions distribution (if test data available)
    if model_data['X_test'] is not None and model_data['y_test'] is not None:
        st.subheader("Prediction Analysis")
        
        # Generate predictions for all models
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            # Prediction distributions
            predictions_data = []
            
            for name, model in model_data['models'].items():
                if name in ['Linear Regression', 'Ridge Regression'] and model_data['scaler']:
                    X_scaled = model_data['scaler'].transform(model_data['X_test'])
                    preds = model.predict(X_scaled)
                else:
                    preds = model.predict(model_data['X_test'])
                
                predictions_data.extend([{'Model': name, 'Prediction': pred} for pred in preds[:100]])  # Sample for performance
            
            pred_df = pd.DataFrame(predictions_data)
            
            fig_pred_dist = px.box(
                pred_df,
                x='Model',
                y='Prediction',
                title='Prediction Distributions by Model (Log Scale)',
                color='Model'
            )
            st.plotly_chart(fig_pred_dist, use_container_width=True)
        
        with pred_col2:
            # Actual vs predicted for best model
            if 'XGBoost' in model_data['models']:
                model = model_data['models']['XGBoost']
                y_pred = model.predict(model_data['X_test'])
                
                # Sample for visualization
                sample_size = min(500, len(model_data['y_test']))
                indices = np.random.choice(len(model_data['y_test']), sample_size, replace=False)
                
                fig_scatter = px.scatter(
                    x=model_data['y_test'].iloc[indices],
                    y=y_pred[indices],
                    title='Actual vs Predicted (XGBoost Sample)',
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    opacity=0.6
                )
                
                # Add perfect prediction line
                min_val = min(model_data['y_test'].min(), y_pred.min())
                max_val = max(model_data['y_test'].max(), y_pred.max())
                fig_scatter.add_shape(
                    type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
def display_full_investment_analysis(predictions, purchase_price, crime_data, county):
    """Display comprehensive investment analysis"""
    
    # Calculate investment score
    market_context = {'is_urban': county in ['Los Angeles', 'San Francisco'], 'has_coordinates': True}
    investment_score, score_breakdown = calculate_investment_score(predictions, purchase_price, crime_data, market_context)
    
    st.header("Investment Analysis Results")
    
    # Investment recommendation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if investment_score >= 75:
            color, recommendation, icon = "#28a745", "STRONG BUY", "🟢"
        elif investment_score >= 60:
            color, recommendation, icon = "#ffc107", "BUY", "🟡"
        elif investment_score >= 45:
            color, recommendation, icon = "#fd7e14", "HOLD/CAUTION", "🟠"
        else:
            color, recommendation, icon = "#dc3545", "AVOID", "🔴"
        
        st.markdown(f"""
        <div class="recommendation-card" style="
            background: linear-gradient(135deg, {color}15, {color}25);
            border-color: {color};
        ">
            <h1 style="color: {color}; margin: 0;">{icon} {investment_score:.1f}/100</h1>
            <h3 style="color: {color}; margin: 0.5rem 0;">{recommendation}</h3>
            <p style="color: #666; margin: 0;">Investment Recommendation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Score breakdown
    st.subheader("Score Components Analysis")
    score_col1, score_col2 = st.columns(2)
    
    with score_col1:
        st.metric("Price Analysis", f"{score_breakdown['price_score']:.1f}/100", 
                 f"{score_breakdown['price_differential']:.1%} vs purchase price")
        st.metric("Safety Score", f"{score_breakdown['safety_score']:.1f}/100",
                 "Based on crime statistics")
    
    with score_col2:
        st.metric("Model Consensus", f"{score_breakdown['consensus_score']:.1f}/100",
                 f"±${score_breakdown['prediction_std']:,.0f} variation")
        st.metric("Market Context", f"{score_breakdown['context_score']:.1f}/100",
                 f"{county} market factors")
    
    # Detailed breakdown chart
    breakdown_data = pd.DataFrame({
        'Component': ['Price Analysis', 'Safety Score', 'Model Consensus', 'Market Context'],
        'Score': [score_breakdown['price_score'], score_breakdown['safety_score'], 
                 score_breakdown['consensus_score'], score_breakdown['context_score']],
        'Weight': [35, 25, 20, 20]
    })
    
    fig_breakdown = px.bar(
        breakdown_data,
        x='Component', y='Score',
        title='Investment Score Components',
        color='Score',
        color_continuous_scale='RdYlGn',
        text='Score'
    )
    fig_breakdown.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_breakdown.update_layout(showlegend=False)
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Predictions visualization
    display_prediction_comparison(predictions, purchase_price)
    
    # Investment insights
    st.subheader("Investment Insights")
    
    insights = []
    
    if score_breakdown['price_differential'] > 0.1:
        insights.append("Models predict property value significantly above purchase price")
    elif score_breakdown['price_differential'] < -0.1:
        insights.append("Models predict property value below purchase price")
    else:
        insights.append("Models predict property value close to purchase price")
    
    if crime_data['safety_score'] > 80:
        insights.append("Excellent safety rating for the area")
    elif crime_data['safety_score'] > 60:
        insights.append("Moderate safety rating for the area")
    else:
        insights.append("Lower safety rating - consider security measures")
    
    if score_breakdown['consensus_score'] > 80:
        insights.append("High model agreement increases confidence")
    else:
        insights.append("Some model disagreement - consider additional research")
    
    for insight in insights:
        st.write(insight)

def display_model_performance(metadata):
    """Display model performance comparison"""
    
    if 'model_performance' not in metadata:
        return
    
    st.subheader("Model Performance Summary")
    
    results = metadata['model_performance']
    perf_df = pd.DataFrame(results).T
    
    # Performance metrics table
    display_cols = ['Test_R2', 'Test_RMSE', 'CV_R2_Mean', 'Generalization_Gap']
    available_cols = [col for col in display_cols if col in perf_df.columns]
    
    if available_cols:
        st.dataframe(perf_df[available_cols].round(4), use_container_width=True)

def display_prediction_comparison(predictions, purchase_price):
    """Display model prediction comparison"""
    
    st.subheader("Model Predictions Comparison")
    
    pred_df = pd.DataFrame({
        'Model': list(predictions.keys()),
        'Predicted_Value': list(predictions.values()),
        'Difference': [v - purchase_price for v in predictions.values()],
        'Difference_Pct': [(v - purchase_price) / purchase_price * 100 for v in predictions.values()]
    })
    
    # Predictions bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Predicted Values',
        x=pred_df['Model'],
        y=pred_df['Predicted_Value'],
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(predictions)],
        text=[f"${v:,.0f}" for v in pred_df['Predicted_Value']],
        textposition='outside'
    ))
    
    fig.add_hline(
        y=purchase_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Purchase Price: ${purchase_price:,}"
    )
    
    fig.update_layout(
        title="Model Predictions vs Purchase Price",
        xaxis_title="Model",
        yaxis_title="Predicted Value ($)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictions table with formatted values
    display_df = pred_df.copy()
    display_df['Predicted_Value'] = display_df['Predicted_Value'].apply(lambda x: f"${x:,.0f}")
    display_df['Difference'] = display_df['Difference'].apply(lambda x: f"${x:,.0f}")
    display_df['Difference_Pct'] = display_df['Difference_Pct'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_df, use_container_width=True)

# ================================
# MAIN APPLICATION
# ================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">Portable Predictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Housing Investment Analysis with ML & Crime Data Integration</div>', unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner("Loading datasets and models..."):
        datasets = load_datasets()
        model_data = load_trained_models()
    
    if not datasets or not model_data:
        st.error("Required data not loaded. Please check data files and run the model trainer first.")
        
        # Provide helpful guidance
        st.markdown("### Setup Instructions")
        st.markdown("""
        **To use this application, you need:**
        
        1. **Trained Models**: Run one of these first:
           - `python csv_model_trainer.py` (for CSV-based training)
           - `python model_trainer.py` (for database-based training)
        
        2. **Data Files** (if using CSV mode):
           - `acs_housing_vw.csv` 
           - `crime_data.csv`
        
        3. **Database Access** (if using database mode):
           - Install: `∂ç`
           - Ensure database connection is working
        
        **Current Status:**
        """)
        
        # Check what's available
        if os.path.exists('saved_models'):
            st.info("Models directory found")
            models_found = [f for f in os.listdir('saved_models') if f.endswith('.pkl')]
            st.write(f"📁 Model files: {len(models_found)} found")
        else:
            st.warning("No models directory found - please run model trainer first")
        
        # Check for CSV files
        csv_files = ['acs_housing_vw.csv', 'crime_data.csv']
        csv_status = [f"{'✅' if os.path.exists(f) else '❌'} {f}" for f in csv_files]
        st.write("CSV Files:")
        for status in csv_status:
            st.write(f"   {status}")
        
        # Database status
        if DATABASE_AVAILABLE:
            st.write("Database libraries: Available")
        else:
            st.write("Database libraries: Not installed")
        
        return
    
    # Sidebar - Model info and performance
    with st.sidebar:
        st.header("System Dashboard")
        
        # Display data and model source
        data_source = datasets.get('data_source', 'Unknown')
        model_source = model_data.get('model_source', 'Unknown')
        
        st.markdown("### Data Pipeline Status")
        st.success(f"**Data Source**: {data_source}")
        if 'database_info' in model_data['metadata']:
            db_info = model_data['metadata']['database_info']
            st.success(f"**Models**: Database-trained")
            st.caption(f"DB: {db_info.get('database', 'Unknown')}")
        else:
            st.success(f"**Models**: {model_source}-trained")
        
        if model_data['metadata']:
            trained_at = model_data['metadata'].get('trained_at', 'Unknown')
            st.info(f"**Last Trained**: {trained_at}")
            
            # Quick performance summary
            if 'model_performance' in model_data['metadata']:
                results = model_data['metadata']['model_performance']
                best_model = max(results.keys(), key=lambda k: results[k].get('Test_R2', 0))
                best_r2 = results[best_model].get('Test_R2', 0)
                st.success(f"**Best Model**: {best_model}")
                st.metric("Best R² Score", f"{best_r2:.3f}")
        
        st.markdown("---")
        
        # System metrics
        st.markdown("### System Coverage")
        st.info(f"**Models**: {len(model_data['models'])} trained")
        st.info(f"**Counties**: {len(datasets['counties'])} covered")
        st.info(f"**Properties**: {len(datasets['housing']):,} records")
        st.info(f"**ZIP Codes**: {datasets['housing']['zip'].nunique():,} areas")
        
        st.markdown("---")
        
        # Quick data insights
        st.markdown("### Market Insights")
        avg_price = datasets['housing']['valp'].mean()
        median_price = datasets['housing']['valp'].median()
        st.metric("Average Price", f"${avg_price:,.0f}")
        st.metric("Median Price", f"${median_price:,.0f}")
        
        avg_safety = datasets['crime']['safety_score'].mean()
        st.metric("Average Safety Score", f"{avg_safety:.1f}/100")
        
        # Feature availability status
        st.markdown("---")
        st.markdown("### Features Available")
        
        # Check for optional features
        features_status = []
        
        # SHAP
        if SHAP_AVAILABLE:
            features_status.append("SHAP Interpretability")
        else:
            features_status.append("SHAP (install: pip install shap)")
        
        # Database
        if DATABASE_AVAILABLE:
            features_status.append("Database Connection")
        else:
            features_status.append("Database (install: pip install psycopg2-binary)")
        
        # Coordinates for mapping
        has_coords = 'latitude' in datasets['housing'].columns and 'longitude' in datasets['housing'].columns
        if has_coords:
            coord_coverage = datasets['housing'][['latitude', 'longitude']].dropna().shape[0]
            coverage_pct = (coord_coverage / len(datasets['housing'])) * 100
            features_status.append(f"Interactive Maps ({coverage_pct:.0f}% coverage)")
        else:
            features_status.append("Limited Mapping (no coordinates)")
        
        for status in features_status:
            st.caption(status)
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["Property Analysis", "Data Explorer", "Model Insights"])
    
    with tab1:
        display_property_analysis(datasets, model_data)
    
    with tab2:
        display_data_explorer(datasets)
    
    with tab3:
        display_model_insights(model_data)
    
    # Footer with system information
    st.markdown("---")
    
    # System information in expandable section
    with st.expander("System Information & Credits"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Technical Stack**")
            st.markdown("""
            - **ML**: scikit-learn, XGBoost
            - **Data**: pandas, numpy  
            - **Viz**: plotly, folium
            - **UI**: Streamlit
            - **DB**: PostgreSQL, SQLAlchemy
            """)
        
        with col2:
            st.markdown("**Model Features**")
            st.markdown(f"""
            - **Features**: {len(model_data['features'])} inputs
            - **Models**: {len(model_data['models'])} algorithms
            - **Data**: {datasets.get('data_source', 'Unknown')} source
            - **Counties**: {len(datasets['counties'])} areas
            - **Properties**: {len(datasets['housing']):,} records
            """)
        
        with col3:
            st.markdown("**Analysis Types**")
            st.markdown("""
            - **Investment Analysis**: Score & recommendation
            - **Model Comparison**: Performance metrics
            - **Interpretability**: SHAP analysis
            - **Diagnostics**: Model validation
            - **Market Explorer**: Data insights
            """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p><strong>Portable Predictions</strong> - Learning Housing Prices Across Diverse Markets</p>
            <p>Built by Joe Bryant, Mahek Patel, Nathan Deering</p>
            <p><em>Advanced ML-powered real estate investment analysis with integrated crime data</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()