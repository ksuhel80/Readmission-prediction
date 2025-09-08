# app.py (updated to use API for predictions)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
from eda import EDA
import requests
import os
import json

# Set page configuration
st.set_page_config(
    page_title="Healthcare Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'api_available' not in st.session_state:
    st.session_state.api_available = False

# API configuration
API_URL = "http://localhost:5000"

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

def check_api_available():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('model_loaded', False)
    except:
        pass
    return False

def check_data_exists():
    """Check if all required data files exist"""
    files = [
        'data/ehr_data.csv',
        'data/demographics.csv',
        'data/lab_results.csv',
        'data/medication_history.csv',
        'data/social_determinants.csv',
        'data/readmission_labels.csv'
    ]
    return all(os.path.exists(f) for f in files)

def load_existing_data():
    """Load existing data files if they exist"""
    if check_data_exists():
        try:
            data_loader = DataLoader()
            data = data_loader.load_data(
                ehr_path='data/ehr_data.csv',
                demo_path='data/demographics.csv',
                lab_path='data/lab_results.csv',
                med_path='data/medication_history.csv',
                social_path='data/social_determinants.csv',
                labels_path='data/readmission_labels.csv'
            )
            
            # Preprocess data
            data = data_loader.preprocess_data()
            
            # Store in session state
            st.session_state.data = data
            st.session_state.data_loader = data_loader
            st.session_state.data_loaded = True
            
            return True
        except Exception as e:
            st.error(f"Error loading existing data: {str(e)}")
            return False
    return False

# Check for existing data and API at startup
if not st.session_state.data_loaded:
    if load_existing_data():
        st.sidebar.success("Existing data loaded successfully!")

# Check API availability
st.session_state.api_available = check_api_available()
if st.session_state.api_available:
    st.sidebar.success("API is available")
else:
    st.sidebar.warning("API is not available. Please start the API server.")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Data Upload", "EDA", "Prediction", "About"])

# Data Upload Page
if page == "Data Upload":
    st.title("Data Upload")
    
    # Show existing data status
    if st.session_state.data_loaded:
        st.success("✅ Data files already exist and are loaded")
        st.write("You can:")
        st.write("1. Generate new synthetic data (will overwrite existing files)")
        st.write("2. Upload new data files (will overwrite existing files)")
        st.write("3. Use the existing data for analysis")
    else:
        st.warning("⚠️ No data files found")
        st.write("Please either:")
        st.write("1. Generate synthetic data")
        st.write("2. Upload your own data files")
    
    st.write("---")
    
    # Option to generate synthetic data
    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating synthetic data..."):
            # Import the generator here to avoid issues
            from data_generator import HealthcareDataGenerator
            
            # Generate synthetic data
            generator = HealthcareDataGenerator(num_patients=1000)
            generator.generate_all_data()
            
            # Check if files were created
            if check_data_exists():
                st.success("Synthetic data generated successfully!")
                
                # Load the generated data
                if load_existing_data():
                    st.dataframe(st.session_state.data.head())
            else:
                st.error("Failed to generate synthetic data. Please check the error messages.")
    
    st.write("---")
    
    # File uploaders
    st.subheader("Upload Your Own Data")
    ehr_file = st.file_uploader("Upload EHR Data", type=["csv"])
    demo_file = st.file_uploader("Upload Demographics Data", type=["csv"])
    lab_file = st.file_uploader("Upload Lab Results Data", type=["csv"])
    med_file = st.file_uploader("Upload Medication History Data", type=["csv"])
    social_file = st.file_uploader("Upload Social Determinants Data", type=["csv"])
    labels_file = st.file_uploader("Upload Readmission Labels Data", type=["csv"])
    
    if st.button("Load and Preprocess Data"):
        if all([ehr_file, demo_file, lab_file, med_file, social_file, labels_file]):
            with st.spinner("Loading and preprocessing data..."):
                # Save uploaded files
                with open('data/ehr_data.csv', 'wb') as f:
                    f.write(ehr_file.getvalue())
                with open('data/demographics.csv', 'wb') as f:
                    f.write(demo_file.getvalue())
                with open('data/lab_results.csv', 'wb') as f:
                    f.write(lab_file.getvalue())
                with open('data/medication_history.csv', 'wb') as f:
                    f.write(med_file.getvalue())
                with open('data/social_determinants.csv', 'wb') as f:
                    f.write(social_file.getvalue())
                with open('data/readmission_labels.csv', 'wb') as f:
                    f.write(labels_file.getvalue())
                
                # Load data
                if load_existing_data():
                    st.success("Data loaded and preprocessed successfully!")
                    st.dataframe(st.session_state.data.head())
        else:
            st.error("Please upload all required datasets")
    
    # Show existing data preview
    if st.session_state.data_loaded:
        st.write("---")
        st.subheader("Current Data Preview")
        st.dataframe(st.session_state.data.head())

# EDA Page
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data or generate synthetic data first")
    else:
        eda = EDA(st.session_state.data)
        
        # Select analysis type
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Patient Cohort Analysis", "Feature Correlation", "Missing Data Patterns", "Categorical Features"]
        )
        
        if analysis_type == "Patient Cohort Analysis":
            eda.patient_cohort_analysis()
            st.pyplot(plt)
        elif analysis_type == "Feature Correlation":
            eda.feature_correlation()
            st.pyplot(plt)
        elif analysis_type == "Missing Data Patterns":
            eda.missing_data_patterns()
            st.pyplot(plt)
        elif analysis_type == "Categorical Features":
            eda.categorical_feature_analysis()
            st.pyplot(plt)

# Prediction Page
elif page == "Prediction":
    st.title("Patient Risk Prediction")
    
    if not st.session_state.api_available:
        st.error("API is not available. Please start the API server.")
        st.write("To start the API server, run the following command in a separate terminal:")
        st.code("python api.py")
    elif not st.session_state.data_loaded:
        st.warning("Please upload data or generate synthetic data first")
    else:
        # Input form
        st.subheader("Enter Patient Information")
        
        # Create input form based on data columns
        input_data = {}
        for col in st.session_state.data.columns:
            if col == 'patient_id' or col == st.session_state.data_loader.target:
                continue
                
            if st.session_state.data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(st.session_state.data[col]):
                # For categorical columns, use selectbox
                input_data[col] = st.selectbox(col, st.session_state.data[col].unique())
            else:
                # For numerical columns, use number_input with median as default
                median_val = st.session_state.data[col].median()
                input_data[col] = st.number_input(col, value=float(median_val))
        
        if st.button("Predict Risk"):
            with st.spinner("Getting prediction from API..."):
                # Make API request
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=input_data,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        explanation = response.json()
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Risk Probability", f"{explanation['risk_probability']:.2%}")
                        col2.metric("Risk Level", explanation['risk_level'])
                        
                        # Display feature contributions
                        st.subheader("Feature Contributions")
                        contributions = pd.DataFrame.from_dict(
                            explanation['feature_contributions'], 
                            orient='index', 
                            columns=['SHAP Value']
                        ).sort_values('SHAP Value', ascending=False)
                        
                        st.dataframe(contributions)
                        
                        # Plot SHAP values
                        fig, ax = plt.subplots()
                        contributions.head(10).plot(kind='barh', ax=ax)
                        ax.invert_yaxis()
                        plt.title("Top 10 Feature Contributions")
                        st.pyplot(fig)
                    else:
                        st.error(f"API returned status code {response.status_code}")
                        st.json(response.json())
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to API: {str(e)}")
        
       

# About Page
elif page == "About":
    st.title("About This Application")
    st.markdown("""
    ### Healthcare Readmission Prediction System
    
    This application predicts the risk of patient readmission within 30 days of discharge using:
    
    - **Electronic Health Records (EHR)**
    - **Patient Demographics**
    - **Lab Results**
    - **Medication History**
    - **Social Determinants of Health**
    
    ### Architecture
    
    This system consists of two main components:
    
    1. **Streamlit Dashboard**: User interface for data exploration and predictions
    2. **Flask API**: Backend service for model predictions
    
    ### Key Features:
    
    1. **Data Integration**: Combines multiple data sources for comprehensive analysis
    2. **Advanced Analytics**: Includes EDA, feature engineering, and outlier detection
    3. **Machine Learning Models**: Implements multiple algorithms with ensemble methods
    4. **Clinical Decision Support**: Provides risk stratification and feature importance
    5. **Explainable AI**: Uses SHAP values for model interpretation
    6. **API Integration**: Uses a REST API for predictions
    
    ### How to Use:
    
    1. **Start the API Server**: In a separate terminal, run:
       ```
       python api.py
       ```
    
    2. **Start the Dashboard**: In another terminal, run:
       ```
       streamlit run app.py
       ```
    
    3. **Upload Data**: On the Data Upload page, either generate synthetic data or upload your own
    
    4. **Explore Data**: Use the EDA tools to understand your data
    
    5. **Make Predictions**: Use the Prediction page to assess patient risk
    
    ### API Endpoints:
    
    - `GET /health`: Health check endpoint
    - `POST /predict`: Predict risk for a single patient
    - `POST /batch_predict`: Predict risk for multiple patients
    - `GET /model_info`: Get model information
    
    ### Contact:
    
    For questions or support, please contact the development team.
    """)