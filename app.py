# app.py (updated with automatic data loading)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
from eda import EDA
from model import ReadmissionPredictor
import shap
import os
import joblib

# Set page configuration
st.set_page_config(
    page_title="Healthcare Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

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

# Check for existing data at startup
if not st.session_state.data_loaded:
    if load_existing_data():
        st.sidebar.success("Existing data loaded successfully!")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Data Upload", "EDA", "Model Training", "Prediction", "About"])

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

# Model Training Page
elif page == "Model Training":
    st.title("Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data or generate synthetic data first")
    else:
        # Check if model already exists
        if os.path.exists('models/best_model.pkl'):
            st.info("⚠️ A trained model already exists. Training new models will overwrite it.")
        
        if not st.session_state.model_trained:
            if st.button("Train Models"):
                with st.spinner("Training models... This may take a while"):
                    # Initialize predictor
                    predictor = ReadmissionPredictor(st.session_state.data_loader)
                    
                    # Prepare data
                    X, y = predictor.prepare_data()
                    
                    # Train models
                    predictor.train_models()
                    
                    # Evaluate best model
                    metrics = predictor.evaluate_model()
                    
                    # Store in session state
                    st.session_state.predictor = predictor
                    st.session_state.model_trained = True
                    
                    st.success("Model training completed!")
                    
                    # Display metrics
                    st.subheader("Model Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("AUC-ROC", f"{metrics['auc']:.4f}")
                    col2.metric("Brier Score", f"{metrics['brier']:.4f}")
                    col3.metric("Sensitivity", f"{metrics['sensitivity']:.4f}")
                    col4.metric("Specificity", f"{metrics['specificity']:.4f}")
        else:
            st.success("Model already trained!")
            
            # Display feature importance
            st.subheader("Feature Importance")
            if os.path.exists('outputs/plots/feature_importance.png'):
                st.image("outputs/plots/feature_importance.png")
            else:
                st.warning("Feature importance plot not found. Please train the model again.")
            
            # Display net benefit analysis
            st.subheader("Net Benefit Analysis")
            if os.path.exists('outputs/plots/net_benefit.png'):
                st.image("outputs/plots/net_benefit.png")
            else:
                st.warning("Net benefit plot not found. Please train the model again.")

# Prediction Page
elif page == "Prediction":
    st.title("Patient Risk Prediction")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first")
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
            # Convert to DataFrame
            patient_df = pd.DataFrame([input_data])
            
            # Get prediction
            explanation = st.session_state.predictor.predict_risk(patient_df)
            
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
    
    ### Key Features:
    
    1. **Data Integration**: Combines multiple data sources for comprehensive analysis
    2. **Advanced Analytics**: Includes EDA, feature engineering, and outlier detection
    3. **Machine Learning Models**: Implements multiple algorithms with ensemble methods
    4. **Clinical Decision Support**: Provides risk stratification and feature importance
    5. **Explainable AI**: Uses SHAP values for model interpretation
    
    ### Technical Implementation:
    
    - **Data Preprocessing**: Handles missing data, outliers, and feature engineering
    - **Imbalanced Data**: Uses SMOTE-Tomek links for balanced training
    - **Model Validation**: Group stratified k-fold cross-validation
    - **Performance Metrics**: AUC-ROC, Brier score, sensitivity, specificity
    - **Deployment**: Streamlit dashboard and Flask API
    
    ### How to Use:
    
    1. Upload your datasets on the Data Upload page (or generate synthetic data)
    2. Explore your data with the EDA tools
    3. Train the prediction models
    4. Use the Prediction page to assess patient risk
    
    ### Contact:
    
    For questions or support, please contact the development team.
    """)