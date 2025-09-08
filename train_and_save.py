# train_predictor.py (fixed to properly save ReadmissionPredictor)

import os
import pandas as pd
import numpy as np
from data_generator import HealthcareDataGenerator
from data_loader import DataLoader
from model import ReadmissionPredictor
import joblib

def main():
    print("=== Training and Saving ReadmissionPredictor ===")
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic data...")
    generator = HealthcareDataGenerator(num_patients=1000)
    generator.generate_all_data()
    
    # Step 2: Load data
    print("\n2. Loading data...")
    data_loader = DataLoader()
    data = data_loader.load_data(
        ehr_path='data/ehr_data.csv',
        demo_path='data/demographics.csv',
        lab_path='data/lab_results.csv',
        med_path='data/medication_history.csv',
        social_path='data/social_determinants.csv',
        labels_path='data/readmission_labels.csv'
    )
    
    # Step 3: Preprocess data
    print("\n3. Preprocessing data...")
    data = data_loader.preprocess_data()
    
    # Step 4: Initialize and train predictor
    print("\n4. Training predictor...")
    predictor = ReadmissionPredictor(data_loader)
    X, y = predictor.prepare_data()
    predictor.train_models()
    
    # Step 5: Evaluate model
    print("\n5. Evaluating model...")
    metrics = predictor.evaluate_model()
    print(f"   Model AUC: {metrics['auc']:.4f}")
    
    # Step 6: Save predictor and data loader
    print("\n6. Saving predictor and data loader...")
    predictor_path = 'models/predictor_instance.pkl'
    data_loader_path = 'models/data_loader.pkl'
    
    # Save the predictor instance (ReadmissionPredictor object)
    joblib.dump(predictor, predictor_path)
    # Save the data loader
    joblib.dump(data_loader, data_loader_path)
    
    print(f"   Predictor saved to: {predictor_path}")
    print(f"   DataLoader saved to: {data_loader_path}")
    
    # Verify files exist
    if os.path.exists(predictor_path) and os.path.exists(data_loader_path):
        print("   ✅ All files verified")
        
        # Verify the saved object type
        loaded_predictor = joblib.load(predictor_path)
        print(f"   Type of saved predictor: {type(loaded_predictor)}")
        print(f"   Has predict_risk method: {hasattr(loaded_predictor, 'predict_risk')}")
    else:
        print("   ❌ Some files not found after saving")
    
    print("\n=== Training Complete ===")
    print("You can now start the API server with: python api.py")

if __name__ == "__main__":
    main()