# main.py
from data_generator import HealthcareDataGenerator
from data_loader import DataLoader
from eda import EDA
from model import ReadmissionPredictor
import os
import joblib

def main():
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    generator = HealthcareDataGenerator(num_patients=2000)
    generator.generate_all_data()
    
    # Initialize data loader
    print("\nLoading data...")
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
    print("\nPreprocessing data...")
    data = data_loader.preprocess_data()
    
    # Perform EDA
    print("\nRunning EDA...")
    eda = EDA(data)
    eda.run_full_eda()
    
    # Initialize predictor
    print("\nInitializing predictor...")
    predictor = ReadmissionPredictor(data_loader)
    
    # Prepare data
    print("\nPreparing data for modeling...")
    X, y = predictor.prepare_data()
    
    # Train models
    print("\nTraining models...")
    predictor.train_models()
    
    joblib.dump(predictor, 'models/predictor_instance.pkl')
    
    # Evaluate best model
    print("\nEvaluating best model...")
    metrics = predictor.evaluate_model()
    
    print("\nModel training completed successfully!")
    print(f"Best model AUC-ROC: {metrics['auc']:.4f}")
    print(f"Best model saved to models/best_model.pkl")
    
    # Save data loader for dashboard
    joblib.dump(data_loader, 'models/data_loader.pkl')

if __name__ == "__main__":
    main()