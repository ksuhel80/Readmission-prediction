# check_saved_model.py

import joblib
import os

def check_saved_model():
    predictor_path = 'models/predictor_instance.pkl'
    data_loader_path = 'models/data_loader.pkl'
    
    print(f"Checking predictor at: {os.path.abspath(predictor_path)}")
    print(f"Checking data_loader at: {os.path.abspath(data_loader_path)}")
    
    if os.path.exists(predictor_path):
        predictor = joblib.load(predictor_path)
        print(f"\nPredictor type: {type(predictor)}")
        print(f"Has predict_risk method: {hasattr(predictor, 'predict_risk')}")
        print(f"Has best_model attribute: {hasattr(predictor, 'best_model')}")
        
        if hasattr(predictor, 'best_model'):
            print(f"Best model type: {type(predictor.best_model)}")
    else:
        print("\nPredictor file not found")
    
    if os.path.exists(data_loader_path):
        data_loader = joblib.load(data_loader_path)
        print(f"\nDataLoader type: {type(data_loader)}")
    else:
        print("\nDataLoader file not found")

if __name__ == "__main__":
    check_saved_model()