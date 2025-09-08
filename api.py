# api.py
from flask import Flask, request, jsonify
from data_loader import DataLoader
from model import ReadmissionPredictor
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load pre-trained model (in production, this would be loaded at startup)
model_path = 'models/best_model.pkl'
if os.path.exists(model_path):
    predictor = joblib.load(model_path)
else:
    predictor = None

@app.route('/predict', methods=['POST'])
def predict():
    """Predict readmission risk for a single patient"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get JSON data
    data = request.get_json()
    
    # Convert to DataFrame
    patient_df = pd.DataFrame([data])
    
    # Get prediction
    try:
        explanation = predictor.predict_risk(patient_df)
        return jsonify(explanation)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict readmission risk for multiple patients"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get JSON data (list of patients)
    data = request.get_json()
    
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a list of patient records'}), 400
    
    # Convert to DataFrame
    patient_df = pd.DataFrame(data)
    
    # Get predictions
    try:
        results = []
        for i in range(len(patient_df)):
            explanation = predictor.predict_risk(patient_df.iloc[[i]])
            results.append(explanation)
        
        return jsonify({'predictions': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    """Return information about the model"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get model performance metrics (in production, these would be stored)
    if os.path.exists('outputs/model_metrics.pkl'):
        metrics = joblib.load('outputs/model_metrics.pkl')
    else:
        metrics = {
            'model_type': 'XGBoost',  # Example
            'version': '1.0',
            'auc_roc': 0.85,
            'brier_score': 0.15,
            'sensitivity': 0.78,
            'specificity': 0.82
        }
    
    return jsonify(metrics)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': predictor is not None})

if __name__ == '__main__':
    # For development only - use a proper WSGI server in production
    app.run(debug=True, host='0.0.0.0', port=5000)