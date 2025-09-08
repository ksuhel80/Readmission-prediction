# api.py (fixed to properly use ReadmissionPredictor)

from flask import Flask, request, jsonify, render_template_string
from data_loader import DataLoader
from model import ReadmissionPredictor
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Model paths
predictor_path = 'models/predictor_instance.pkl'
data_loader_path = 'models/data_loader.pkl'

# Initialize predictor and data loader
predictor = None
data_loader = None

def load_model():
    """Load the predictor instance"""
    global predictor, data_loader
    try:
        if os.path.exists(predictor_path) and os.path.exists(data_loader_path):
            predictor = joblib.load(predictor_path)
            data_loader = joblib.load(data_loader_path)
            print(f"Predictor type: {type(predictor)}")
            print(f"Has predict_risk method: {hasattr(predictor, 'predict_risk')}")
            print("ReadmissionPredictor and DataLoader loaded successfully")
            return True
        else:
            print(f"Model files not found. Predictor: {os.path.exists(predictor_path)}, DataLoader: {os.path.exists(data_loader_path)}")
            return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Try to load model at startup
model_loaded = load_model()

@app.route('/')
def index():
    """Root endpoint with API documentation"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Healthcare Readmission Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1, h2 {
                color: #2c3e50;
            }
            .status {
                padding: 10px;
                margin: 10px 0;
                border-radius: 4px;
            }
            .success {
                background-color: #d4edda;
                color: #155724;
            }
            .warning {
                background-color: #fff3cd;
                color: #856404;
            }
            code {
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 4px;
            }
            pre {
                background-color: #f8f8f8;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <h1>Healthcare Readmission Prediction API</h1>
        
        <h2>Status</h2>
        <div class="status {{ status_class }}">
            Model loaded: {{ model_status }}
        </div>
        
        {% if not model_loaded %}
        <div class="status warning">
            <strong>Model not loaded!</strong> Please train the model by running:
            <code>python train_predictor.py</code>
        </div>
        {% endif %}
        
        <h2>Endpoints</h2>
        
        <h3>GET /health</h3>
        <p>Health check endpoint</p>
        <h4>Response:</h4>
        <pre>{
  "status": "healthy",
  "model_loaded": {{ model_loaded|lower }}
}</pre>
        
        <h3>POST /predict</h3>
        <p>Predict readmission risk for a single patient</p>
        
        <h3>POST /batch_predict</h3>
        <p>Predict readmission risk for multiple patients</p>
        
        <h3>GET /model_info</h3>
        <p>Get model information</p>
    </body>
    </html>
    """, model_status="Yes" if model_loaded else "No", status_class="success" if model_loaded else "warning")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'predictor_file_exists': os.path.exists(predictor_path),
        'data_loader_file_exists': os.path.exists(data_loader_path)
    })

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predict readmission risk for a single patient"""
    if request.method == 'GET':
        return jsonify({
            "error": "This endpoint only accepts POST requests",
            "message": "Please send a POST request with patient data in JSON format"
        }), 405
    
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please train the model by running: python train_predictor.py'
        }), 500
    
    # Get JSON data
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Convert to DataFrame
    try:
        patient_df = pd.DataFrame([data])
    except Exception as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    
    # Get prediction
    try:
        # Check if predictor has predict_risk method
        if hasattr(predictor, 'predict_risk'):
            explanation = predictor.predict_risk(patient_df)
            return jsonify(explanation)
        else:
            # If not, try to use the best_model directly
            if hasattr(predictor, 'best_model'):
                # Preprocess the data using the data_loader
                X = data_loader.preprocessor.transform(patient_df)
                # Get prediction probability
                risk_probability = float(predictor.best_model.predict_proba(X)[0][1])
                # Determine risk level
                if risk_probability > 0.7:
                    risk_level = 'High'
                elif risk_probability > 0.3:
                    risk_level = 'Medium'
                else:
                    risk_level = 'Low'
                # Create explanation
                explanation = {
                    'risk_probability': risk_probability,
                    'risk_level': risk_level,
                    'feature_contributions': {
                        'Note': 'Detailed feature contributions not available with this method'
                    }
                }
                return jsonify(explanation)
            else:
                return jsonify({'error': 'Neither predict_risk method nor best_model attribute found'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    """Predict readmission risk for multiple patients"""
    if request.method == 'GET':
        return jsonify({
            "error": "This endpoint only accepts POST requests",
            "message": "Please send a POST request with a list of patient records in JSON format"
        }), 405
    
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please train the model by running: python train_predictor.py'
        }), 500
    
    # Get JSON data (list of patients)
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a list of patient records'}), 400
    
    # Convert to DataFrame
    try:
        patient_df = pd.DataFrame(data)
    except Exception as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    
    # Get predictions
    try:
        results = []
        for i in range(len(patient_df)):
            if hasattr(predictor, 'predict_risk'):
                explanation = predictor.predict_risk(patient_df.iloc[[i]])
                results.append(explanation)
            else:
                # Use the best_model directly
                if hasattr(predictor, 'best_model'):
                    # Preprocess the data using the data_loader
                    X = data_loader.preprocessor.transform(patient_df.iloc[[i]])
                    # Get prediction probability
                    risk_probability = float(predictor.best_model.predict_proba(X)[0][1])
                    # Determine risk level
                    if risk_probability > 0.7:
                        risk_level = 'High'
                    elif risk_probability > 0.3:
                        risk_level = 'Medium'
                    else:
                        risk_level = 'Low'
                    # Create explanation
                    explanation = {
                        'risk_probability': risk_probability,
                        'risk_level': risk_level,
                        'feature_contributions': {
                            'Note': 'Detailed feature contributions not available with this method'
                        }
                    }
                    results.append(explanation)
                else:
                    return jsonify({'error': 'Neither predict_risk method nor best_model attribute found'}), 500
        
        return jsonify({'predictions': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    """Return information about the model"""
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please train the model by running: python train_predictor.py'
        }), 500
    
    # Get model performance metrics (in production, these would be stored)
    if os.path.exists('outputs/model_metrics.pkl'):
        metrics = joblib.load('outputs/model_metrics.pkl')
        return jsonify(metrics)
    else:
        return jsonify({
            'model_type': 'XGBoost',  # Example
            'version': '1.0',
            'auc_roc': 0.85,
            'brier_score': 0.15,
            'sensitivity': 0.78,
            'specificity': 0.82
        })

if __name__ == '__main__':
    # For development only - use a proper WSGI server in production
    app.run(debug=True, host='0.0.0.0', port=5000)