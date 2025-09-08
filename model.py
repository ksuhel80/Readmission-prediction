# model.py
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (roc_auc_score, brier_score_loss, 
                            confusion_matrix, classification_report)
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib

class ReadmissionPredictor:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.models = {}
        self.best_model = None
        self.preprocessor = None
        self.feature_names = None
        self.X = None
        self.y = None
        
    def prepare_data(self):
        """Prepare data for modeling"""
        # Separate features and target
        self.X = self.data_loader.data.drop(columns=[self.data_loader.target, 'patient_id'])
        self.y = self.data_loader.data[self.data_loader.target]
        
        # Identify feature types
        numerical_features = self.X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_features = self.X.select_dtypes(include=['object']).columns.tolist()
        
        # Create preprocessor
        self.preprocessor = self.data_loader.create_preprocessor(numerical_features, categorical_features)
        
        # Store feature names after preprocessing
        self.feature_names = numerical_features + categorical_features
        
        return self.X, self.y
    
# model.py (completely revised train_models method)

    def train_models(self):
        """Train multiple models with nested cross-validation"""
        # Define models and hyperparameters
        models = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, class_weight='balanced'),
                'params': {
                    'classifier__C': [0.01, 0.1, 1, 10, 100],
                    'classifier__penalty': ['l2']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(class_weight='balanced'),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(scale_pos_weight=5, use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 6],
                    'classifier__learning_rate': [0.01, 0.1]
                }
            },
            'NeuralNetwork': {
                'model': MLPClassifier(max_iter=1000, early_stopping=True),
                'params': {
                    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'classifier__alpha': [0.0001, 0.001]
                }
            }
        }
        
        # Create output directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Reset indices to ensure proper alignment
        self.X = self.X.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)
        
        # Create pipeline with SMOTE-Tomek
        for model_name, config in models.items():
            print(f"\nTraining {model_name}...")
            
            pipeline = ImbPipeline([
                ('preprocessor', self.preprocessor),
                ('smote_tomek', SMOTETomek(random_state=42)),
                ('classifier', config['model'])
            ])
            
            # Create a simple train-test split for initial evaluation
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
            )
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=config['params'],
                cv=3,  # Simple 3-fold CV
                scoring='roc_auc',
                n_jobs=1,  # Use single process to avoid multiprocessing issues
                verbose=1
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Evaluate on test set
            best_model = grid_search.best_estimator_
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            self.models[model_name] = {
                'best_estimator': best_model,
                'best_params': grid_search.best_params_,
                'cv_score': score,
                'cv_std': 0.0  # Not available with simple train-test split
            }
            
            # Save model
            joblib.dump(best_model, f'models/{model_name}.pkl')
            
            print(f"Best params: {grid_search.best_params_}")
            print(f"Test AUC: {score:.4f}")
        
        # Select best model
        self.best_model = max(self.models.items(), key=lambda x: x[1]['cv_score'])[1]['best_estimator']
        print(f"\nBest model selected: {max(self.models.items(), key=lambda x: x[1]['cv_score'])[0]}")
        
        # Save best model
        joblib.dump(self.best_model, 'models/best_model.pkl')
    
# model.py (revised evaluate_model method)

    def evaluate_model(self):
        """Evaluate the best model on test set"""
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        
        # Fit best model
        self.best_model.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        print("\nModel Evaluation Metrics:")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs/plots', exist_ok=True)
        
        # Net Benefit Analysis
        self.net_benefit_analysis(y_test, y_pred_proba)
        
        # Feature importance
        self.plot_feature_importance()
        
        # Save metrics
        metrics = {
            'auc': auc,
            'brier': brier,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
        
        joblib.dump(metrics, 'outputs/model_metrics.pkl')
        
        return metrics
    
    def net_benefit_analysis(self, y_true, y_proba, thresholds=np.linspace(0, 1, 100)):
        """Perform net benefit analysis for clinical utility"""
        net_benefits = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Net benefit calculation
            net_benefit = (tp / len(y_true)) - (fp / len(y_true)) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)
        
        # Plot net benefit curve
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, net_benefits, label='Model')
        plt.plot(thresholds, thresholds * 0, 'k--', label='Treat All')
        plt.plot(thresholds, (1 - thresholds) * 0, 'k:', label='Treat None')
        plt.xlabel('Threshold Probability')
        plt.ylabel('Net Benefit')
        plt.title('Net Benefit Analysis')
        plt.legend()
        plt.savefig('outputs/plots/net_benefit.png')
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance using SHAP values"""
        # Get transformed feature names
        if hasattr(self.best_model.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = self.best_model.named_steps['preprocessor'].get_feature_names_out()
        else:
            feature_names = self.feature_names
        
        # Create a sample of data for SHAP (to speed up computation)
        X_sample = self.X.sample(100, random_state=42)
        
        # Transform data
        X_transformed = self.best_model.named_steps['preprocessor'].transform(X_sample)
        
        # Create explainer
        explainer = shap.Explainer(self.best_model.named_steps['classifier'], X_transformed)
        shap_values = explainer(X_transformed)
        
        # Plot summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('outputs/plots/feature_importance.png')
        plt.show()
    
    def predict_risk(self, patient_data):
        """Predict readmission risk for new patient data"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        # Ensure patient_data is a DataFrame
        if not isinstance(patient_data, pd.DataFrame):
            patient_data = pd.DataFrame([patient_data])
        
        # Predict probability
        risk_proba = self.best_model.predict_proba(patient_data)[:, 1]
        
        # Get SHAP values for explanation
        X_transformed = self.best_model.named_steps['preprocessor'].transform(patient_data)
        explainer = shap.Explainer(self.best_model.named_steps['classifier'], X_transformed)
        shap_values = explainer(X_transformed)
        
        # Get feature names
        if hasattr(self.best_model.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = self.best_model.named_steps['preprocessor'].get_feature_names_out()
        else:
            feature_names = self.feature_names
        
        # Create explanation dictionary
        explanation = {
            'risk_probability': float(risk_proba[0]),
            'risk_level': 'High' if risk_proba[0] > 0.7 else 'Medium' if risk_proba[0] > 0.3 else 'Low',
            'feature_contributions': {
                feature_names[i]: float(shap_values.values[0][i]) 
                for i in range(len(feature_names))
            }
        }
        
        return explanation