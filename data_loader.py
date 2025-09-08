# data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re

class DataLoader:
    def __init__(self):
        self.data = None
        self.patient_ids = None
        self.target = 'readmitted_within_30days'
        
    def load_data(self, ehr_path, demo_path, lab_path, med_path, social_path, labels_path):
        """Load and merge all data sources"""
        # Load individual datasets
        ehr = pd.read_csv(ehr_path)
        demo = pd.read_csv(demo_path)
        lab = pd.read_csv(lab_path)
        med = pd.read_csv(med_path)
        social = pd.read_csv(social_path)
        labels = pd.read_csv(labels_path)
        
        # Merge datasets on patient_id
        self.data = ehr.merge(demo, on='patient_id', how='left')
        self.data = self.data.merge(lab, on='patient_id', how='left')
        self.data = self.data.merge(med, on='patient_id', how='left')
        self.data = self.data.merge(social, on='patient_id', how='left')
        self.data = self.data.merge(labels, on='patient_id', how='left')
        
        # Store patient IDs for group stratification
        self.patient_ids = self.data['patient_id'].values
        
        return self.data
    
    def preprocess_data(self):
        """Clean and preprocess the merged dataset"""
        # Convert target to binary
        self.data[self.target] = self.data[self.target].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Handle missing values
        self._handle_missing_data()
        
        # Process clinical notes (simplified NLP)
        self._process_clinical_notes()
        
        # Extract time-series features from lab data
        self._extract_lab_features()
        
        # Remove outliers
        self._remove_outliers()
        
        # Create derived features
        self._create_derived_features()
        
        return self.data
    
    def _handle_missing_data(self):
        """Handle missing data patterns"""
        # For numerical columns, impute with median
        num_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            if col != self.target:
                self.data[col].fillna(self.data[col].median(), inplace=True)
        
        # For categorical columns, impute with mode
        cat_cols = self.data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)
    
    def _process_clinical_notes(self):
        """Simple NLP processing for clinical notes"""
        if 'clinical_notes' in self.data.columns:
            # Extract keywords related to high risk conditions
            risk_keywords = ['sepsis', 'pneumonia', 'heart failure', 'diabetes', 'copd', 'infection']
            
            for keyword in risk_keywords:
                self.data[f'note_{keyword}'] = self.data['clinical_notes'].apply(
                    lambda x: 1 if re.search(keyword, x, re.IGNORECASE) else 0
                )
            
            # Drop original notes column
            self.data.drop('clinical_notes', axis=1, inplace=True)
    
    def _extract_lab_features(self):
        """Extract features from lab data"""
        lab_cols = [col for col in self.data.columns if col in ['glucose', 'creatinine', 'sodium', 'potassium', 'hemoglobin', 'white_blood_cell', 'platelet', 'albumin', 'bilirubin']]
        
        for col in lab_cols:
            # Create abnormal flags
            if col == 'glucose':
                self.data[f'{col}_abnormal'] = (self.data[col] > 180).astype(int)
            elif col == 'creatinine':
                self.data[f'{col}_abnormal'] = (self.data[col] > 1.5).astype(int)
            elif col == 'sodium':
                self.data[f'{col}_abnormal'] = ((self.data[col] < 135) | (self.data[col] > 145)).astype(int)
            elif col == 'potassium':
                self.data[f'{col}_abnormal'] = ((self.data[col] < 3.5) | (self.data[col] > 5.0)).astype(int)
            elif col == 'hemoglobin':
                self.data[f'{col}_abnormal'] = (self.data[col] < 12).astype(int)
            elif col == 'white_blood_cell':
                self.data[f'{col}_abnormal'] = ((self.data[col] < 4) | (self.data[col] > 11)).astype(int)
            elif col == 'platelet':
                self.data[f'{col}_abnormal'] = ((self.data[col] < 150) | (self.data[col] > 450)).astype(int)
            elif col == 'albumin':
                self.data[f'{col}_abnormal'] = (self.data[col] < 3.5).astype(int)
            elif col == 'bilirubin':
                self.data[f'{col}_abnormal'] = (self.data[col] > 1.2).astype(int)
    
    def _remove_outliers(self):
        """Remove outliers using modified Z-score"""
        num_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        num_cols = [col for col in num_cols if col != self.target and not col.endswith('_abnormal')]
        
        for col in num_cols:
            # Calculate modified Z-score
            median = np.median(self.data[col])
            mad = np.median(np.abs(self.data[col] - median))
            if mad == 0:  # Avoid division by zero
                continue
            modified_z_score = 0.6745 * (self.data[col] - median) / mad
            
            # Identify outliers (threshold = 3.5)
            outliers = np.abs(modified_z_score) > 3.5
            
            # Cap outliers instead of removing
            if outliers.sum() > 0:
                upper_bound = median + 3.5 * mad / 0.6745
                lower_bound = median - 3.5 * mad / 0.6745
                self.data[col] = np.where(self.data[col] > upper_bound, upper_bound, 
                                         np.where(self.data[col] < lower_bound, lower_bound, self.data[col]))
    

    def _create_derived_features(self):
        """Create derived features"""
        # Age groups
        self.data['age_group'] = pd.cut(self.data['age'], bins=[0, 40, 65, 80, 100], labels=['<40', '40-65', '65-80', '80+'])
        
        # Convert age_group to string to avoid categorical issues
        self.data['age_group'] = self.data['age_group'].astype(str)
        
        # Charlson Comorbidity Index (simplified)
        self.data['charlson_score'] = (
            (self.data['primary_diagnosis'] == 'Cancer').astype(int) * 2 +
            (self.data['primary_diagnosis'] == 'Heart Failure').astype(int) * 1 +
            (self.data['primary_diagnosis'] == 'COPD').astype(int) * 1 +
            (self.data['primary_diagnosis'] == 'Diabetes').astype(int) * 1 +
            (self.data['primary_diagnosis'] == 'Kidney Disease').astype(int) * 1 +
            (self.data['primary_diagnosis'] == 'Stroke').astype(int) * 1
        )
        
        # Frailty index (simplified)
        self.data['frailty_index'] = (
            (self.data['age'] > 80).astype(int) * 0.3 +
            (self.data['hemoglobin'] < 12).astype(int) * 0.2 +
            (self.data['albumin'] < 3.5).astype(int) * 0.2 +
            (self.data['social_support'] == 'None').astype(int) * 0.15 +
            (self.data['housing_status'] != 'Stable').astype(int) * 0.15
        )
        
        # Medication complexity score
        self.data['medication_complexity'] = (
            self.data['number_of_medications'] / 10 +
            self.data['high_risk_medications'] * 0.5 +
            (1 - self.data['medication_adherence']) * 0.5
        )
        
        # Convert all categorical columns to string type to avoid issues
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            self.data[col] = self.data[col].astype(str)
    
    def create_preprocessor(self, numerical_features, categorical_features):
        """Create preprocessing pipeline"""
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
   
    # data_loader.py (revised get_stratified_split method)

    def get_stratified_split(self, n_splits=5):
        """Create group stratified k-fold splits"""
        from sklearn.model_selection import GroupKFold
        import numpy as np
        
        # Ensure data index is reset
        self.data = self.data.reset_index(drop=True)
        
        # Create a simple GroupKFold
        group_kfold = GroupKFold(n_splits=n_splits)
        
        # Generate splits
        for train_idx, test_idx in group_kfold.split(
            self.data, 
            self.data[self.target], 
            groups=self.patient_ids
        ):
            # Return the indices directly
            yield train_idx, test_idx