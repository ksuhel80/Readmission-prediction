# data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class HealthcareDataGenerator:
    def __init__(self, num_patients=1000):
        self.num_patients = num_patients
        np.random.seed(42)
        random.seed(42)
        
    def generate_demographics(self):
        """Generate patient demographics data"""
        data = {
            'patient_id': range(1, self.num_patients + 1),
            'age': np.random.normal(65, 15, self.num_patients).clip(18, 100).astype(int),
            'gender': np.random.choice(['Male', 'Female'], self.num_patients),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], self.num_patients, p=[0.6, 0.15, 0.12, 0.08, 0.05]),
            'insurance_type': np.random.choice(['Medicare', 'Medicaid', 'Private', 'Uninsured'], self.num_patients, p=[0.45, 0.2, 0.3, 0.05]),
            'marital_status': np.random.choice(['Married', 'Single', 'Divorced', 'Widowed'], self.num_patients, p=[0.5, 0.2, 0.15, 0.15])
        }
        return pd.DataFrame(data)
    
    def generate_ehr(self):
        """Generate electronic health records data"""
        data = {
            'patient_id': range(1, self.num_patients + 1),
            'admission_date': [datetime(2022, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(self.num_patients)],
            'discharge_date': [datetime(2022, 1, 1) + timedelta(days=np.random.randint(3, 30)) for _ in range(self.num_patients)],
            'length_of_stay': np.random.randint(1, 30, self.num_patients),
            'primary_diagnosis': np.random.choice([
                'Heart Failure', 'Pneumonia', 'COPD', 'Diabetes', 'Sepsis', 
                'Kidney Disease', 'Cancer', 'Stroke', 'Hip Fracture', 'Other'
            ], self.num_patients),
            'number_of_diagnoses': np.random.randint(1, 15, self.num_patients),
            'number_of_procedures': np.random.randint(0, 10, self.num_patients),
            'emergency_admission': np.random.choice([0, 1], self.num_patients, p=[0.7, 0.3]),
            'previous_admissions': np.random.poisson(1.5, self.num_patients).clip(0, 10),
            'clinical_notes': self._generate_clinical_notes()
        }
        return pd.DataFrame(data)
    
    def _generate_clinical_notes(self):
        """Generate synthetic clinical notes"""
        notes = []
        conditions = ['diabetes', 'hypertension', 'heart failure', 'copd', 'kidney disease', 
                     'pneumonia', 'sepsis', 'anemia', 'depression', 'dementia']
        
        for _ in range(self.num_patients):
            num_conditions = np.random.randint(1, 4)
            selected_conditions = np.random.choice(conditions, num_conditions, replace=False)
            
            note = f"Patient admitted with {selected_conditions[0]}. "
            if len(selected_conditions) > 1:
                note += f"History of {', '.join(selected_conditions[1:])}. "
            
            note += np.random.choice([
                "Patient stable for discharge. Follow up in 2 weeks.",
                "Condition improved. Continue medications as prescribed.",
                "Patient responding well to treatment. Discharge home.",
                "Significant improvement observed. Recommend outpatient follow-up."
            ])
            
            notes.append(note)
        
        return notes
    
    def generate_lab_results(self):
        """Generate lab results data"""
        data = {
            'patient_id': range(1, self.num_patients + 1),
            'glucose': np.random.normal(120, 40, self.num_patients).clip(70, 300),
            'creatinine': np.random.normal(1.2, 0.5, self.num_patients).clip(0.5, 5.0),
            'sodium': np.random.normal(140, 3, self.num_patients).clip(130, 150),
            'potassium': np.random.normal(4.2, 0.5, self.num_patients).clip(3.0, 6.0),
            'hemoglobin': np.random.normal(12.5, 2.0, self.num_patients).clip(7.0, 17.0),
            'white_blood_cell': np.random.normal(8.0, 3.0, self.num_patients).clip(2.0, 20.0),
            'platelet': np.random.normal(250, 75, self.num_patients).clip(50, 600),
            'albumin': np.random.normal(3.8, 0.5, self.num_patients).clip(2.0, 5.0),
            'bilirubin': np.random.normal(0.8, 0.4, self.num_patients).clip(0.2, 3.0)
        }
        return pd.DataFrame(data)
    
    def generate_medication_history(self):
        """Generate medication history data"""
        medications = ['Metformin', 'Lisinopril', 'Atorvastatin', 'Furosemide', 
                      'Omeprazole', 'Aspirin', 'Metoprolol', 'Insulin', 
                      'Levothyroxine', 'Albuterol']
        
        data = {
            'patient_id': range(1, self.num_patients + 1),
            'number_of_medications': np.random.randint(0, 15, self.num_patients),
            'high_risk_medications': np.random.choice([0, 1], self.num_patients, p=[0.7, 0.3]),
            'medication_adherence': np.random.beta(8, 2, self.num_patients),  # Mostly high adherence
            'antibiotics': np.random.choice([0, 1], self.num_patients, p=[0.6, 0.4]),
            'anticoagulants': np.random.choice([0, 1], self.num_patients, p=[0.85, 0.15]),
            'opioids': np.random.choice([0, 1], self.num_patients, p=[0.75, 0.25])
        }
        return pd.DataFrame(data)
    
    def generate_social_determinants(self):
        """Generate social determinants of health data"""
        data = {
            'patient_id': range(1, self.num_patients + 1),
            'zip_code': [f"{np.random.randint(10000, 99999)}" for _ in range(self.num_patients)],
            'employment_status': np.random.choice(['Employed', 'Unemployed', 'Retired', 'Disabled'], self.num_patients, p=[0.4, 0.2, 0.3, 0.1]),
            'education_level': np.random.choice(['High School', 'College', 'Graduate', 'Less than High School'], self.num_patients, p=[0.4, 0.3, 0.15, 0.15]),
            'housing_status': np.random.choice(['Stable', 'Unstable', 'Homeless'], self.num_patients, p=[0.85, 0.1, 0.05]),
            'transportation_access': np.random.choice(['Good', 'Limited', 'None'], self.num_patients, p=[0.7, 0.2, 0.1]),
            'social_support': np.random.choice(['Good', 'Limited', 'None'], self.num_patients, p=[0.6, 0.3, 0.1]),
            'food_insecurity': np.random.choice([0, 1], self.num_patients, p=[0.8, 0.2]),
            'health_literacy': np.random.choice(['High', 'Medium', 'Low'], self.num_patients, p=[0.5, 0.35, 0.15])
        }
        return pd.DataFrame(data)
    
    def generate_readmission_labels(self, demographics, ehr, lab, medication, social):
        """Generate readmission labels based on risk factors"""
        # Base risk
        risk = np.random.normal(0.15, 0.05, self.num_patients).clip(0.05, 0.4)
        
        # Age risk
        age_risk = (demographics['age'] - 50) / 100
        risk += age_risk * 0.1
        
        # Length of stay risk
        los_risk = ehr['length_of_stay'] / 30
        risk += los_risk * 0.15
        
        # Previous admissions risk
        prev_adm_risk = ehr['previous_admissions'] / 10
        risk += prev_adm_risk * 0.2
        
        # Lab abnormalities risk
        lab_risk = (
            (lab['glucose'] > 180).astype(int) * 0.1 +
            (lab['creatinine'] > 2.0).astype(int) * 0.15 +
            (lab['sodium'] < 135).astype(int) * 0.1 +
            (lab['hemoglobin'] < 10).astype(int) * 0.15
        )
        risk += lab_risk
        
        # Medication risk
        med_risk = (
            medication['high_risk_medications'] * 0.1 +
            (1 - medication['medication_adherence']) * 0.15
        )
        risk += med_risk
        
        # Social risk
        social_risk = (
            (social['housing_status'] != 'Stable').astype(int) * 0.1 +
            (social['transportation_access'] == 'None').astype(int) * 0.1 +
            social['food_insecurity'] * 0.1 +
            (social['health_literacy'] == 'Low').astype(int) * 0.1
        )
        risk += social_risk
        
        # Generate labels
        labels = (risk > np.random.uniform(0.2, 0.3, self.num_patients)).astype(int)
        
        data = {
            'patient_id': range(1, self.num_patients + 1),
            'readmitted_within_30days': ['Yes' if x == 1 else 'No' for x in labels]
        }
        return pd.DataFrame(data)
    
    def generate_all_data(self):
        """Generate all datasets"""
        print("Generating demographics data...")
        demographics = self.generate_demographics()
        
        print("Generating EHR data...")
        ehr = self.generate_ehr()
        
        print("Generating lab results...")
        lab = self.generate_lab_results()
        
        print("Generating medication history...")
        medication = self.generate_medication_history()
        
        print("Generating social determinants...")
        social = self.generate_social_determinants()
        
        print("Generating readmission labels...")
        labels = self.generate_readmission_labels(demographics, ehr, lab, medication, social)
        
        # Save to CSV
        demographics.to_csv('data/demographics.csv', index=False)
        ehr.to_csv('data/ehr_data.csv', index=False)
        lab.to_csv('data/lab_results.csv', index=False)
        medication.to_csv('data/medication_history.csv', index=False)
        social.to_csv('data/social_determinants.csv', index=False)
        labels.to_csv('data/readmission_labels.csv', index=False)
        
        print("All data generated successfully!")
        return demographics, ehr, lab, medication, social, labels

if __name__ == "__main__":
    generator = HealthcareDataGenerator(num_patients=2000)
    generator.generate_all_data()