# eda.py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

class EDA:
    def __init__(self, data):
        self.data = data
        self.target = 'readmitted_within_30days'
        
    def patient_cohort_analysis(self):
        """Analyze patient cohorts"""
        plt.figure(figsize=(15, 10))
        
        # Age distribution by readmission
        plt.subplot(2, 2, 1)
        sns.histplot(data=self.data, x='age', hue=self.target, kde=True, bins=30)
        plt.title('Age Distribution by Readmission Status')
        
        # Gender distribution
        plt.subplot(2, 2, 2)
        gender_counts = pd.crosstab(self.data['gender'], self.data[self.target])
        gender_counts.div(gender_counts.sum(1), axis=0).plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Readmission Rate by Gender')
        
        # Primary diagnosis distribution
        plt.subplot(2, 2, 3)
        top_diagnoses = self.data['primary_diagnosis'].value_counts().nlargest(10)
        sns.barplot(x=top_diagnoses.values, y=top_diagnoses.index)
        plt.title('Top 10 Primary Diagnoses')
        
        # Length of stay distribution
        plt.subplot(2, 2, 4)
        sns.boxplot(x=self.target, y='length_of_stay', data=self.data)
        plt.title('Length of Stay by Readmission Status')
        
        plt.tight_layout()
        plt.savefig('outputs/plots/cohort_analysis.png')
        plt.show()
    
    def feature_correlation(self):
        """Analyze feature correlations"""
        plt.figure(figsize=(20, 15))
        
        # Select only numerical features
        num_data = self.data.select_dtypes(include=['float64', 'int64'])
        
        # Calculate correlation matrix
        corr = num_data.corr()
        
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Draw the heatmap
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=0.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Feature Correlation Matrix')
        plt.savefig('outputs/plots/correlation_matrix.png')
        plt.show()
        
        # Top correlations with target
        target_corr = corr[self.target].sort_values(ascending=False)
        print("Top correlations with target variable:")
        print(target_corr.head(10))
    
    def missing_data_patterns(self):
        """Analyze missing data patterns"""
        plt.figure(figsize=(15, 8))
        
        # Missing data heatmap
        plt.subplot(1, 2, 1)
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Data Heatmap')
        
        # Missing data percentage
        plt.subplot(1, 2, 2)
        missing = self.data.isnull().mean().sort_values(ascending=False)
        missing = missing[missing > 0]
        sns.barplot(x=missing, y=missing.index)
        plt.title('Percentage of Missing Data')
        plt.xlabel('Missing Percentage')
        
        plt.tight_layout()
        plt.savefig('outputs/plots/missing_data.png')
        plt.show()
    
    def categorical_feature_analysis(self):
        """Analyze categorical features"""
        cat_features = self.data.select_dtypes(include=['object']).columns
        
        for feature in cat_features:
            if feature == 'patient_id':
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Create contingency table
            contingency_table = pd.crosstab(self.data[feature], self.data[self.target])
            
            # Perform chi-square test
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            # Plot
            contingency_table.div(contingency_table.sum(1), axis=0).plot(kind='bar', stacked=True)
            plt.title(f'{feature} vs Readmission (p-value: {p:.4f})')
            plt.ylabel('Proportion')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'outputs/plots/categorical_{feature}.png')
            plt.show()
    
    def run_full_eda(self):
        """Run all EDA analyses"""
        print("Running Patient Cohort Analysis...")
        self.patient_cohort_analysis()
        
        print("\nRunning Feature Correlation Analysis...")
        self.feature_correlation()
        
        print("\nRunning Missing Data Analysis...")
        self.missing_data_patterns()
        
        print("\nRunning Categorical Feature Analysis...")
        self.categorical_feature_analysis()
        
        print("\nEDA completed. Plots saved to outputs/plots/")