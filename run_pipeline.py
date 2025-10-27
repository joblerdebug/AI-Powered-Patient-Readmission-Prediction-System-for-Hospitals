#!/usr/bin/env python3
"""
Main pipeline runner for the Patient Readmission Prediction System
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.preprocessing import MedicalDataPreprocessor
from models.advanced_training import AdvancedModelTrainer

def create_sample_data():
    """Create realistic sample patient data for demonstration"""
    print("Creating sample patient data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'patient_id': [f'P{str(i).zfill(4)}' for i in range(n_samples)],
        'age': np.random.normal(65, 15, n_samples).astype(int),
        'blood_pressure_systolic': np.random.normal(140, 20, n_samples).astype(int),
        'blood_pressure_diastolic': np.random.normal(85, 10, n_samples).astype(int),
        'cholesterol': np.random.normal(210, 40, n_samples).astype(int),
        'blood_glucose': np.random.normal(110, 25, n_samples),
        'bmi': np.random.normal(28, 5, n_samples),
        'previous_admissions': np.random.poisson(2, n_samples),
        'length_of_stay': np.random.exponential(5, n_samples).astype(int) + 1,
        'emergency_admission': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'chronic_conditions': np.random.poisson(2, n_samples),
        'medication_count': np.random.poisson(5, n_samples),
    }
    
    # Create realistic readmission risk (target variable)
    # Higher risk for older patients with more conditions and longer stays
    base_risk = (
        sample_data['age'] / 100 * 0.3 +
        sample_data['chronic_conditions'] / 10 * 0.3 +
        sample_data['length_of_stay'] / 20 * 0.2 +
        sample_data['previous_admissions'] / 10 * 0.2
    )
    
    # Add some noise and convert to binary classification
    noise = np.random.normal(0, 0.1, n_samples)
    risk_score = base_risk + noise
    sample_data['readmission_risk'] = (risk_score > 0.5).astype(int)
    
    df = pd.DataFrame(sample_data)
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_patient_data.csv', index=False)
    print(f"Sample data created with {n_samples} patients")
    return df

def run_complete_pipeline():
    """Run the complete AI pipeline from data to model training"""
    print("🚀 Starting Patient Readmission Prediction Pipeline...")
    
    # Step 1: Create sample data
    create_sample_data()
    
    # Step 2: Preprocess data
    print("\n📊 Step 1: Preprocessing medical data...")
    preprocessor = MedicalDataPreprocessor()
    df = preprocessor.load_data('data/sample_patient_data.csv')
    
    if df is not None:
        preprocessor.handle_missing_values()
        preprocessor.engineer_features()
        preprocessor.encode_categorical_features()
        preprocessor.remove_outliers()
        X, y, features = preprocessor.prepare_features()
        
        print(f"✅ Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Step 3: Train models
        print("\n🤖 Step 2: Training machine learning models...")
        trainer = AdvancedModelTrainer()
        X_train, X_test, y_train, y_test = trainer.train_models(X, y)
        
        # Step 4: Generate results
        print("\n📈 Step 3: Generating model analysis and visualizations...")
        trainer.plot_model_comparison()
        trainer.plot_confusion_matrices()
        trainer.feature_importance_analysis(features)
        
        # Step 5: Save everything
        print("\n💾 Step 4: Saving models and results...")
        trainer.save_models_and_results()
        
        # Create models directory if it doesn't exist
        os.makedirs('models/saved_models', exist_ok=True)
        
        print("\n🎉 Pipeline completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run the API: python deployment/fastapi_app.py")
        print("   2. Test the API: python deployment/test_api.py")
        print("   3. View results in models/ directory")
        
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    run_complete_pipeline()
