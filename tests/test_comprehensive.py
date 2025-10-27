import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import MedicalDataPreprocessor
from models.advanced_training import AdvancedModelTrainer

class TestMedicalAISystem:
    def test_data_preprocessing(self):
        """Test the complete data preprocessing pipeline"""
        preprocessor = MedicalDataPreprocessor()
        
        # Create sample medical data
        sample_data = {
            'patient_id': range(100),
            'age': np.random.normal(55, 15, 100),
            'blood_pressure': np.random.normal(120, 20, 100),
            'cholesterol': np.random.normal(200, 40, 100),
            'previous_admissions': np.random.poisson(2, 100),
            'readmission_risk': np.random.choice([0, 1], 100, p=[0.7, 0.3])
        }
        
        df = pd.DataFrame(sample_data)
        
        # Test preprocessing steps
        preprocessor.df = df.copy()
        preprocessor.handle_missing_values()
        preprocessor.engineer_features()
        preprocessor.encode_categorical_features()
        preprocessor.remove_outliers()
        X, y, features = preprocessor.prepare_features()
        
        assert X.shape[0] == 100, "Should process all samples"
        assert len(features) > 0, "Should select important features"
    
    def test_model_training(self):
        """Test model training pipeline"""
        # Generate sample data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        trainer = AdvancedModelTrainer()
        X_train, X_test, y_train, y_test = trainer.train_models(X, y)
        
        # Check if models were trained
        assert len(trainer.models) > 0, "Should train multiple models"
        assert trainer.best_model is not None, "Should select a best model"
        
        # Check if results are stored
        assert len(trainer.results) > 0, "Should store model results"
        
        # Check if best model has reasonable performance
        best_model_name = max(trainer.results, key=lambda x: trainer.results[x]['roc_auc'])
        auc_score = trainer.results[best_model_name]['roc_auc']
        assert auc_score > 0.5, "Best model should perform better than random"
    
    def test_feature_importance(self):
        """Test feature importance analysis"""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        feature_names = [f'feature_{i}' for i in range(5)]
        
        trainer = AdvancedModelTrainer()
        trainer.train_models(X, y)
        
        # This should run without errors
        try:
            trainer.feature_importance_analysis(feature_names)
            assert True
        except Exception as e:
            pytest.fail(f"Feature importance analysis failed: {e}")
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        trainer = AdvancedModelTrainer()
        trainer.train_models(X, y)
        
        # Test saving
        try:
            trainer.save_models_and_results()
            assert os.path.exists('models/training_results.json')
            assert os.path.exists('models/saved_models/best_model.joblib')
        except Exception as e:
            pytest.fail(f"Model saving failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
