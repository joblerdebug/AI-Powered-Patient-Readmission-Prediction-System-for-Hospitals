import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, auc)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

class AdvancedModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """Train multiple models with hyperparameter tuning"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Define models and parameters for tuning
        model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=random_state),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6],
                    'learning_rate': [0.01, 0.1]
                }
            }
        }
        
        # Train and tune each model
        for name, config in model_configs.items():
            print(f"Training {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], config['params'], 
                cv=5, scoring='roc_auc', n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store best model
            self.models[name] = grid_search.best_estimator_
            
            # Evaluate on test set
            y_pred = self.models[name].predict(X_test)
            y_pred_proba = self.models[name].predict_proba(X_test)[:, 1]
            
            # Store results
            self.results[name] = {
                'best_params': grid_search.best_params_,
                'test_accuracy': grid_search.score(X_test, y_test),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"{name} - Best params: {grid_search.best_params_}")
            print(f"{name} - ROC AUC: {self.results[name]['roc_auc']:.4f}")
        
        # Select best model based on ROC AUC
        best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        self.best_model = self.models[best_model_name]
        
        print(f"\nBest model: {best_model_name} with ROC AUC: {self.results[best_model_name]['roc_auc']:.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def plot_model_comparison(self):
        """Create visualization comparing model performance"""
        models = list(self.results.keys())
        auc_scores = [self.results[model]['roc_auc'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, auc_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title('Model Comparison - ROC AUC Scores', fontsize=16)
        plt.ylabel('ROC AUC Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.4f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name.title()}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self, feature_names):
        """Analyze and plot feature importance for tree-based models"""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 8))
            plt.title('Feature Importance - Best Model')
            plt.barh(range(min(10, len(feature_names))), 
                    importances[indices][:10][::-1], 
                    color='teal', alpha=0.7)
            plt.yticks(range(min(10, len(feature_names))), 
                      [feature_names[i] for i in indices[:10]][::-1])
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_models_and_results(self):
        """Save all models and results"""
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f'models/saved_models/{name}_model.joblib')
        
        # Save best model separately
        joblib.dump(self.best_model, 'models/saved_models/best_model.joblib')
        
        # Save results as JSON
        with open('models/training_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for model_name, result in self.results.items():
                json_results[model_name] = {
                    'best_params': result['best_params'],
                    'test_accuracy': float(result['test_accuracy']),
                    'roc_auc': float(result['roc_auc']),
                    'confusion_matrix': result['confusion_matrix'].tolist(),
                    'classification_report': result['classification_report']
                }
            json.dump(json_results, f, indent=2)
        
        print("Models and results saved successfully!")

# Example usage
if __name__ == "__main__":
    # Generate sample data for demonstration
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, n_clusters_per_class=1, random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Train models
    trainer = AdvancedModelTrainer()
    X_train, X_test, y_train, y_test = trainer.train_models(X, y)
    
    # Generate visualizations
    trainer.plot_model_comparison()
    trainer.plot_confusion_matrices()
    trainer.feature_importance_analysis(feature_names)
    
    # Save everything
    trainer.save_models_and_results()
