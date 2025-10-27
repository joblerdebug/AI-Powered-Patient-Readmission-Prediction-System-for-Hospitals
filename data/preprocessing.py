import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class MedicalDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_selector = SelectKBest(f_classif, k=20)
        self.encoders = {}
        
    def load_data(self, filepath):
        """Load and validate medical data"""
        try:
            self.df = pd.read_csv(filepath)
            print(f"Data loaded: {self.df.shape[0]} patients, {self.df.shape[1]} features")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def handle_missing_values(self):
        """Advanced missing value handling"""
        print("Handling missing values...")
        
        # Separate numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Use KNN imputer for numeric features
        if not numeric_cols.empty:
            self.df[numeric_cols] = self.imputer.fit_transform(self.df[numeric_cols])
        
        # For categorical, use mode or 'Unknown'
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        return self.df
    
    def engineer_features(self):
        """Create advanced medical features"""
        print("Engineering features...")
        
        # Comorbidity scores
        if 'diagnosis_codes' in self.df.columns:
            self.df['comorbidity_count'] = self.df['diagnosis_codes'].apply(
                lambda x: len(x.split(',')) if isinstance(x, str) else 0
            )
        
        # Medication complexity
        if 'medications' in self.df.columns:
            self.df['medication_count'] = self.df['medications'].apply(
                lambda x: len(x.split(',')) if isinstance(x, str) else 0
            )
        
        # Age groups
        if 'age' in self.df.columns:
            self.df['age_group'] = pd.cut(
                self.df['age'], 
                bins=[0, 30, 50, 65, 100], 
                labels=['Young', 'Adult', 'Senior', 'Elderly']
            )
        
        # Previous healthcare utilization
        if 'previous_admissions' in self.df.columns:
            self.df['frequent_admitter'] = (self.df['previous_admissions'] > 3).astype(int)
        
        print(f"Added {len([col for col in self.df.columns if 'engineered' in col])} engineered features")
        return self.df
    
    def encode_categorical_features(self):
        """Encode categorical variables with proper handling"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in ['patient_id', 'readmission_risk']:
                self.encoders[col] = LabelEncoder()
                self.df[col] = self.encoders[col].fit_transform(self.df[col].astype(str))
        
        return self.df
    
    def remove_outliers(self):
        """Remove medical outliers using IQR method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['readmission_risk', 'patient_id']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing for medical data
                self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
        
        return self.df
    
    def prepare_features(self, target_column='readmission_risk'):
        """Final feature preparation"""
        # Separate features and target
        X = self.df.drop(columns=[target_column, 'patient_id'] if 'patient_id' in self.df.columns else [target_column])
        y = self.df[target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        feature_names = X.columns.tolist()
        
        # Select best features
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        selected_mask = self.feature_selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        print(f"Selected {len(selected_features)} most important features")
        print(f"Selected features: {selected_features}")
        
        return X_selected, y, selected_features

# Example usage
if __name__ == "__main__":
    preprocessor = MedicalDataPreprocessor()
    
    # Sample data creation for demonstration
    sample_data = {
        'patient_id': range(1000),
        'age': np.random.normal(55, 15, 1000),
        'blood_pressure': np.random.normal(120, 20, 1000),
        'cholesterol': np.random.normal(200, 40, 1000),
        'diagnosis_codes': [','.join(np.random.choice(['I10', 'E11', 'J45'], 2)) for _ in range(1000)],
        'medications': [','.join(np.random.choice(['A', 'B', 'C'], 3)) for _ in range(1000)],
        'previous_admissions': np.random.poisson(2, 1000),
        'readmission_risk': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/sample_patient_data.csv', index=False)
    
    # Process the data
    preprocessor.load_data('data/sample_patient_data.csv')
    preprocessor.handle_missing_values()
    preprocessor.engineer_features()
    preprocessor.encode_categorical_features()
    preprocessor.remove_outliers()
    X, y, features = preprocessor.prepare_features()
    
    print("Data preprocessing completed successfully!")
