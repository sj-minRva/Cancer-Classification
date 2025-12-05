#!/usr/bin/env python3
"""
Integrated Model Service for Cancer Classification
Handles model loading, training, and prediction serving for multiple cancer types
"""

import joblib
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, cohen_kappa_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.utils import shuffle
from xgboost import XGBClassifier
import xgboost as xgb

class CancerModelService:
    """Integrated service class for managing cancer classification models"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.models = {}
        self.datasets_info = {
            'breast': {
                'file_path': 'C:/RHEA/S7/Project/Code/Datasets/BRCA_gene_expression.csv',
                'target_column': 'classes',
                'index_col': 'Unnamed: 0'
            },
            'gastric': {
                'file_path': 'C:/RHEA/S7/Project/Code/Datasets/Gastric_gene.csv',
                'target_column': 'Sample_Characteristics',
                'index_col': None
            },
            'lung': {
                'file_path': 'C:/RHEA/S7/Project/Code/Datasets/Lung_gene.csv',
                'target_column': 'Sample_Characteristics',
                'index_col': None
            }
        }
        self.load_all_models()
    
    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess dataset for training"""
        if dataset_name not in self.datasets_info:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        info = self.datasets_info[dataset_name]
        
        # Load dataset
        if info['index_col']:
            df = pd.read_csv(info['file_path'], index_col=info['index_col'])
        else:
            df = pd.read_csv(info['file_path'])
        
        # Encode target variable
        le = LabelEncoder()
        df[info['target_column']] = le.fit_transform(df[info['target_column']])
        
        # Define features and target
        X = df.drop(columns=[info['target_column']])
        y = df[info['target_column']]
        
        return X, y
    
    def create_synthetic_data(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic data for datasets that don't exist"""
        if dataset_name == 'gastric':
            np.random.seed(42)
            n_samples, n_features = 500, 20
            X = np.random.randn(n_samples, n_features)
            X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
            X[:, 2] = X[:, 0] - 0.3 * X[:, 1] + 0.4 * np.random.randn(n_samples)
            y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        elif dataset_name == 'lung':
            np.random.seed(123)
            n_samples, n_features = 600, 18
            X = np.random.randn(n_samples, n_features)
            for i in range(1, n_features):
                X[:, i] = 0.3 * X[:, 0] + 0.7 * np.random.randn(n_samples)
            y = (2 * X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.7 > 0).astype(int)
        else:
            raise ValueError(f"Synthetic data not available for {dataset_name}")
        
        return X, y
    
    def train_ensemble_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                           y_train: np.ndarray, y_test: np.ndarray, 
                           meta_model_type: str, dataset_name: str) -> Dict[str, Any]:
        """Train ensemble model with XGBoost as base and specified meta-learner"""
        
        # Train XGBoost base model
        xgb_model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=6,
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        
        # Get XGBoost predictions as features for meta-learner
        xgb_train_pred = xgb_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
        xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
        
        # Combine original features with XGBoost predictions
        meta_train_features = np.hstack([X_train, xgb_train_pred])
        meta_test_features = np.hstack([X_test, xgb_test_pred])
        
        # Initialize meta-learner based on type
        if meta_model_type == 'svm':
            meta_model = SVC(probability=True, random_state=42)
        elif meta_model_type == 'lr':
            meta_model = LogisticRegression(random_state=42, max_iter=1000)
        elif meta_model_type == 'rf':
            meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown meta model type: {meta_model_type}")
        
        # Train meta-learner
        meta_model.fit(meta_train_features, y_train)
        
        # Make final predictions
        y_pred = meta_model.predict(meta_test_features)
        y_pred_proba = meta_model.predict_proba(meta_test_features)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        print(f"{dataset_name} XGB+{meta_model_type.upper()} Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Kappa: {kappa:.4f}")
        print()
        
        # Create ensemble model for saving
        ensemble_model = {
            'xgb_model': xgb_model,
            'meta_model': meta_model,
            'scaler': StandardScaler().fit(X_train),
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'auc': auc,
                'kappa': kappa
            }
        }
        
        return ensemble_model
    
    def train_models_for_dataset(self, dataset_name: str, use_synthetic: bool = False) -> None:
        """Train all three ensemble models for a given dataset"""
        
        print(f"\nTraining models for {dataset_name} cancer dataset...")
        
        # Load or create data
        if use_synthetic or not os.path.exists(self.datasets_info[dataset_name]['file_path']):
            print(f"Using synthetic data for {dataset_name}")
            X, y = self.create_synthetic_data(dataset_name)
        else:
            try:
                X, y = self.load_dataset(dataset_name)
            except Exception as e:
                print(f"Error loading {dataset_name} dataset: {e}")
                print(f"Using synthetic data for {dataset_name}")
                X, y = self.create_synthetic_data(dataset_name)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Features: {X.shape[1]}")
        
        # Create output directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Train all three ensemble models
        model_types = ['svm', 'lr', 'rf']
        for model_type in model_types:
            try:
                ensemble_model = self.train_ensemble_model(
                    X_train_scaled, X_test_scaled, y_train, y_test, 
                    model_type, dataset_name
                )
                
                # Save model
                model_path = os.path.join(self.models_dir, f'{dataset_name}_cancer_xgb_{model_type}.pkl')
                joblib.dump(ensemble_model, model_path)
                print(f"Saved {dataset_name} XGB+{model_type.upper()} model to {model_path}")
                
            except Exception as e:
                print(f"Error training {dataset_name} XGB+{model_type.upper()}: {e}")
    
    def load_all_models(self) -> None:
        """Load all trained models from disk"""
        datasets = ['breast', 'gastric', 'lung']
        model_types = ['xgb_svm', 'xgb_lr', 'xgb_rf']
        
        for dataset in datasets:
            self.models[dataset] = {}
            for model_type in model_types:
                model_path = os.path.join(self.models_dir, f'{dataset}_cancer_{model_type}.pkl')
                if os.path.exists(model_path):
                    try:
                        self.models[dataset][model_type] = joblib.load(model_path)
                        print(f"Loaded {dataset} {model_type} model")
                    except Exception as e:
                        print(f"Error loading {model_path}: {e}")
                        self.models[dataset][model_type] = None
                else:
                    print(f"Model file not found: {model_path}")
                    self.models[dataset][model_type] = None
    
    def predict_single(self, dataset: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction for a single sample"""
        if dataset not in self.models:
            raise ValueError(f"Dataset {dataset} not supported")
        
        # Convert features to array
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        predictions = {}
        for model_type, model in self.models[dataset].items():
            if model is None:
                continue
                
            try:
                # Scale features if scaler is available
                if 'scaler' in model:
                    feature_array_scaled = model['scaler'].transform(feature_array)
                else:
                    feature_array_scaled = feature_array
                
                # Get XGBoost prediction
                xgb_pred = model['xgb_model'].predict_proba(feature_array_scaled)[:, 1]
                
                # Combine with original features for meta-learner
                meta_features = np.hstack([feature_array_scaled, xgb_pred.reshape(-1, 1)])
                
                # Get final prediction
                prediction = model['meta_model'].predict(meta_features)[0]
                probability = model['meta_model'].predict_proba(meta_features)[0, 1]
                
                predictions[model_type] = {
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'label': 'Malignant' if prediction == 1 else 'Benign'
                }
                
            except Exception as e:
                print(f"Error predicting with {model_type}: {e}")
                predictions[model_type] = {
                    'prediction': 0,
                    'probability': 0.5,
                    'label': 'Error'
                }
        
        return predictions
    
    def predict_batch(self, dataset: str, features_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Make predictions for multiple samples"""
        results = []
        for i, features in enumerate(features_list):
            try:
                predictions = self.predict_single(dataset, features)
                results.append({
                    'row': i + 1,
                    'predictions': predictions,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'row': i + 1,
                    'predictions': {},
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def get_model_metrics(self, dataset: str = None) -> Dict[str, Any]:
        """Get stored model metrics"""
        metrics = []
        
        datasets_to_check = [dataset] if dataset else self.models.keys()
        
        for ds in datasets_to_check:
            if ds not in self.models:
                continue
                
            for model_type, model in self.models[ds].items():
                if model is None or 'metrics' not in model:
                    continue
                
                model_metrics = model['metrics'].copy()
                model_metrics['dataset'] = ds
                model_metrics['model'] = model_type
                metrics.append(model_metrics)
        
        return {'metrics': metrics}
    
    def health_check(self) -> Dict[str, Any]:
        """Check if all models are loaded properly"""
        status = {}
        total_models = 0
        loaded_models = 0
        
        for dataset, models in self.models.items():
            status[dataset] = {}
            for model_type, model in models.items():
                is_loaded = model is not None
                status[dataset][model_type] = is_loaded
                total_models += 1
                if is_loaded:
                    loaded_models += 1
        
        return {
            'status': 'healthy' if loaded_models == total_models else 'partial',
            'loaded_models': loaded_models,
            'total_models': total_models,
            'details': status
        }
    
    def retrain_all_models(self, use_synthetic: bool = False) -> None:
        """Retrain all models for all datasets"""
        print("Retraining all cancer classification models...")
        print("=" * 50)
        
        datasets = ['breast', 'gastric', 'lung']
        for dataset in datasets:
            self.train_models_for_dataset(dataset, use_synthetic)
        
        # Reload models after training
        self.load_all_models()
        print("\nModel retraining completed!")

# Global model service instance
model_service = None

def get_model_service() -> CancerModelService:
    """Get global model service instance"""
    global model_service
    if model_service is None:
        model_service = CancerModelService()
    return model_service

if __name__ == "__main__":
    # Test the model service
    service = CancerModelService()
    
    # Health check
    health = service.health_check()
    print("Health check:", health)
    
    # Test prediction
    test_features = {
        'feature_1': 14.127,
        'feature_2': 19.289,
        'feature_3': 91.969,
        'feature_4': 654.889,
        'feature_5': 0.096,
        'feature_6': 0.104,
        'feature_7': 0.089,
        'feature_8': 0.048
    }
    
    try:
        predictions = service.predict_single('breast', test_features)
        print("Test predictions:", predictions)
    except Exception as e:
        print("Test prediction failed:", e)
    
    # Get metrics
    metrics = service.get_model_metrics()
    print("Model metrics:", metrics)