"""
Train ML models to predict optimal device (CPU vs GPU) for task offloading
Uses the expanded_training_dataset.csv with 1200+ rows
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns

class TaskOffloadingPredictor:
    """ML model to predict CPU vs GPU offloading decisions"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize predictor with specified model type
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'logistic_regression'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.task_encoder = LabelEncoder()
        self.feature_names = None
        self.trained = False
        
    def load_data(self, csv_path='/mnt/d/OSprojects/GPU-Offloading-Task-Scheduler/nsights_code/expanded_training_dataset.csv'):
        """Load and prepare training data"""
        print(f"📂 Loading data from {csv_path}...")
        df = pd.read_csv("/mnt/d/OSprojects/GPU-Offloading-Task-Scheduler/nsights_code/expanded_training_dataset.csv")
        
        print(f"   Loaded {len(df)} rows")
        print(f"   Tasks: {df['task'].unique()}")
        print(f"   Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def engineer_features(self, df):
        """Create engineered features for better prediction"""
        print("\n🔧 Engineering features...")
        
        df = df.copy()
        
        # Logarithmic size (many ML algorithms work better with log-scaled features)
        df['size_log'] = np.log10(df['size'] + 1)
        
        # Time ratio (strong indicator)
        df['time_ratio'] = df['cpu_time_ms'] / (df['gpu_time_ms'] + 1e-6)
        df['time_ratio_log'] = np.log10(df['time_ratio'] + 1)
        
        # Relative standard deviations (measure of consistency)
        df['cpu_cv'] = df['cpu_std_ms'] / (df['cpu_time_ms'] + 1e-6)
        df['gpu_cv'] = df['gpu_std_ms'] / (df['gpu_time_ms'] + 1e-6)
        
        # Task encoding
        df['task_encoded'] = self.task_encoder.fit_transform(df['task'])
        
        # Thread efficiency (performance per thread)
        df['threads_log'] = np.log10(df['num_cpu_threads'])
        
        # Size category (small, medium, large)
        df['size_category'] = pd.cut(df['size'], 
                                      bins=[0, 1000, 100000, np.inf],
                                      labels=[0, 1, 2])
        
        print(f"   Created {len([c for c in df.columns if c not in ['task', 'optimal_device']])} features")
        
        return df
    
    def prepare_features(self, df, inference_mode=False):
        """
        Prepare feature matrix and target vector
        
        Args:
            df: DataFrame with data
            inference_mode: If True, don't expect 'label' column
        """
        # Select features for training
        feature_cols = [
            'size', 'size_log', 'size_category',
            'num_cpu_threads', 'threads_log',
            'task_encoded',
            'time_ratio_log', 'cpu_cv', 'gpu_cv'
        ]
        
        # For inference, we might not have timing data
        if inference_mode:
            feature_cols = [
                'size', 'size_log', 'size_category',
                'num_cpu_threads', 'threads_log',
                'task_encoded'
            ]
        
        self.feature_names = feature_cols
        X = df[feature_cols].values
        
        if not inference_mode:
            y = df['label'].values
            return X, y
        else:
            return X
    
    def train(self, X_train, y_train, X_test=None, y_test=None, tune_hyperparameters=False):
        """Train the ML model"""
        print(f"\n🎓 Training {self.model_type} model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize model
        if self.model_type == 'random_forest':
            if tune_hyperparameters:
                print("   Tuning hyperparameters with GridSearch...")
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
                base_model = RandomForestClassifier(random_state=42)
                self.model = GridSearchCV(base_model, param_grid, cv=3, 
                                         scoring='f1', n_jobs=-1, verbose=1)
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
        
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        if tune_hyperparameters and self.model_type == 'random_forest':
            print(f"   Best parameters: {self.model.best_params_}")
            self.model = self.model.best_estimator_
        
        # Evaluate on training set
        y_train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"   Training Accuracy: {train_acc:.4f}")
        
        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            y_test_pred = self.model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            print(f"   Test Accuracy: {test_acc:.4f}")
            print(f"   Test F1 Score: {test_f1:.4f}")
            
            # Detailed classification report
            print("\n📊 Classification Report:")
            print(classification_report(y_test, y_test_pred, 
                                       target_names=['CPU', 'GPU']))
        
        self.trained = True
        print("✅ Model training complete!")
        
    def predict(self, X):
        """Predict optimal device for given features"""
        if not self.trained:
            raise ValueError("Model not trained! Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def predict_device(self, task_name, size, num_threads):
        """
        Predict optimal device for a single task
        
        Args:
            task_name: 'VectorAddition', 'MatrixMultiplication', or 'ParallelReduction'
            size: Problem size
            num_threads: Number of CPU threads
            
        Returns:
            device: 'CPU' or 'GPU'
            confidence: Prediction confidence (0-1)
        """
        if not self.trained:
            raise ValueError("Model not trained! Call train() first.")
        
        # Create feature vector
        task_encoded = self.task_encoder.transform([task_name])[0]
        size_log = np.log10(size + 1)
        threads_log = np.log10(num_threads)
        
        # Size category
        if size <= 1000:
            size_category = 0
        elif size <= 100000:
            size_category = 1
        else:
            size_category = 2
        
        # For inference without timing data, use dummy values
        # The model won't rely heavily on these since we don't have real timing
        time_ratio_log = 0  # Neutral value
        cpu_cv = 0.1  # Typical CV
        gpu_cv = 0.1  # Typical CV
        
        # Create feature array matching training features
        features = np.array([[
            size, size_log, size_category,
            num_threads, threads_log,
            task_encoded,
            time_ratio_log, cpu_cv, gpu_cv
        ]])
        
        # Predict
        X_scaled = self.scaler.transform(features)
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        device = 'GPU' if prediction == 1 else 'CPU'
        confidence = probabilities[prediction]
        
        return device, confidence
    
    def get_feature_importance(self):
        """Get feature importance (for tree-based models)"""
        if self.model_type in ['random_forest', 'gradient_boosting']:
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance
        else:
            return None
    
    def save_model(self, filepath='gpu_offloading_model.pkl'):
        """Save trained model to disk"""
        if not self.trained:
            raise ValueError("Model not trained! Nothing to save.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'task_encoder': self.task_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath='gpu_offloading_model.pkl'):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.task_encoder = model_data['task_encoder']
        predictor.feature_names = model_data['feature_names']
        predictor.trained = True
        
        print(f"✅ Model loaded from {filepath}")
        return predictor


def train_and_evaluate_models():
    """Train and compare multiple models"""
    print("="*70)
    print(" GPU OFFLOADING ML MODEL TRAINING")
    print("="*70)
    
    # Try different models
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f" Training {model_type.upper().replace('_', ' ')}")
        print(f"{'='*70}")
        
        # Initialize predictor
        predictor = TaskOffloadingPredictor(model_type=model_type)
        
        # Load and prepare data
        df = predictor.load_data('expanded_training_dataset.csv')
        df = predictor.engineer_features(df)
        X, y = predictor.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
        
        # Train model
        predictor.train(X_train, y_train, X_test, y_test)
        
        # Evaluate
        X_test_scaled = predictor.scaler.transform(X_test)
        y_pred = predictor.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[model_type] = {
            'accuracy': accuracy,
            'f1': f1,
            'predictor': predictor
        }
        
        # Show feature importance for tree models
        if model_type in ['random_forest', 'gradient_boosting']:
            print("\n🎯 Feature Importance:")
            importance_df = predictor.get_feature_importance()
            print(importance_df.to_string(index=False))
    
    # Compare results
    print(f"\n{'='*70}")
    print(" MODEL COMPARISON")
    print(f"{'='*70}")
    
    for model_type, metrics in results.items():
        print(f"\n{model_type.upper().replace('_', ' ')}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Select best model
    best_model_type = max(results, key=lambda k: results[k]['f1'])
    best_predictor = results[best_model_type]['predictor']
    
    print(f"\n🏆 Best Model: {best_model_type.upper().replace('_', ' ')}")
    print(f"   F1 Score: {results[best_model_type]['f1']:.4f}")
    
    # Save best model
    best_predictor.save_model('gpu_offloading_model_best.pkl')
    
    # Demo predictions
    print(f"\n{'='*70}")
    print(" DEMO PREDICTIONS")
    print(f"{'='*70}")
    
    test_cases = [
        ('VectorAddition', 1000, 4),
        ('VectorAddition', 1000000, 8),
        ('MatrixMultiplication', 64, 4),
        ('MatrixMultiplication', 512, 8),
        ('ParallelReduction', 10000, 2),
        ('ParallelReduction', 10000000, 16),
    ]
    
    for task, size, threads in test_cases:
        device, confidence = best_predictor.predict_device(task, size, threads)
        print(f"\n{task} (size={size:,}, threads={threads}):")
        print(f"  → Predicted: {device} (confidence: {confidence:.2%})")
    
    return best_predictor, results


if __name__ == "__main__":
    predictor, results = train_and_evaluate_models()
    print("\n✅ Training complete! Model saved as 'gpu_offloading_model_best.pkl'")
