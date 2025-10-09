"""
Simple GPU Offloading Predictor using XGBoost
Predicts whether a task should run on CPU or GPU
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb


def load_and_prepare_data(csv_path='expanded_training_dataset.csv'):
    """Load training data and create features"""
    df = pd.read_csv(csv_path)
    
    # Encode task names to numbers
    task_encoder = LabelEncoder()
    df['task_encoded'] = task_encoder.fit_transform(df['task'])
    
    # Create useful features
    df['size_log'] = np.log10(df['size'] + 1)
    df['threads_log'] = np.log10(df['num_cpu_threads'])
    
    # Size categories: 0=small, 1=medium, 2=large
    df['size_category'] = pd.cut(df['size'], 
                                  bins=[0, 1000, 100000, np.inf],
                                  labels=[0, 1, 2])
    
    # Select features for training
    feature_cols = ['size', 'size_log', 'size_category', 
                    'num_cpu_threads', 'threads_log', 'task_encoded']
    
    X = df[feature_cols].values
    y = df['label'].values  # 0=CPU, 1=GPU
    
    return X, y, task_encoder, feature_cols


def train_model(X_train, y_train):
    """Train XGBoost model"""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Test model accuracy"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['CPU', 'GPU']))
    
    return accuracy


def save_model(model, scaler, task_encoder, feature_names, filepath='model.pkl'):
    """Save trained model to file"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'task_encoder': task_encoder,
        'feature_names': feature_names
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved: {filepath}")


def predict_device(model, scaler, task_encoder, task_name, size, num_threads):
    """
    Predict optimal device for a task
    
    Args:
        task_name: 'VectorAddition', 'MatrixMultiplication', or 'ParallelReduction'
        size: Problem size (number of elements or matrix dimension)
        num_threads: Number of CPU threads available
    
    Returns:
        device: 'CPU' or 'GPU'
        confidence: Prediction confidence (0-1)
    """
    # Encode task
    task_encoded = task_encoder.transform([task_name])[0]
    
    # Calculate features
    size_log = np.log10(size + 1)
    threads_log = np.log10(num_threads)
    
    if size <= 1000:
        size_category = 0
    elif size <= 100000:
        size_category = 1
    else:
        size_category = 2
    
    # Create feature array
    features = np.array([[size, size_log, size_category, 
                         num_threads, threads_log, task_encoded]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    device = 'GPU' if prediction == 1 else 'CPU'
    confidence = probabilities[prediction]
    
    return device, confidence


def main():
    print("Training GPU Offloading Predictor...")
    
    # Load data
    X, y, task_encoder, feature_names = load_and_prepare_data()
    print(f"Loaded {len(X)} training samples")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training XGBoost model...")
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate
    accuracy = evaluate_model(model, X_test_scaled, y_test)
    
    # Save model
    save_model(model, scaler, task_encoder, feature_names, 'xgboost_model.pkl')
    
    # Demo predictions
    print("\n" + "="*60)
    print("Demo Predictions:")
    print("="*60)
    
    test_cases = [
        ('VectorAddition', 1000, 4),
        ('VectorAddition', 1000000, 8),
        ('MatrixMultiplication', 64, 4),
        ('MatrixMultiplication', 512, 8),
        ('ParallelReduction', 10000, 2),
        ('ParallelReduction', 10000000, 16),
    ]
    
    for task, size, threads in test_cases:
        device, confidence = predict_device(model, scaler, task_encoder, 
                                           task, size, threads)
        print(f"{task:25s} (size={size:>8,}, threads={threads:>2}) → {device:3s} ({confidence:.0%})")
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
