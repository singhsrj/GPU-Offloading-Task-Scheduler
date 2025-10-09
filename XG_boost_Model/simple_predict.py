"""
Load trained model and make predictions
Simple script to predict CPU vs GPU for tasks
"""

import pickle
import numpy as np


def load_model(filepath='xgboost_model.pkl'):
    """Load the trained model"""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✓ Model loaded from {filepath}")
    return model_data


def predict(model_data, task_name, size, num_threads):
    """
    Predict optimal device
    
    Args:
        task_name: 'VectorAddition', 'MatrixMultiplication', or 'ParallelReduction'
        size: Problem size
        num_threads: Number of CPU threads
    
    Returns:
        device: 'CPU' or 'GPU'
        confidence: 0-1 probability
    """
    model = model_data['model']
    scaler = model_data['scaler']
    task_encoder = model_data['task_encoder']
    
    # Encode task to number
    task_encoded = task_encoder.transform([task_name])[0]
    
    # Calculate features
    size_log = np.log10(size + 1)
    threads_log = np.log10(num_threads)
    
    # Size category
    if size <= 1000:
        size_category = 0
    elif size <= 100000:
        size_category = 1
    else:
        size_category = 2
    
    # Build feature array
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
    # Load model
    model_data = load_model('xgboost_model.pkl')
    
    print("\nExample Predictions:")
    print("-" * 60)
    
    # Test cases
    examples = [
        ('VectorAddition', 1000, 4),
        ('VectorAddition', 1000000, 8),
        ('MatrixMultiplication', 64, 4),
        ('MatrixMultiplication', 512, 8),
        ('ParallelReduction', 10000, 2),
        ('ParallelReduction', 10000000, 16),
    ]
    
    for task, size, threads in examples:
        device, conf = predict(model_data, task, size, threads)
        print(f"{task:25s} size={size:>9,} threads={threads:>2} → {device:3s} ({conf:>5.1%})")
    
    print("\n✓ Predictions complete!")


if __name__ == "__main__":
    main()
