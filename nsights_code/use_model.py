"""
Demo script showing how to use the trained ML model for GPU offloading predictions
"""

import pickle
import numpy as np
from train_ml_model import TaskOffloadingPredictor

def load_trained_model(model_path='gpu_offloading_model_best.pkl'):
    """Load the pre-trained model"""
    print("🔧 Loading trained model...")
    predictor = TaskOffloadingPredictor.load_model(model_path)
    return predictor

def predict_examples():
    """Run predictions on example tasks"""
    print("\n" + "="*70)
    print(" GPU OFFLOADING PREDICTION DEMO")
    print("="*70)
    
    # Load model
    predictor = load_trained_model()
    
    # Define test cases
    test_cases = [
        # (task_name, size, num_threads, expected_result)
        ('VectorAddition', 500, 1, 'Small vector - might be CPU'),
        ('VectorAddition', 100000, 4, 'Medium vector - likely GPU'),
        ('VectorAddition', 10000000, 8, 'Large vector - definitely GPU'),
        
        ('MatrixMultiplication', 32, 2, 'Tiny matrix - CPU overhead'),
        ('MatrixMultiplication', 128, 4, 'Small matrix - borderline'),
        ('MatrixMultiplication', 512, 8, 'Medium matrix - likely GPU'),
        ('MatrixMultiplication', 1024, 16, 'Large matrix - definitely GPU'),
        
        ('ParallelReduction', 1000, 1, 'Small reduction - might be CPU'),
        ('ParallelReduction', 100000, 4, 'Medium reduction - likely GPU'),
        ('ParallelReduction', 10000000, 16, 'Large reduction - definitely GPU'),
    ]
    
    print("\n📊 PREDICTION RESULTS:\n")
    
    for task, size, threads, description in test_cases:
        device, confidence = predictor.predict_device(task, size, threads)
        
        print(f"Task: {task}")
        print(f"  Size: {size:,} | Threads: {threads}")
        print(f"  Description: {description}")
        print(f"  ✓ Prediction: {device} (confidence: {confidence:.1%})")
        print()

def interactive_prediction():
    """Interactive prediction mode"""
    print("\n" + "="*70)
    print(" INTERACTIVE PREDICTION MODE")
    print("="*70)
    
    predictor = load_trained_model()
    
    print("\nAvailable tasks:")
    print("  1. VectorAddition")
    print("  2. MatrixMultiplication")
    print("  3. ParallelReduction")
    
    while True:
        print("\n" + "-"*70)
        
        # Get input
        task_choice = input("Select task (1-3, or 'q' to quit): ").strip()
        
        if task_choice.lower() == 'q':
            break
        
        if task_choice not in ['1', '2', '3']:
            print("Invalid choice!")
            continue
        
        task_map = {
            '1': 'VectorAddition',
            '2': 'MatrixMultiplication',
            '3': 'ParallelReduction'
        }
        task_name = task_map[task_choice]
        
        try:
            size = int(input("Enter problem size: "))
            threads = int(input("Enter number of CPU threads: "))
        except ValueError:
            print("Invalid input! Please enter numbers.")
            continue
        
        # Make prediction
        device, confidence = predictor.predict_device(task_name, size, threads)
        
        print(f"\n🎯 RESULT:")
        print(f"   Task: {task_name}")
        print(f"   Size: {size:,}")
        print(f"   Threads: {threads}")
        print(f"   → Recommended Device: {device}")
        print(f"   → Confidence: {confidence:.1%}")
        
        if confidence < 0.6:
            print("   ⚠️  Low confidence - performance may be similar on both devices")
        elif confidence > 0.9:
            print("   ✓ High confidence - strong recommendation")

def batch_prediction_from_file(input_file='tasks_to_predict.csv'):
    """Predict optimal device for batch of tasks from CSV file"""
    import pandas as pd
    
    print("\n" + "="*70)
    print(" BATCH PREDICTION MODE")
    print("="*70)
    
    try:
        df = pd.read_csv(input_file)
        print(f"\n📂 Loaded {len(df)} tasks from {input_file}")
    except FileNotFoundError:
        print(f"\n⚠️  File not found: {input_file}")
        print("Creating example file...")
        
        # Create example file
        example_data = {
            'task': ['VectorAddition', 'MatrixMultiplication', 'ParallelReduction'] * 3,
            'size': [1000, 64, 10000, 100000, 256, 1000000, 10000000, 1024, 100000000],
            'num_threads': [4, 4, 4, 8, 8, 8, 16, 16, 16]
        }
        df = pd.DataFrame(example_data)
        df.to_csv(input_file, index=False)
        print(f"✓ Created example file: {input_file}")
    
    # Load model
    predictor = load_trained_model()
    
    # Make predictions
    predictions = []
    confidences = []
    
    for _, row in df.iterrows():
        device, confidence = predictor.predict_device(
            row['task'], row['size'], row['num_threads']
        )
        predictions.append(device)
        confidences.append(confidence)
    
    df['predicted_device'] = predictions
    df['confidence'] = confidences
    
    # Save results
    output_file = input_file.replace('.csv', '_predictions.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\n📊 Results:")
    print(df.to_string(index=False))
    print(f"\n💾 Saved predictions to: {output_file}")
    
    # Summary statistics
    print(f"\n📈 Summary:")
    print(f"   CPU recommended: {(df['predicted_device'] == 'CPU').sum()} tasks")
    print(f"   GPU recommended: {(df['predicted_device'] == 'GPU').sum()} tasks")
    print(f"   Average confidence: {df['confidence'].mean():.1%}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == 'interactive':
            interactive_prediction()
        elif mode == 'batch':
            batch_file = sys.argv[2] if len(sys.argv) > 2 else 'tasks_to_predict.csv'
            batch_prediction_from_file(batch_file)
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python use_model.py [interactive|batch] [batch_file]")
    else:
        # Default: run examples
        predict_examples()
        
        print("\n" + "="*70)
        print(" OTHER MODES")
        print("="*70)
        print("\nTo use interactive mode:")
        print("  python use_model.py interactive")
        print("\nTo predict from file:")
        print("  python use_model.py batch tasks_to_predict.csv")
