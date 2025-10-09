"""
Standalone predictor that loads the trained ML model and makes predictions
Simple interface for GPU/CPU offloading decisions
"""

import pickle
import numpy as np


class DevicePredictor:
    """Simple class to load model and make predictions"""
    
    def __init__(self, model_path='gpu_offloading_model_best.pkl'):
        """
        Initialize predictor by loading the trained model
        
        Args:
            model_path: Path to the saved .pkl model file
        """
        print(f"🔧 Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.task_encoder = model_data['task_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        
        print(f"✅ Loaded {self.model_type} model successfully!")
        print(f"   Features used: {len(self.feature_names)}")
        print(f"   Model ready for predictions\n")
    
    def predict(self, task_name, size, num_threads=4):
        """
        Predict optimal device for a computational task
        
        Args:
            task_name: 'VectorAddition', 'MatrixMultiplication', or 'ParallelReduction'
            size: Problem size (vector length or matrix dimension)
            num_threads: Number of CPU threads available (default: 4)
            
        Returns:
            device: 'CPU' or 'GPU'
            confidence: Prediction confidence (0.0 to 1.0)
        """
        # Encode task name
        task_encoded = self.task_encoder.transform([task_name])[0]
        
        # Calculate derived features
        size_log = np.log10(size + 1)
        threads_log = np.log10(num_threads)
        
        # Size category: 0=small, 1=medium, 2=large
        if size <= 1000:
            size_category = 0
        elif size <= 100000:
            size_category = 1
        else:
            size_category = 2
        
        # Create feature vector
        # Note: Last 3 features (time_ratio_log, cpu_cv, gpu_cv) use dummy values
        # since we don't have actual timing data during inference
        features = np.array([[
            size,              # Raw size
            size_log,          # Log of size
            size_category,     # Size category
            num_threads,       # Thread count
            threads_log,       # Log of threads
            task_encoded,      # Task type (0/1/2)
            0.0,               # time_ratio_log (placeholder)
            0.1,               # cpu_cv (placeholder)
            0.1                # gpu_cv (placeholder)
        ]])
        
        # Scale features (model expects scaled input)
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Convert to device name
        device = 'GPU' if prediction == 1 else 'CPU'
        confidence = probabilities[prediction]
        
        return device, confidence
    
    def predict_batch(self, tasks):
        """
        Predict for multiple tasks at once
        
        Args:
            tasks: List of tuples [(task_name, size, num_threads), ...]
            
        Returns:
            results: List of tuples [(device, confidence), ...]
        """
        results = []
        for task_name, size, num_threads in tasks:
            device, confidence = self.predict(task_name, size, num_threads)
            results.append((device, confidence))
        return results
    
    def explain_prediction(self, task_name, size, num_threads=4):
        """
        Make a prediction and explain the reasoning
        
        Args:
            task_name: Task type
            size: Problem size
            num_threads: CPU threads
        """
        device, confidence = self.predict(task_name, size, num_threads)
        
        print(f"\n{'='*60}")
        print(f" PREDICTION EXPLANATION")
        print(f"{'='*60}")
        print(f"\n📋 Input:")
        print(f"   Task: {task_name}")
        print(f"   Size: {size:,}")
        print(f"   CPU Threads: {num_threads}")
        
        print(f"\n🎯 Prediction:")
        print(f"   Recommended Device: {device}")
        print(f"   Confidence: {confidence:.1%}")
        
        # Confidence interpretation
        if confidence >= 0.9:
            print(f"   ✓ Very high confidence - strong recommendation")
        elif confidence >= 0.75:
            print(f"   ✓ High confidence - reliable recommendation")
        elif confidence >= 0.6:
            print(f"   ⚠️  Moderate confidence - borderline case")
        else:
            print(f"   ⚠️  Low confidence - performance may be similar")
        
        # Task-specific insights
        print(f"\n💡 Reasoning:")
        if task_name == 'MatrixMultiplication':
            if size < 128:
                print(f"   • Small matrix - GPU overhead not worth it")
            elif size >= 512:
                print(f"   • Large matrix - GPU excels at matrix operations (O(n³))")
            else:
                print(f"   • Medium matrix - borderline case")
        elif task_name == 'VectorAddition':
            if size < 10000:
                print(f"   • Small vector - CPU overhead lower")
            else:
                print(f"   • Large vector - GPU memory bandwidth advantage")
        elif task_name == 'ParallelReduction':
            if size < 10000:
                print(f"   • Small reduction - CPU more efficient")
            else:
                print(f"   • Large reduction - GPU parallel advantage")
        
        print(f"{'='*60}\n")
        
        return device, confidence


def demo_predictions():
    """Demo function showing various prediction examples"""
    
    print("="*70)
    print(" GPU OFFLOADING PREDICTOR - DEMO")
    print("="*70)
    
    # Initialize predictor
    predictor = DevicePredictor('gpu_offloading_model_best.pkl')
    
    # Test cases covering different scenarios
    test_cases = [
        # Vector Addition tests
        ('VectorAddition', 500, 1, 'Very small vector'),
        ('VectorAddition', 10000, 4, 'Small vector'),
        ('VectorAddition', 100000, 8, 'Medium vector'),
        ('VectorAddition', 1000000, 8, 'Large vector'),
        ('VectorAddition', 50000000, 16, 'Very large vector'),
        
        # Matrix Multiplication tests
        ('MatrixMultiplication', 32, 2, 'Tiny matrix (32x32)'),
        ('MatrixMultiplication', 64, 4, 'Small matrix (64x64)'),
        ('MatrixMultiplication', 128, 4, 'Medium-small matrix (128x128)'),
        ('MatrixMultiplication', 256, 8, 'Medium matrix (256x256)'),
        ('MatrixMultiplication', 512, 8, 'Large matrix (512x512)'),
        ('MatrixMultiplication', 1024, 16, 'Very large matrix (1024x1024)'),
        
        # Parallel Reduction tests
        ('ParallelReduction', 1000, 1, 'Small reduction'),
        ('ParallelReduction', 10000, 2, 'Medium reduction'),
        ('ParallelReduction', 100000, 4, 'Large reduction'),
        ('ParallelReduction', 1000000, 8, 'Very large reduction'),
        ('ParallelReduction', 10000000, 16, 'Massive reduction'),
    ]
    
    print("\n📊 PREDICTIONS:\n")
    
    for task, size, threads, description in test_cases:
        device, confidence = predictor.predict(task, size, threads)
        
        # Color coding based on device
        device_icon = "🟢" if device == "GPU" else "🔵"
        conf_bar = "█" * int(confidence * 10)
        
        print(f"{device_icon} {task:<25} | Size: {size:>12,} | Threads: {threads:>2} | "
              f"→ {device:<3} ({confidence:>5.1%}) {conf_bar}")
    
    # Show detailed explanation for a few cases
    print("\n" + "="*70)
    print(" DETAILED EXPLANATIONS")
    print("="*70)
    
    # Case 1: Small matrix (should be CPU)
    predictor.explain_prediction('MatrixMultiplication', 64, 4)
    
    # Case 2: Large matrix (should be GPU)
    predictor.explain_prediction('MatrixMultiplication', 1024, 16)
    
    # Case 3: Large vector (should be GPU)
    predictor.explain_prediction('VectorAddition', 1000000, 8)


def interactive_mode():
    """Interactive mode for custom predictions"""
    
    print("\n" + "="*70)
    print(" INTERACTIVE PREDICTION MODE")
    print("="*70)
    
    predictor = DevicePredictor('gpu_offloading_model_best.pkl')
    
    print("\nAvailable tasks:")
    print("  1. VectorAddition")
    print("  2. MatrixMultiplication")
    print("  3. ParallelReduction")
    
    while True:
        print("\n" + "-"*70)
        
        task_input = input("\nSelect task (1-3) or 'q' to quit: ").strip()
        
        if task_input.lower() == 'q':
            print("\n👋 Goodbye!\n")
            break
        
        if task_input not in ['1', '2', '3']:
            print("❌ Invalid choice! Please enter 1, 2, 3, or 'q'")
            continue
        
        task_map = {
            '1': 'VectorAddition',
            '2': 'MatrixMultiplication',
            '3': 'ParallelReduction'
        }
        task_name = task_map[task_input]
        
        try:
            size = int(input("Enter problem size: "))
            threads = int(input("Enter number of CPU threads (default 4): ") or "4")
        except ValueError:
            print("❌ Invalid input! Please enter numbers.")
            continue
        
        # Make prediction with explanation
        predictor.explain_prediction(task_name, size, threads)


def quick_predict(task_name, size, num_threads=4):
    """
    Quick prediction function for external use
    
    Usage:
        from predict_device import quick_predict
        device, conf = quick_predict('VectorAddition', 1000000, 8)
        print(f"Use {device}")
    """
    predictor = DevicePredictor('gpu_offloading_model_best.pkl')
    return predictor.predict(task_name, size, num_threads)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'interactive':
            interactive_mode()
        elif mode == 'demo':
            demo_predictions()
        elif mode == 'predict' and len(sys.argv) >= 4:
            # Direct prediction: python predict_device.py predict VectorAddition 1000000 8
            task = sys.argv[2]
            size = int(sys.argv[3])
            threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4
            
            device, conf = quick_predict(task, size, threads)
            print(f"\n✓ Recommendation: {device} (confidence: {conf:.1%})")
        else:
            print("Usage:")
            print("  python predict_device.py demo        - Run demo predictions")
            print("  python predict_device.py interactive - Interactive mode")
            print("  python predict_device.py predict TaskName Size Threads")
    else:
        # Default: run demo
        demo_predictions()
