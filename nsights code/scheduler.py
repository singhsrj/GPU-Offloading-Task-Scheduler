"""
GPU/CPU Task Scheduler with ML-based Offloading Decision
Architecture and Implementation Guide
"""

import numpy as np
import pynvml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import time

class TaskScheduler:
    """
    Intelligent task scheduler that decides CPU/GPU offloading
    based on runtime NVML metrics and trained ML model
    """
    
    def __init__(self, model_path=None):
        # Initialize NVML
        pynvml.nvmlInit()
        self.device = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Load or initialize ML model
        if model_path:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = None
            
    def get_runtime_features(self, task_size):
        """
        Collect NVML runtime features for inference
        
        Args:
            task_size: Size parameter for the task (e.g., array size)
            
        Returns:
            Feature vector for ML model
        """
        # GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(self.device)
        gpu_util = util.gpu
        mem_util = util.memory
        
        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device)
        mem_free = mem_info.free / (1024**3)  # GB
        mem_total = mem_info.total / (1024**3)
        
        # Temperature and power
        temp = pynvml.nvmlDeviceGetTemperature(self.device, 
                                               pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(self.device) / 1000  # Watts
        
        # Clock speeds
        graphics_clock = pynvml.nvmlDeviceGetClockInfo(self.device, 
                                                       pynvml.NVML_CLOCK_GRAPHICS)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(self.device, 
                                                  pynvml.NVML_CLOCK_MEM)
        
        features = np.array([
            task_size,
            gpu_util,
            mem_util,
            mem_free,
            mem_total,
            temp,
            power,
            graphics_clock,
            mem_clock
        ])
        
        return features
    
    def predict_device(self, task_size):
        """
        Predict whether to offload to CPU or GPU
        
        Returns:
            'GPU' or 'CPU'
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
            
        features = self.get_runtime_features(task_size).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        
        return 'GPU' if prediction == 1 else 'CPU'
    
    def schedule_task(self, task_func, task_size, device=None):
        """
        Schedule and execute task on predicted or specified device
        
        Args:
            task_func: Function with signature func(size, device)
            task_size: Size parameter for task
            device: Force specific device, or None for ML prediction
            
        Returns:
            (result, execution_time, device_used)
        """
        if device is None:
            device = self.predict_device(task_size)
            
        start = time.time()
        result = task_func(task_size, device)
        exec_time = time.time() - start
        
        return result, exec_time, device
    
    def cleanup(self):
        pynvml.nvmlShutdown()


class ModelTrainer:
    """
    Train ML model using NSight profiling data
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def prepare_training_data(self, nsight_data_path):
        """
        Load and prepare NSight profiling data
        
        Expected CSV format:
        task_size, exec_time_gpu, exec_time_cpu, mem_bandwidth, 
        sm_occupancy, transfer_time, kernel_overhead, [other nsight metrics],
        optimal_device (0=CPU, 1=GPU)
        
        Note: You'll need to map NSight metrics to NVML-like features
        """
        # Load NSight data
        data = np.loadtxt(nsight_data_path, delimiter=',', skiprows=1)
        
        # Extract features (map NSight to NVML-compatible features)
        # This is the critical mapping step
        X = data[:, :-1]  # All columns except label
        y = data[:, -1]   # Label (CPU=0, GPU=1)
        
        return X, y
    
    def train(self, X, y):
        """Train the model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}")
        
        return train_score, test_score
    
    def save_model(self, path='scheduler_model.pkl'):
        """Save trained model"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")
        
    def feature_importance(self, feature_names):
        """Analyze which features are most important"""
        importances = self.model.feature_importances_
        
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.4f}")


# Example usage workflow
if __name__ == "__main__":
    # Phase 1: Training (offline, using NSight data)
    print("=== Training Phase ===")
    trainer = ModelTrainer()
    
    # Load NSight profiling data
    # X, y = trainer.prepare_training_data('nsight_profile_data.csv')
    # trainer.train(X, y)
    # trainer.save_model('scheduler_model.pkl')
    
    # Phase 2: Runtime Scheduling (online, using NVML)
    print("\n=== Runtime Scheduling Phase ===")
    scheduler = TaskScheduler(model_path='scheduler_model.pkl')
    
    # Example task scheduling
    # result, time, device = scheduler.schedule_task(
    #     task_func=vector_addition,
    #     task_size=1000000
    # )
    # print(f"Executed on {device} in {time:.4f}s")
    
    scheduler.cleanup()