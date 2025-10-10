"""
Intelligent Task Scheduler with ML-Based GPU Offloading
Automatically predicts and schedules tasks to CPU or GPU using XGBoost model
"""

import queue
import threading
import time
import uuid
import random
import pickle
import numpy as np
import cupy as cp
from enum import Enum, auto
from dataclasses import dataclass, field


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

class TaskType(Enum):
    """Types of computational tasks"""
    VECTOR_ADD = auto()
    MATRIX_MUL = auto()
    PARALLEL_REDUCTION = auto()


class ExecutionPolicy(Enum):
    """Scheduling policy"""
    PRIORITY = auto()
    ROUND_ROBIN = auto()


@dataclass
class Task:
    """Task data structure"""
    priority: int
    task_type: TaskType
    size: int
    data: dict
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submit_time: float = field(default_factory=time.time)

    def __lt__(self, other):
        if self.priority == other.priority:
            return self.submit_time < other.submit_time
        return self.priority < other.priority


# ============================================================================
# COMPUTE KERNELS (CPU and GPU implementations)
# ============================================================================

class Kernels:
    """CPU and GPU compute functions"""

    @staticmethod
    def cpu_vector_add(a, b):
        """Vector addition on CPU"""
        return np.add(a, b)

    @staticmethod
    def gpu_vector_add(a, b):

        """
        Perform element-wise vector addition on the GPU using CuPy.
        Args:
            a (cupy.ndarray): First input vector.
            b (cupy.ndarray): Second input vector.
        Returns:
            cupy.ndarray: Resultant vector.
        """
        # Ensure inputs are on the GPU
        a_gpu = cp.array(a)
        b_gpu = cp.array(b)

        # Perform addition on the GPU
        result_gpu = a_gpu + b_gpu

        return result_gpu


    @staticmethod
    def cpu_matrix_mul(a, b):
        """Matrix multiplication on CPU"""
        return np.dot(a, b)

    @staticmethod
    def gpu_matrix_mul(a, b):
   
        """
        Perform matrix multiplication on the GPU using CuPy.
        Args:
            a (cupy.ndarray): First input matrix.
            b (cupy.ndarray): Second input matrix.
        Returns:
            cupy.ndarray: Resultant matrix.
        """
        # Ensure inputs are on the GPU
        a_gpu = cp.array(a)
        b_gpu = cp.array(b)

        # Perform matrix multiplication on the GPU
        result_gpu = cp.dot(a_gpu, b_gpu)

        return result_gpu
    @staticmethod
    def cpu_parallel_reduction(arr):
        """Parallel reduction (sum) on CPU"""
        return np.sum(arr)

    @staticmethod
    def gpu_parallel_reduction(arr):
 
        """
        Perform parallel reduction (sum) on the GPU using CuPy.
        Args:
            arr (cupy.ndarray): Input array.
        Returns:
            float: Sum of all elements in the array.
        """
        # Ensure input is on the GPU
        arr_gpu = cp.array(arr)

        # Perform reduction (sum) on the GPU
        result_gpu = cp.sum(arr_gpu)

        return result_gpu



# ============================================================================
# ML MODEL INTEGRATION
# ============================================================================

class MLPredictor:
    """Load and use trained XGBoost model for device prediction"""
    
    def __init__(self, model_path='/mnt/d/OSprojects/GPU-Offloading-Task-Scheduler/XG_boost_Model/xgboost_model.pkl'):
        """Load the trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.task_encoder = model_data['task_encoder']
            self.loaded = True
            print(f"✓ ML Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"⚠️  Model not found: {model_path}")
            print("   Falling back to heuristic-based scheduling")
            self.loaded = False
    
    def predict_device(self, task_name, size, num_threads=4):
        """
        Predict optimal device using ML model
        
        Args:
            task_name: 'VectorAddition', 'MatrixMultiplication', or 'ParallelReduction'
            size: Problem size
            num_threads: Number of CPU threads (default: 4)
        
        Returns:
            device: 'CPU' or 'GPU'
            confidence: Prediction confidence (0-1)
        """
        if not self.loaded:
            # Fallback heuristic if model not loaded
            return self._heuristic_fallback(task_name, size), 0.5
        
        try:
            # Encode task name to number
            task_encoded = self.task_encoder.transform([task_name])[0]
            
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
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            device = 'GPU' if prediction == 1 else 'CPU'
            confidence = probabilities[prediction]
            
            return device, confidence
            
        except Exception as e:
            print(f"⚠️  ML Prediction error: {e}")
            return self._heuristic_fallback(task_name, size), 0.5
    
    def _heuristic_fallback(self, task_name, size):
        """Simple heuristic if ML model fails"""
        if task_name == 'MatrixMultiplication' and size < 128:
            return 'CPU'
        elif task_name == 'VectorAddition' and size < 1000:
            return 'CPU'
        else:
            return 'GPU'


# ============================================================================
# INTELLIGENT TASK SCHEDULER
# ============================================================================

class IntelligentTaskScheduler:
    """Task scheduler with ML-based device selection"""
    
    def __init__(self, policy: ExecutionPolicy = ExecutionPolicy.PRIORITY):
        """Initialize scheduler with ML model"""
        self.policy_type = policy
        
        # Initialize queues
        if self.policy_type == ExecutionPolicy.PRIORITY:
            self.task_queue = queue.PriorityQueue()
            print("✓ Scheduler initialized with PRIORITY policy")
        else:
            self.task_queue = queue.Queue()
            print("✓ Scheduler initialized with ROUND_ROBIN policy")
        
        self.cpu_queue = queue.Queue()
        self.gpu_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Load ML model for intelligent scheduling
        self.ml_predictor = MLPredictor()
        
        # Statistics tracking
        self.stats = {
            'total_tasks': 0,
            'cpu_tasks': 0,
            'gpu_tasks': 0,
            'cpu_time': 0.0,
            'gpu_time': 0.0
        }
        
        # Worker threads
        self.dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.cpu_worker_thread = threading.Thread(target=self._cpu_worker, daemon=True)
        self.gpu_worker_thread = threading.Thread(target=self._gpu_worker, daemon=True)
    
    def start_workers(self):
        """Start all worker threads"""
        print("✓ Starting Dispatcher, CPU Worker, and GPU Worker threads...")
        self.dispatcher_thread.start()
        self.cpu_worker_thread.start()
        self.gpu_worker_thread.start()
    
    def submit_task(self, task: Task):
        """Submit a task to the scheduler"""
        self.stats['total_tasks'] += 1
        print(f"[SUBMIT] Task {task.task_id[:6]} | Type: {task.task_type.name} | Size: {task.size:,} | Priority: {task.priority}")
        self.task_queue.put(task)
    
    def _decide_device(self, task: Task):
        """Use ML model to decide CPU or GPU"""
        # Map TaskType enum to task names
        task_name_map = {
            TaskType.VECTOR_ADD: 'VectorAddition',
            TaskType.MATRIX_MUL: 'MatrixMultiplication',
            TaskType.PARALLEL_REDUCTION: 'ParallelReduction'
        }
        
        task_name = task_name_map.get(task.task_type, 'VectorAddition')
        
        # Get ML prediction
        device, confidence = self.ml_predictor.predict_device(task_name, task.size)
        
        print(f"[ML-PREDICT] Task {task.task_id[:6]} → {device} (confidence: {confidence:.1%})")
        
        return device
    
    def _dispatch_loop(self):
        """Dispatcher thread: routes tasks to CPU or GPU queue"""
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
                if task is None:
                    continue
                
                # Use ML model to decide device
                destination = self._decide_device(task)
                
                if destination == "GPU":
                    self.gpu_queue.put(task)
                    self.stats['gpu_tasks'] += 1
                else:
                    self.cpu_queue.put(task)
                    self.stats['cpu_tasks'] += 1
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[DISPATCH-ERROR] {e}")
    
    def _cpu_worker(self):
        """CPU worker thread"""
        while not self.stop_event.is_set():
            try:
                task = self.cpu_queue.get(timeout=0.1)
                if task is None:
                    continue
                
                start_time = time.perf_counter()
                
                # Execute task on CPU
                if task.task_type == TaskType.VECTOR_ADD:
                    result = Kernels.cpu_vector_add(task.data['a'], task.data['b'])
                    
                elif task.task_type == TaskType.MATRIX_MUL:
                    result = Kernels.cpu_matrix_mul(task.data['a'], task.data['b'])
                    
                elif task.task_type == TaskType.PARALLEL_REDUCTION:
                    result = Kernels.cpu_parallel_reduction(task.data['arr'])
                
                duration = (time.perf_counter() - start_time) * 1000
                self.stats['cpu_time'] += duration
                
                print(f"  [CPU] ✓ Task {task.task_id[:6]} completed in {duration:.2f} ms")
                
                self.cpu_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CPU-ERROR] {e}")
                self.cpu_queue.task_done()
    
    def _gpu_worker(self):
        """GPU worker thread"""
        while not self.stop_event.is_set():
            try:
                task = self.gpu_queue.get(timeout=0.1)
                if task is None:
                    continue
                
                start_time = time.perf_counter()
                
                # Execute task on GPU
                if task.task_type == TaskType.VECTOR_ADD:
                    a_gpu = cp.asarray(task.data['a'])
                    b_gpu = cp.asarray(task.data['b'])
                    result = Kernels.gpu_vector_add(a_gpu, b_gpu)
                    
                elif task.task_type == TaskType.MATRIX_MUL:
                    a_gpu = cp.asarray(task.data['a'])
                    b_gpu = cp.asarray(task.data['b'])
                    result = Kernels.gpu_matrix_mul(a_gpu, b_gpu)
                    
                elif task.task_type == TaskType.PARALLEL_REDUCTION:
                    arr_gpu = cp.asarray(task.data['arr'])
                    result = Kernels.gpu_parallel_reduction(arr_gpu)
                
                cp.cuda.Stream.null.synchronize()
                duration = (time.perf_counter() - start_time) * 1000
                self.stats['gpu_time'] += duration
                
                print(f"  [GPU] ✓ Task {task.task_id[:6]} completed in {duration:.2f} ms")
                
                self.gpu_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[GPU-ERROR] {e} - Falling back to CPU")
                # Re-queue to CPU on GPU error
                self.cpu_queue.put(task)
                self.gpu_queue.task_done()
    
    def shutdown(self):
        """Gracefully shutdown the scheduler"""
        print("\n" + "="*70)
        print("Shutting down scheduler...")
        
        # Wait for all queues to empty
        self.task_queue.join()
        self.cpu_queue.join()
        self.gpu_queue.join()
        
        # Stop worker threads
        self.stop_event.set()
        self.task_queue.put(None)
        self.cpu_queue.put(None)
        self.gpu_queue.put(None)
        
        # Print statistics
        print("\n" + "="*70)
        print("EXECUTION STATISTICS")
        print("="*70)
        print(f"Total Tasks:      {self.stats['total_tasks']}")
        print(f"CPU Tasks:        {self.stats['cpu_tasks']} ({self.stats['cpu_tasks']/max(self.stats['total_tasks'],1)*100:.1f}%)")
        print(f"GPU Tasks:        {self.stats['gpu_tasks']} ({self.stats['gpu_tasks']/max(self.stats['total_tasks'],1)*100:.1f}%)")
        print(f"Total CPU Time:   {self.stats['cpu_time']:.2f} ms")
        print(f"Total GPU Time:   {self.stats['gpu_time']:.2f} ms")
        print(f"Avg CPU Time:     {self.stats['cpu_time']/max(self.stats['cpu_tasks'],1):.2f} ms per task")
        print(f"Avg GPU Time:     {self.stats['gpu_time']/max(self.stats['gpu_tasks'],1):.2f} ms per task")
        print("="*70)
        print("✓ Shutdown complete!")


# ============================================================================
# WORKLOAD GENERATOR (FOR TESTING)
# ============================================================================

def generate_workload(scheduler, num_tasks=20):
    """Generate random tasks to test the scheduler"""
    print("\n" + "="*70)
    print(f"GENERATING WORKLOAD: {num_tasks} tasks")
    print("="*70 + "\n")
    
    for i in range(num_tasks):
        priority = random.randint(1, 10)
        
        # Randomly choose task type
        task_type = random.choice([TaskType.VECTOR_ADD, TaskType.MATRIX_MUL, TaskType.PARALLEL_REDUCTION])
        
        # Generate task based on type
        if task_type == TaskType.VECTOR_ADD:
            size = random.choice([500, 1000, 10000, 100000, 1000000])
            a = np.random.rand(size).astype(np.float32)
            b = np.random.rand(size).astype(np.float32)
            data = {'a': a, 'b': b}
            
        elif task_type == TaskType.MATRIX_MUL:
            size = random.choice([32, 64, 128, 256, 512])
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)
            data = {'a': a, 'b': b}
            
        elif task_type == TaskType.PARALLEL_REDUCTION:
            size = random.choice([1000, 10000, 100000, 1000000, 10000000])
            arr = np.random.rand(size).astype(np.float32)
            data = {'arr': arr}
        
        # Create and submit task
        task = Task(
            priority=priority,
            task_type=task_type,
            size=size,
            data=data
        )
        
        scheduler.submit_task(task)
        time.sleep(random.uniform(0.05, 0.15))  # Simulate task arrival rate
    
    print("\n✓ All tasks submitted!\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("INTELLIGENT TASK SCHEDULER WITH ML-BASED GPU OFFLOADING")
    print("="*70 + "\n")
    
    # Create scheduler with ML-based decision making
    scheduler = IntelligentTaskScheduler(policy=ExecutionPolicy.PRIORITY)
    
    # Start worker threads
    scheduler.start_workers()
    
    # Generate and submit tasks
    generate_workload(scheduler, num_tasks=15)
    
    # Wait a bit for tasks to complete
    time.sleep(2)
    
    # Shutdown and show statistics
    scheduler.shutdown()
