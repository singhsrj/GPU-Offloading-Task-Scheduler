import queue
import threading
import time
import uuid
import random
import numpy as np
import cupy as cp
import pandas as pd
from enum import Enum, auto
from dataclasses import dataclass, field

# --- Phase 1: Task Definitions & Compute Kernels ---

class TaskType(Enum):
    """Defines the types of computational tasks we accept."""
    VECTOR_ADD = auto()
    MATRIX_MUL = auto()
    NEURAL_NETWORK_TRAINING = auto() # New task for NN training

class ExecutionPolicy(Enum):
    """Defines which scheduling policy the main queue should use."""
    ROUND_ROBIN = auto()
    PRIORITY = auto()

@dataclass
class Task:
    """Data structure for a single task, orderable for the PriorityQueue."""
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

class Kernels:
    """Container for our CPU (NumPy) and GPU (CuPy) compute functions."""

    @staticmethod
    def cpu_vector_add(a, b): return np.add(a, b)
    @staticmethod
    def gpu_vector_add(a_gpu, b_gpu): return cp.add(a_gpu, b_gpu)
    @staticmethod
    def cpu_matrix_mul(a, b): return np.dot(a, b)
    @staticmethod
    def gpu_matrix_mul(a_gpu, b_gpu): return cp.dot(a_gpu, b_gpu)

    @staticmethod
    def train_neural_network(X_train, Y_train, iterations, alpha=0.1, h1=256, h2=128):
        """
        Encapsulated neural network training function.
        It intelligently uses NumPy or CuPy based on the type of the input arrays.
        """
        xp = cp.get_array_module(X_train) # Use CuPy if data is on GPU, else NumPy

        # --- Helper Functions (using xp for dual CPU/GPU compatibility) ---
        def he_initialization(fan_in, num_neurons):
            std = xp.sqrt(2.0 / fan_in)
            W = xp.random.randn(num_neurons, fan_in) * std
            b = xp.zeros((num_neurons, 1))
            return W, b

        def ReLU(Z): return xp.maximum(0, Z)
        def derivative_ReLU(Z): return (Z > 0).astype(xp.float32)
        def softplus(Z): return xp.log1p(xp.exp(Z))
        def derivative_softplus(Z): return 1 / (1 + xp.exp(-Z))
        def softmax(Z):
            shift = Z - xp.max(Z, axis=0, keepdims=True)
            exp_Z = xp.exp(shift)
            return exp_Z / xp.sum(exp_Z, axis=0, keepdims=True)
        
        def one_hot(Y, num_classes=10):
            m = Y.size
            Y_oh = xp.zeros((num_classes, m))
            Y_oh[Y, xp.arange(m)] = 1
            return Y_oh

        # --- Main Training Logic ---
        n_x, n_y = X_train.shape[0], 10
        W1, b1 = he_initialization(n_x, h1)
        W2, b2 = he_initialization(h1, h2)
        W3, b3 = he_initialization(h2, n_y)

        for i in range(iterations):
            # Forward propagation
            Z1 = W1.dot(X_train) + b1
            A1 = ReLU(Z1)
            Z2 = W2.dot(A1) + b2
            A2 = softplus(Z2)
            Z3 = W3.dot(A2) + b3
            A3 = softmax(Z3)

            # Backward propagation
            m = Y_train.size
            Y_oh = one_hot(Y_train)
            dZ3 = A3 - Y_oh
            dW3 = (1/m) * dZ3.dot(A2.T)
            db3 = (1/m) * xp.sum(dZ3, axis=1, keepdims=True)
            dZ2 = W3.T.dot(dZ3) * derivative_softplus(Z2)
            dW2 = (1/m) * dZ2.dot(A1.T)
            db2 = (1/m) * xp.sum(dZ2, axis=1, keepdims=True)
            dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
            dW1 = (1/m) * dZ1.dot(X_train.T)
            db1 = (1/m) * xp.sum(dZ1, axis=1, keepdims=True)

            # Update parameters
            W1 -= alpha * dW1; b1 -= alpha * db1
            W2 -= alpha * dW2; b2 -= alpha * db2
            W3 -= alpha * dW3; b3 -= alpha * db3

            if i % 100 == 0: alpha *= 0.9

        return W1, b1, W2, b2, W3, b3 # Return trained weights

# --- Phases 1 & 2: Task Scheduler, Dispatcher, and Policy Logic ---

class TaskScheduler:
    def __init__(self, policy: ExecutionPolicy = ExecutionPolicy.PRIORITY):
        self.policy_type = policy
        if self.policy_type == ExecutionPolicy.PRIORITY:
            self.task_queue = queue.PriorityQueue()
            print("Scheduler initialized with PRIORITY policy.")
        else:
            self.task_queue = queue.Queue()
            print("Scheduler initialized with ROUND_ROBIN (FIFO) policy.")

        self.cpu_queue = queue.Queue()
        self.gpu_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.cpu_worker_thread = threading.Thread(target=self._cpu_worker, daemon=True)
        self.gpu_worker_thread = threading.Thread(target=self._gpu_worker, daemon=True)

    def start_workers(self):
        print("Starting Dispatcher, CPU Worker, and GPU Worker threads...")
        self.dispatcher_thread.start()
        self.cpu_worker_thread.start()
        self.gpu_worker_thread.start()

    def submit_task(self, task: Task):
        print(f"[QUEUE] Submitted Task {task.task_id[:6]} (Type: {task.task_type.name}, Prio: {task.priority}, Size: {task.size})")
        self.task_queue.put(task)

    def _decide_device(self, task: Task):
        # Heuristic: NN training is almost always better on GPU
        if task.task_type == TaskType.NEURAL_NETWORK_TRAINING:
            return "GPU"
        if task.task_type == TaskType.VECTOR_ADD and task.size > 1000:
            return "GPU"
        if task.task_type == TaskType.MATRIX_MUL and task.size > 128:
            return "GPU"
        return "CPU"

    def _dispatch_loop(self):
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get()
                if task is None: continue
                destination = self._decide_device(task)
                if destination == "GPU": self.gpu_queue.put(task)
                else: self.cpu_queue.put(task)
                self.task_queue.task_done()
            except Exception as e: print(f"[DISPATCH-ERROR] {e}")

    def _cpu_worker(self):
        while not self.stop_event.is_set():
            try:
                task = self.cpu_queue.get()
                if task is None: continue
                
                start_time = time.perf_counter()
                if task.task_type == TaskType.VECTOR_ADD:
                    Kernels.cpu_vector_add(task.data['a'], task.data['b'])
                elif task.task_type == TaskType.MATRIX_MUL:
                    Kernels.cpu_matrix_mul(task.data['a'], task.data['b'])
                elif task.task_type == TaskType.NEURAL_NETWORK_TRAINING:
                    Kernels.train_neural_network(task.data['X_train'], task.data['Y_train'], task.data['iterations'])
                
                duration = (time.perf_counter() - start_time) * 1000
                print(f"  [CPU-WORKER] Completed Task {task.task_id[:6]} (Prio: {task.priority}) in {duration:.4f} ms")
                self.cpu_queue.task_done()
            except Exception as e:
                print(f"[CPU-WORKER-ERROR] {e}")
                self.cpu_queue.task_done()

    def _gpu_worker(self):
        while not self.stop_event.is_set():
            try:
                task = self.gpu_queue.get()
                if task is None: continue
                start_time = time.perf_counter()

                if task.task_type == TaskType.NEURAL_NETWORK_TRAINING:
                    # Specific data handling for NN training
                    X_train_gpu = cp.asarray(task.data['X_train'])
                    Y_train_gpu = cp.asarray(task.data['Y_train'])
                    Kernels.train_neural_network(X_train_gpu, Y_train_gpu, task.data['iterations'])
                else:
                    # Standard data handling for other tasks
                    a_gpu = cp.asarray(task.data['a'])
                    b_gpu = cp.asarray(task.data['b'])
                    if task.task_type == TaskType.VECTOR_ADD: Kernels.gpu_vector_add(a_gpu, b_gpu)
                    elif task.task_type == TaskType.MATRIX_MUL: Kernels.gpu_matrix_mul(a_gpu, b_gpu)
                
                cp.cuda.Stream.null.synchronize()
                duration = (time.perf_counter() - start_time) * 1000
                print(f"  [GPU-WORKER] Completed Task {task.task_id[:6]} (Prio: {task.priority}) in {duration:.4f} ms")
                self.gpu_queue.task_done()
            except Exception as e:
                print(f"[GPU-WORKER-ERROR] Error on Task {task.task_id[:6]}. Re-queueing to CPU. Error: {e}")
                self.cpu_queue.put(task)
                self.gpu_queue.task_done()

    def shutdown(self):
        print("\nShutting down scheduler...")
        self.task_queue.join(); self.cpu_queue.join(); self.gpu_queue.join()
        self.stop_event.set()
        self.task_queue.put(None); self.cpu_queue.put(None); self.gpu_queue.put(None)
        print("All tasks complete. Shutdown complete.")

# --- Phase 5: Workload Generator & Demo ---

def workload_generator(scheduler, num_tasks=10):
    print(f"--- Starting Workload Generator: Submitting {num_tasks} mixed tasks ---")
    
    # Pre-load NN training data once
    try:
        train_df = pd.read_csv('C:/Users/suraj/Desktop/OS PROJECT/GPU-Offloading-Task-Scheduler/Task Scheduler Prototype/data_csv/test.csv')
        data = train_df.values
        np.random.shuffle(data)
        X = data[:, 1:].T / 255.0
        Y = data[:, 0].astype(int)
        nn_data_loaded = True
    except FileNotFoundError:
        print("WARNING: 'data_csv/train.csv' not found. Skipping NN tasks.")
        nn_data_loaded = False

    for _ in range(num_tasks):
        priority = random.randint(1, 10)
        
        # Add NN training to the mix of possible tasks
        task_choices = [TaskType.VECTOR_ADD, TaskType.MATRIX_MUL]
        if nn_data_loaded:
            task_choices.append(TaskType.NEURAL_NETWORK_TRAINING)
        
        task_choice = random.choice(task_choices)
        
        if task_choice == TaskType.NEURAL_NETWORK_TRAINING:
            iterations = random.randint(90, 400)
            task = Task(priority=priority, task_type=task_choice, size=iterations,
                        data={'X_train': X, 'Y_train': Y, 'iterations': iterations})
        else:
            size = random.choice([50, 500]) if task_choice == TaskType.MATRIX_MUL else random.choice([500, 500000])
            a = np.random.rand(size, size).astype(np.float32) if task_choice == TaskType.MATRIX_MUL else np.random.rand(size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32) if task_choice == TaskType.MATRIX_MUL else np.random.rand(size).astype(np.float32)
            task = Task(priority=priority, task_type=task_choice, size=size, data={'a': a, 'b': b})
            
        scheduler.submit_task(task)
        time.sleep(random.uniform(0.1, 0.3))

if __name__ == "__main__":
    # Ensure you have a 'data_csv/train.csv' file for the demo to run fully.
    
    scheduler = TaskScheduler(policy=ExecutionPolicy.PRIORITY)
    scheduler.start_workers()
    workload_generator(scheduler, num_tasks=15)
    scheduler.shutdown()