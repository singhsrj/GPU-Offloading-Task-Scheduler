import queue
import threading
import time
import uuid
import random
import numpy as np
import cupy as cp
from enum import Enum, auto
from dataclasses import dataclass, field

# --- Phase 1: Task Definitions & Compute Kernels ---

class TaskType(Enum):
    """Defines the types of computational tasks we accept."""
    VECTOR_ADD = auto()
    MATRIX_MUL = auto()

class ExecutionPolicy(Enum):
    """Defines which scheduling policy the main queue should use."""
    ROUND_ROBIN = auto() # FIFO queue
    PRIORITY = auto()    # Priority queue

@dataclass
class Task:
    """
    Data structure for a single task. We make it orderable for the PriorityQueue.
    Priority: Lower number = higher priority.
    """
    priority: int
    task_type: TaskType
    size: int
    data: dict
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submit_time: float = field(default_factory=time.time)

    # Implement __lt__ (less-than) for the priority queue.
    # It compares tasks based on priority. If priorities are equal,
    # it falls back to the submission time (Earliest Task First).
    def __lt__(self, other):
        if self.priority == other.priority:
            return self.submit_time < other.submit_time
        return self.priority < other.priority

class Kernels:
    """
    Container for our CPU (NumPy) and GPU (CuPy) compute functions.
    This matches the operations defined in your attempt2.cu file.
   
    """

    @staticmethod
    def cpu_vector_add(a, b):
        """CPU Vector Add using NumPy."""
        return np.add(a, b)

    @staticmethod
    def gpu_vector_add(a_gpu, b_gpu):
        """GPU Vector Add using CuPy."""
        return cp.add(a_gpu, b_gpu)

    @staticmethod
    def cpu_matrix_mul(a, b):
        """CPU Matrix Multiply using NumPy."""
        return np.dot(a, b)

    @staticmethod
    def gpu_matrix_mul(a_gpu, b_gpu):
        """GPU Matrix Multiply using CuPy."""
        return cp.dot(a_gpu, b_gpu)


# --- Phases 1 & 2: Task Scheduler, Dispatcher, and Policy Logic ---

class TaskScheduler:
    """
    Implements the core scheduler, dispatcher, and worker threads.
    It manages the main task queue and dispatches tasks to dedicated
    CPU and GPU worker queues.
    """

    def __init__(self, policy: ExecutionPolicy = ExecutionPolicy.PRIORITY):
        self.policy_type = policy
        
        # Phase 2: Policy Switch implementation.
        # Select the queue type based on the chosen policy.
        if self.policy_type == ExecutionPolicy.PRIORITY:
            # PriorityQueue automatically handles task ordering based on the
            # Task object's __lt__ method (priority, then submission time).
            self.task_queue = queue.PriorityQueue()
            print("Scheduler initialized with PRIORITY policy.")
        else:
            # Standard FIFO queue for Round-Robin.
            self.task_queue = queue.Queue()
            print("Scheduler initialized with ROUND_ROBIN (FIFO) policy.")

        # Dedicated queues for workers
        self.cpu_queue = queue.Queue()
        self.gpu_queue = queue.Queue()
        self.stop_event = threading.Event()

        # Worker threads
        self.dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.cpu_worker_thread = threading.Thread(target=self._cpu_worker, daemon=True)
        self.gpu_worker_thread = threading.Thread(target=self._gpu_worker, daemon=True)

    def start_workers(self):
        """Starts all background threads."""
        print("Starting Dispatcher, CPU Worker, and GPU Worker threads...")
        self.dispatcher_thread.start()
        self.cpu_worker_thread.start()
        self.gpu_worker_thread.start()

    def submit_task(self, task: Task):
        """Public method for submitting a new task to the scheduler."""
        print(f"[QUEUE] Submitted Task {task.task_id[:6]} (Type: {task.task_type.name}, Prio: {task.priority}, Size: {task.size})")
        # The queue itself handles the policy (FIFO vs Priority).
        self.task_queue.put(task)

    def _decide_device(self, task: Task):
        """
        THIS IS THE CORE SCHEDULING LOGIC.
        
        Currently implements the simple heuristic policy from your attempt2.cu file.
       
        
        *** ROADMAP INTEGRATION: ***
        This method is where you will integrate your ML model from model_training.ipynb.
       
        
        You would:
        1. Collect current system stats (psutil, pynvml) per Phase 3.
        2. Create a feature vector (e.g., [task.size, cpu_load, gpu_util, ...]).
        3. Scale the features using your saved scaler.
        4. pred = self.ml_model.predict(features_scaled)
        5. if pred == 0: return "GPU" else: return "CPU"
        """
        
        # Using heuristic from attempt2.cu as placeholder
        if task.task_type == TaskType.VECTOR_ADD and task.size > 1000:
            return "GPU"
        if task.task_type == TaskType.MATRIX_MUL and task.size > 128: # N > 128
            return "GPU"
            
        return "CPU"

    def _dispatch_loop(self):
        """
        Phase 1: Dispatcher.
        Continuously pulls tasks from the main policy queue (Priority or RR)
        and dispatches them to the correct worker queue based on the decision logic.
        """
        while not self.stop_event.is_set():
            try:
                # Blocks until a task is available.
                # Gets the HIGHEST PRIORITY task (if PriorityQueue)
                # or NEXT task (if FIFO Queue).
                task = self.task_queue.get()
                if task is None:
                    continue

                # Phase 3/4 integration point:
                # Get system state (pynvml, psutil) BEFORE making decision.
                # gpu_status = self.monitoring.get_status()
                
                destination = self._decide_device(task) # Call policy
                
                if destination == "GPU":
                    # print(f"[DISPATCH] Task {task.task_id[:6]} -> GPU")
                    self.gpu_queue.put(task)
                else:
                    # print(f"[DISPATCH] Task {task.task_id[:6]} -> CPU")
                    self.cpu_queue.put(task)
                
                self.task_queue.task_done()

            except Exception as e:
                print(f"[DISPATCH-ERROR] {e}")
                time.sleep(0.1)

    def _cpu_worker(self):
        """
        Worker thread that processes tasks from the cpu_queue.
        """
        while not self.stop_event.is_set():
            try:
                task = self.cpu_queue.get()
                if task is None:
                    continue
                
                start_time = time.perf_counter()
                if task.task_type == TaskType.VECTOR_ADD:
                    Kernels.cpu_vector_add(task.data['a'], task.data['b'])
                elif task.task_type == TaskType.MATRIX_MUL:
                    Kernels.cpu_matrix_mul(task.data['a'], task.data['b'])
                
                duration = (time.perf_counter() - start_time) * 1000
                print(f"  [CPU-WORKER] ✅ Completed Task {task.task_id[:6]} (Prio: {task.priority}) in {duration:.4f} ms")
                self.cpu_queue.task_done()
                
            except Exception as e:
                print(f"[CPU-WORKER-ERROR] {e}")
                self.cpu_queue.task_done()

    def _gpu_worker(self):
        """
        Worker thread that processes tasks from the gpu_queue.
        Handles data transfer to/from the GPU.
        Includes basic Phase 4 Fallback Handling (try/except).
        """
        while not self.stop_event.is_set():
            try:
                task = self.gpu_queue.get()
                if task is None:
                    continue

                start_time = time.perf_counter()
                
                # 1. Transfer data from Host (RAM) to Device (VRAM)
                a_gpu = cp.asarray(task.data['a'])
                b_gpu = cp.asarray(task.data['b'])
                
                # 2. Execute kernel
                if task.task_type == TaskType.VECTOR_ADD:
                    result_gpu = Kernels.gpu_vector_add(a_gpu, b_gpu)
                elif task.task_type == TaskType.MATRIX_MUL:
                    result_gpu = Kernels.gpu_matrix_mul(a_gpu, b_gpu)
                
                # 3. Synchronize: Wait for kernel to finish and transfer result back
                cp.cuda.Stream.null.synchronize()
                result_host = cp.asnumpy(result_gpu) # Transfer Device -> Host

                duration = (time.perf_counter() - start_time) * 1000
                print(f"  [GPU-WORKER] 🔥 Completed Task {task.task_id[:6]} (Prio: {task.priority}) in {duration:.4f} ms")
                self.gpu_queue.task_done()

            except cp.cuda.runtime.CudaError as e:
                # --- Phase 4: Graceful Degradation / Fallback Handling ---
                print(f"  [GPU-WORKER-ERROR] ❌ CUDA Error on Task {task.task_id[:6]}: {e}. FALLING BACK TO CPU.")
                # Re-queue the failed task to the CPU queue instead of failing.
                self.cpu_queue.put(task) 
                self.gpu_queue.task_done()
            except Exception as e:
                print(f"[GPU-WORKER-ERROR] General Error: {e}")
                self.gpu_queue.task_done()


    def shutdown(self):
        """Waits for all queues to empty then stops worker threads."""
        print("\nShutting down scheduler...")
        # Wait for all submitted tasks to be processed
        self.task_queue.join()
        self.cpu_queue.join()
        self.gpu_queue.join()
        
        # Stop the worker loops
        self.stop_event.set()
        # Add dummy tasks to unblock workers waiting on .get()
        self.task_queue.put(None) 
        self.cpu_queue.put(None)
        self.gpu_queue.put(None)
        print("All tasks complete. Shutdown complete.")


# --- Phase 5: Workload Generator & Demo ---

def workload_generator(scheduler, num_tasks=20):
    """Submits a mix of random tasks to the scheduler."""
    print(f"--- Starting Workload Generator: Submitting {num_tasks} mixed tasks ---")
    for i in range(num_tasks):
        # Generate a random task
        priority = random.randint(1, 10) # 1 = highest prio
        task_choice = random.choice([TaskType.VECTOR_ADD, TaskType.MATRIX_MUL])
        
        if task_choice == TaskType.VECTOR_ADD:
            # Small tasks (heuristic picks CPU), Large tasks (heuristic picks GPU)
            size = random.choice([500, 50_000_000]) 
            a = np.random.rand(size).astype(np.float32)
            b = np.random.rand(size).astype(np.float32)
        
        else: # MATRIX_MUL
            # Small tasks (CPU), Large tasks (GPU)
            size = random.choice([50, 500]) # N (size x size matrix)
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)

        task = Task(
            priority=priority,
            task_type=task_choice,
            size=size,
            data={'a': a, 'b': b}
        )
        scheduler.submit_task(task)
        time.sleep(random.uniform(0.05, 0.2)) # Submit tasks intermittently

if __name__ == "__main__":
    
    # --- DEMO 1: PRIORITY POLICY ---
    # Tasks will be executed primarily by priority (1 is highest).
    # Note how high-priority tasks (like Prio: 1) submitted LATER 
    # jump ahead of low-priority tasks submitted EARLIER.
    
    # Initialize scheduler using the policy from Phase 2
    priority_scheduler = TaskScheduler(policy=ExecutionPolicy.PRIORITY)
    priority_scheduler.start_workers()
    workload_generator(priority_scheduler, num_tasks=10)
    
    # Wait for all tasks to finish before shutting down
    priority_scheduler.shutdown()

    print("\n" + "="*50 + "\n")
    
    # --- DEMO 2: ROUND ROBIN (FIFO) POLICY ---
    # Tasks will be executed strictly in the order they are submitted,
    # regardless of their priority value.
    
    rr_scheduler = TaskScheduler(policy=ExecutionPolicy.ROUND_ROBIN)
    rr_scheduler.start_workers()
    workload_generator(rr_scheduler, num_tasks=10)
    rr_scheduler.shutdown()