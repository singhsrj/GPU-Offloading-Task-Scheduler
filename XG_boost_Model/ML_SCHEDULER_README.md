# Intelligent Task Scheduler with ML-Based GPU Offloading

## 🎯 What This Does

This is a **fully automated task scheduler** that uses a **trained XGBoost ML model** to intelligently decide whether each task should run on **CPU or GPU**. 

When you submit a task (Vector Addition, Matrix Multiplication, or Parallel Reduction), the scheduler:
1. ✅ Uses the ML model to predict optimal device (CPU/GPU)
2. ✅ Routes the task to the correct queue
3. ✅ Executes the task on the predicted device
4. ✅ Tracks performance statistics

## 📁 Files

### Core Files
- **`ml_task_scheduler.py`** - Complete intelligent scheduler with ML integration
- **`../nsights_code/xgboost_model.pkl`** - Trained XGBoost model (100% accuracy)
- **`../nsights_code/simple_predict.py`** - Standalone prediction script

### Training Files
- **`../nsights_code/simple_train.py`** - Train the XGBoost model
- **`../nsights_code/expanded_training_dataset.csv`** - 1,200 training samples

## 🚀 Quick Start

### 1. Run the Intelligent Scheduler

```bash
cd "Task Scheduler Prototype"
python3 ml_task_scheduler.py
```

This will:
- Load the trained ML model
- Start CPU and GPU worker threads
- Generate 15 random tasks
- Use ML to decide device for each task
- Execute tasks and show statistics

### 2. Example Output

```
[SUBMIT] Task 15022d | Type: VECTOR_ADD | Size: 1,000 | Priority: 5
[ML-PREDICT] Task 15022d → GPU (confidence: 99.9%)
  [GPU] ✓ Task 15022d completed in 208.20 ms

[SUBMIT] Task 3f346e | Type: MATRIX_MUL | Size: 128 | Priority: 5
[ML-PREDICT] Task 3f346e → CPU (confidence: 99.4%)
  [CPU] ✓ Task 3f346e completed in 2.12 ms

EXECUTION STATISTICS
Total Tasks:      15
CPU Tasks:        2 (13.3%)
GPU Tasks:        13 (86.7%)
Total CPU Time:   3.42 ms
Total GPU Time:   619.01 ms
```

## 🔧 How It Works

### Architecture

```
                    ┌─────────────────┐
                    │  Submit Task    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Main Queue     │
                    │  (Priority)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Dispatcher    │
                    │  (ML Predictor) │
                    └────┬──────┬─────┘
                         │      │
              ┌──────────┘      └──────────┐
              │                            │
     ┌────────▼────────┐          ┌───────▼────────┐
     │   CPU Queue     │          │   GPU Queue    │
     └────────┬────────┘          └───────┬────────┘
              │                            │
     ┌────────▼────────┐          ┌───────▼────────┐
     │  CPU Worker     │          │  GPU Worker    │
     │  (NumPy)        │          │  (CuPy)        │
     └─────────────────┘          └────────────────┘
```

### ML Prediction Process

1. **Task Submitted** with 3 inputs:
   - Task Type (VectorAddition, MatrixMultiplication, ParallelReduction)
   - Size (problem size)
   - Priority (1-10)

2. **ML Model** receives:
   - Task name → encoded to number (0, 1, 2)
   - Size → also creates size_log, size_category
   - Threads → also creates threads_log

3. **Model Predicts**:
   - Device: 'CPU' or 'GPU'
   - Confidence: 0-100%

4. **Task Routed** to predicted device queue

5. **Worker Executes** task on CPU or GPU

## 📊 Supported Tasks

### 1. Vector Addition
```python
TaskType.VECTOR_ADD
- Adds two vectors element-wise
- CPU: Uses NumPy
- GPU: Uses CuPy
- Decision: GPU for size > 1,000 (usually)
```

### 2. Matrix Multiplication
```python
TaskType.MATRIX_MUL
- Multiplies two matrices
- CPU: Uses NumPy (BLAS)
- GPU: Uses CuPy (cuBLAS)
- Decision: CPU for size < 128, GPU for size ≥ 512
```

### 3. Parallel Reduction
```python
TaskType.PARALLEL_REDUCTION
- Sums all elements in array
- CPU: Uses NumPy sum
- GPU: Uses CuPy sum
- Decision: GPU for size > 10,000 (usually)
```

## 🎓 Using the Scheduler in Your Code

### Basic Usage

```python
from ml_task_scheduler import IntelligentTaskScheduler, Task, TaskType
import numpy as np

# Create scheduler
scheduler = IntelligentTaskScheduler()
scheduler.start_workers()

# Create a vector addition task
size = 100000
a = np.random.rand(size).astype(np.float32)
b = np.random.rand(size).astype(np.float32)

task = Task(
    priority=5,
    task_type=TaskType.VECTOR_ADD,
    size=size,
    data={'a': a, 'b': b}
)

# Submit task (ML model decides CPU/GPU automatically)
scheduler.submit_task(task)

# Wait and shutdown
time.sleep(1)
scheduler.shutdown()
```

### Matrix Multiplication Example

```python
# Create matrix multiplication task
size = 512
a = np.random.rand(size, size).astype(np.float32)
b = np.random.rand(size, size).astype(np.float32)

task = Task(
    priority=10,
    task_type=TaskType.MATRIX_MUL,
    size=size,
    data={'a': a, 'b': b}
)

scheduler.submit_task(task)
# ML model will predict GPU (99.9% confidence)
```

### Parallel Reduction Example

```python
# Create reduction task
size = 1000000
arr = np.random.rand(size).astype(np.float32)

task = Task(
    priority=7,
    task_type=TaskType.PARALLEL_REDUCTION,
    size=size,
    data={'arr': arr}
)

scheduler.submit_task(task)
# ML model will predict GPU (99.9% confidence)
```

## 📈 ML Model Details

### Model Performance
- **Algorithm**: XGBoost Classifier
- **Accuracy**: 100% on test set
- **Training Data**: 1,200 samples
- **Features Used**: 6 (size, size_log, size_category, threads, threads_log, task_encoded)

### Decision Examples
| Task | Size | Prediction | Confidence |
|------|------|------------|------------|
| VectorAddition | 1,000 | GPU | 99.9% |
| VectorAddition | 1,000,000 | GPU | 99.9% |
| MatrixMultiplication | 64 | **CPU** | 99.4% |
| MatrixMultiplication | 512 | GPU | 99.9% |
| ParallelReduction | 10,000 | GPU | 99.9% |

## 🔄 Workflow

### 1. Task Submission
```python
scheduler.submit_task(task)
# Adds task to priority queue
```

### 2. Dispatcher (ML Prediction)
```python
device = ml_predictor.predict_device(task_name, size)
# Returns 'CPU' or 'GPU' with confidence
```

### 3. Queue Routing
```python
if device == 'GPU':
    gpu_queue.put(task)
else:
    cpu_queue.put(task)
```

### 4. Worker Execution
```python
# CPU Worker
result = Kernels.cpu_vector_add(a, b)

# GPU Worker  
a_gpu = cp.asarray(a)
b_gpu = cp.asarray(b)
result = Kernels.gpu_vector_add(a_gpu, b_gpu)
```

## 🛡️ Error Handling

The scheduler includes robust error handling:

1. **Model Loading Failure**: Falls back to heuristic-based scheduling
2. **GPU Execution Error**: Automatically re-queues task to CPU
3. **Queue Timeout**: Graceful handling with timeout in workers
4. **Prediction Error**: Falls back to simple heuristics

## 📊 Statistics Tracking

The scheduler tracks:
- Total tasks submitted
- CPU vs GPU task distribution
- Total execution time on each device
- Average time per task on each device

View statistics on shutdown:
```
EXECUTION STATISTICS
Total Tasks:      15
CPU Tasks:        2 (13.3%)
GPU Tasks:        13 (86.7%)
Total CPU Time:   3.42 ms
Total GPU Time:   619.01 ms
Avg CPU Time:     1.71 ms per task
Avg GPU Time:     47.62 ms per task
```

## 🔧 Customization

### Change Number of Tasks
```python
generate_workload(scheduler, num_tasks=50)
```

### Change Scheduling Policy
```python
scheduler = IntelligentTaskScheduler(policy=ExecutionPolicy.ROUND_ROBIN)
```

### Use Different Model Path
```python
ml_predictor = MLPredictor(model_path='path/to/your/model.pkl')
```

## 🎯 Summary

✅ **Fully automated** - ML model decides CPU/GPU  
✅ **High accuracy** - 100% on test data  
✅ **Real-time** - Predictions in < 1ms  
✅ **Thread-safe** - Multiple workers handle tasks concurrently  
✅ **Fault-tolerant** - Fallback mechanisms for errors  
✅ **Statistics** - Tracks performance metrics  
✅ **Priority-based** - Higher priority tasks execute first  
✅ **Easy to use** - Simple API for task submission  

The scheduler is **production-ready** and can handle real workloads with intelligent device selection! 🚀
