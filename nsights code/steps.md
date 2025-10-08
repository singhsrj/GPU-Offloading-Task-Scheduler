# GPU Task Scheduler - Complete Profiling Guide

## 📋 Prerequisites

### 1. Install Required Packages
```bash
# Python packages
pip install numpy cupy-cuda12x numba nvtx pandas matplotlib scikit-learn pynvml

# For CUDA 11.x, use: cupy-cuda11x
```

### 2. Set Environment Variables
```bash
# Enable OpenMP in numba (use your CPU core count)
export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=8

# For better NumPy performance with OpenBLAS
export OPENBLAS_NUM_THREADS=8
```

### 3. Verify Installation
```bash
# Check CUDA
nvidia-smi

# Check NSight Systems
nsys --version

# Check NSight Compute
ncu --version
```

---

## 🚀 Phase 1: Basic Profiling with NSight Systems

### Step 1: Run Basic Profiling
```bash
# Basic profiling with CUDA, NVTX, and OpenMP tracking
nsys profile \
    -t cuda,nvtx,openmp \
    --openmp-tracking=true \
    --cuda-memory-usage=true \
    -o basic_profile \
    python tasks.py
```

### Step 2: View in GUI
```bash
# Open NSight Systems GUI
nsys-ui basic_profile.nsys-rep
```

**What to look for:**
- ✅ NVTX markers showing task boundaries (colored regions)
- ✅ OpenMP threads executing in parallel
- ✅ GPU kernel execution timeline
- ✅ Memory transfer patterns (H2D, D2H)

---

## 📊 Phase 2: Detailed Profiling for Each Task

### Task 1: Vector Addition
```bash
# Focus on memory bandwidth
nsys profile \
    -t cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=0 \
    -o vector_add_profile \
    python -c "
from tasks import Task1_VectorAddition
import nvtx
with nvtx.annotate('VectorAdd_1M'):
    Task1_VectorAddition.cpu_impl(1000000)
    Task1_VectorAddition.gpu_impl(1000000)
"
```

### Task 2: Matrix Multiplication
```bash
# Focus on compute throughput
nsys profile \
    -t cuda,nvtx,cublas \
    --cuda-memory-usage=true \
    -o matmul_profile \
    python -c "
from tasks import Task2_MatrixMultiplication
import nvtx
with nvtx.annotate('MatMul_512x512'):
    Task2_MatrixMultiplication.cpu_impl(512)
    Task2_MatrixMultiplication.gpu_impl(512)
"
```

### Task 3: Parallel Reduction
```bash
# Focus on synchronization and reduction patterns
nsys profile \
    -t cuda,nvtx,openmp \
    --openmp-tracking=true \
    -o reduction_profile \
    python -c "
from tasks import Task3_ParallelReduction
import nvtx
with nvtx.annotate('Reduction_10M'):
    Task3_ParallelReduction.cpu_impl(10000000)
    Task3_ParallelReduction.gpu_impl(10000000)
"
```

---

## 🔬 Phase 3: Deep Dive with NSight Compute

NSight Compute provides **kernel-level** analysis.

### Analyze GPU Kernels
```bash
# Profile GPU kernels with full metrics
ncu \
    --set full \
    --target-processes all \
    --kernel-name regex:vector_add \
    -o vector_add_detailed \
    python -c "
from tasks import Task1_VectorAddition
Task1_VectorAddition.gpu_impl_custom(1000000)
"
```

### Key Metrics to Collect

#### Memory Metrics:
```bash
ncu --metrics \
    dram_throughput.avg,\
    l1tex_throughput.avg,\
    l2_throughput.avg,\
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
    -o memory_metrics \
    python your_script.py
```

#### Compute Metrics:
```bash
ncu --metrics \
    sm_efficiency,\
    achieved_occupancy,\
    ipc,\
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum \
    -o compute_metrics \
    python your_script.py
```

---

## 📈 Phase 4: Export Data for ML Training

### Export NSight Systems Statistics
```bash
# Export CUDA API summary
nsys stats \
    --report cuda_api_sum \
    --format csv \
    --output cuda_api \
    basic_profile.nsys-rep

# Export GPU kernel summary
nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output gpu_kernels \
    basic_profile.nsys-rep

# Export NVTX trace (most useful for task boundaries)
nsys stats \
    --report nvtx_sum \
    --format csv \
    --output nvtx_markers \
    basic_profile.nsys-rep

# Export NVTX push/pop trace with timing
nsys stats \
    --report nvtx_pushpop_trace \
    --format csv \
    --output nvtx_timing \
    basic_profile.nsys-rep
```

### Export NSight Compute Data
```bash
# Export to CSV
ncu --csv \
    --page details \
    --import vector_add_detailed.ncu-rep \
    > vector_add_metrics.csv
```

---

## 🎯 Phase 5: Collect Training Data

### Run Complete Profiling Suite
```bash
# Run the profiling script
python tasks.py

# This generates:
# - task1_vector_addition_profile.csv
# - task2_matrix_multiplication_profile.csv
# - task3_parallel_reduction_profile.csv
```

### Combine with NSight Data
```bash
# Profile with NSight while collecting timing data
nsys profile \
    -t cuda,nvtx,openmp \
    --openmp-tracking=true \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -o full_profile \
    python tasks.py

# Export all statistics
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum \
    --format csv \
    full_profile.nsys-rep
```

---

## 📊 Understanding NVTX Markers

Your code has colored NVTX markers:
- 🔵 **Blue** - CPU operations
- 🟢 **Green** - GPU operations  
- 🟡 **Yellow** - Memory transfers
- 🔴 **Red** - Profiling/measurement overhead
- 🔷 **Cyan** - Memory allocation

In NSight Systems GUI:
1. Zoom into timeline (scroll or drag)
2. Click NVTX markers to see timing
3. Compare CPU vs GPU execution overlap
4. Identify data transfer bottlenecks

---

## 🔍 Key Performance Indicators (KPIs)

### From NSight Systems:
- **Kernel Duration**: Total GPU execution time
- **Memory Transfer Time**: H2D + D2H overhead
- **CPU-GPU Overlap**: Concurrent execution
- **SM Utilization**: % of GPU being used
- **Memory Throughput**: GB/s achieved

### From NSight Compute:
- **Achieved Occupancy**: Thread utilization
- **Memory Bandwidth**: % of peak bandwidth
- **Compute Throughput**: FLOPS achieved
- **Warp Efficiency**: Thread divergence
- **L1/L2 Cache Hit Rates**

---

## 🎓 Analysis Workflow

### 1. Identify Crossover Points
```python
# After profiling, plot results
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('task1_vector_addition_profile.csv')

plt.figure(figsize=(10, 6))
plt.plot(df['size'], df['cpu_time_ms'], 'b-o', label='CPU')
plt.plot(df['size'], df['gpu_time_ms'], 'g-s', label='GPU')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Input Size')
plt.ylabel('Execution Time (ms)')
plt.title('CPU vs GPU Performance - Vector Addition')
plt.legend()
plt.grid(True)
plt.savefig('task1_crossover.png')
```

### 2. Extract Features from NSight
Key metrics to extract for ML training:
- `nvtx_sum.csv`: Task execution times
- `cuda_gpu_kern_sum.csv`: Kernel times, grid/block sizes
- `cuda_api_sum.csv`: Memory transfer times

### 3. Bottleneck Analysis Checklist
- [ ] Is memory transfer > compute time? → Try unified memory
- [ ] Is CPU faster for small inputs? → Set threshold
- [ ] Low GPU utilization? → Increase parallelism
- [ ] High memory usage? → Optimize data structures
- [ ] Thread divergence? → Restructure algorithm

---

## 🚨 Common Issues & Solutions

### Issue 1: OpenMP Not Showing in NSight
```bash
# Make sure to use --openmp-tracking=true
# Also set environment variable
export OMP_NUM_THREADS=8

# Verify numba is using OpenMP
python -c "import numba; print(numba.config.NUMBA_NUM_THREADS)"
```

### Issue 2: NVTX Markers Not Visible
```bash
# Ensure nvtx package is installed
pip install nvtx

# Check if NVTX tracking is enabled in nsys
nsys profile -t nvtx,cuda ...
```

### Issue 3: CUDA Out of Memory
```python
# Add memory management
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()
```

### Issue 4: Inconsistent Timing
```python
# Add proper warmup and synchronization
cp.cuda.Stream.null.synchronize()  # GPU
cuda.synchronize()  # Custom kernels
```

---

## 📝 Next Steps

1. ✅ Run profiling: `python tasks.py`
2. ✅ Visualize with NSight Systems: `nsys-ui full_profile.nsys-rep`
3. ✅ Export statistics for ML training
4. ⏭️ Train ML model with collected data
5. ⏭️ Implement NVML-based runtime scheduler
6. ⏭️ Optimize bottlenecks identified in profiling

---

## 📚 Additional Resources

- [NSight Systems User Guide](