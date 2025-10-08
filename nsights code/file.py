"""
Three Computational Tasks with CPU/GPU Implementations
Includes NVTX annotations for NSight profiling and OpenMP parallelization

Required packages:
    pip install numpy cupy-cuda12x numba nvtx pandas

For OpenMP support in numba:
    Set environment variable: export NUMBA_NUM_THREADS=<num_cores>
"""

import numpy as np
import cupy as cp
import nvtx
from numba import cuda, prange, jit, njit
import time
import os


# =============================================================================
# TASK 1: VECTOR ADDITION (Element-wise operation)
# =============================================================================

class Task1_VectorAddition:
    """
    Vector Addition: C = A + B
    Tests: Memory bandwidth, thread divergence
    """
    
    @staticmethod
    @nvtx.annotate("Task1_CPU_VectorAdd", color="blue")
    def cpu_impl(n):
        """CPU implementation with OpenMP parallelization via numba"""
        
        @njit(parallel=True, fastmath=True)
        def parallel_vector_add(a, b, c, n):
            # OpenMP parallel loop via prange
            for i in prange(n):
                c[i] = a[i] + b[i]
        
        with nvtx.annotate("CPU_Memory_Allocation", color="cyan"):
            a = np.random.rand(n).astype(np.float32)
            b = np.random.rand(n).astype(np.float32)
            c = np.zeros(n, dtype=np.float32)
        
        with nvtx.annotate("CPU_Compute_VectorAdd", color="blue"):
            parallel_vector_add(a, b, c, n)
        
        with nvtx.annotate("CPU_Validation", color="cyan"):
            # Ensure computation is complete
            checksum = np.sum(c[:100])
            
        return c, checksum
    
    @staticmethod
    @nvtx.annotate("Task1_GPU_VectorAdd", color="green")
    def gpu_impl(n):
        """GPU implementation using CUDA kernel via CuPy"""
        
        with nvtx.annotate("GPU_Memory_Allocation", color="yellow"):
            a = cp.random.rand(n, dtype=cp.float32)
            b = cp.random.rand(n, dtype=cp.float32)
            c = cp.zeros(n, dtype=cp.float32)
        
        with nvtx.annotate("GPU_Kernel_VectorAdd", color="green"):
            # CuPy uses optimized CUDA kernels
            c = a + b
            cp.cuda.Stream.null.synchronize()
        
        with nvtx.annotate("GPU_D2H_Transfer", color="yellow"):
            checksum = float(cp.sum(c[:100]))
            result = cp.asnumpy(c)
        
        return result, checksum
    
    @staticmethod
    @nvtx.annotate("Task1_GPU_Custom_Kernel", color="green")
    def gpu_impl_custom(n):
        """GPU implementation with custom CUDA kernel"""
        
        @cuda.jit
        def vector_add_kernel(a, b, c):
            idx = cuda.grid(1)
            if idx < c.shape[0]:
                c[idx] = a[idx] + b[idx]
        
        with nvtx.annotate("GPU_Custom_Memory_Allocation", color="yellow"):
            a = cp.random.rand(n, dtype=cp.float32)
            b = cp.random.rand(n, dtype=cp.float32)
            c = cp.zeros(n, dtype=cp.float32)
        
        with nvtx.annotate("GPU_Custom_Kernel_Launch", color="green"):
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            
            vector_add_kernel[blocks_per_grid, threads_per_block](a, b, c)
            cuda.synchronize()
        
        with nvtx.annotate("GPU_Custom_D2H_Transfer", color="yellow"):
            checksum = float(cp.sum(c[:100]))
            result = cp.asnumpy(c)
        
        return result, checksum


# =============================================================================
# TASK 2: MATRIX MULTIPLICATION (Compute-intensive operation)
# =============================================================================

class Task2_MatrixMultiplication:
    """
    Matrix Multiplication: C = A @ B
    Tests: Compute throughput, cache utilization, memory patterns
    """
    
    @staticmethod
    @nvtx.annotate("Task2_CPU_MatMul", color="blue")
    def cpu_impl(n):
        """CPU implementation using NumPy (OpenBLAS/MKL with OpenMP)"""
        
        with nvtx.annotate("CPU_MatMul_Allocation", color="cyan"):
            a = np.random.rand(n, n).astype(np.float32)
            b = np.random.rand(n, n).astype(np.float32)
        
        with nvtx.annotate("CPU_MatMul_Compute", color="blue"):
            # NumPy uses optimized BLAS with OpenMP threads
            c = np.matmul(a, b)
        
        with nvtx.annotate("CPU_MatMul_Validation", color="cyan"):
            checksum = np.sum(c[0, :10])
        
        return c, checksum
    
    @staticmethod
    @nvtx.annotate("Task2_CPU_Manual_MatMul", color="blue")
    def cpu_impl_manual(n):
        """CPU implementation with manual OpenMP parallelization"""
        
        @njit(parallel=True, fastmath=True)
        def parallel_matmul(a, b, c, n):
            # Parallel outer loop with OpenMP
            for i in prange(n):
                for j in range(n):
                    tmp = 0.0
                    for k in range(n):
                        tmp += a[i, k] * b[k, j]
                    c[i, j] = tmp
        
        with nvtx.annotate("CPU_Manual_MatMul_Allocation", color="cyan"):
            a = np.random.rand(n, n).astype(np.float32)
            b = np.random.rand(n, n).astype(np.float32)
            c = np.zeros((n, n), dtype=np.float32)
        
        with nvtx.annotate("CPU_Manual_MatMul_Compute", color="blue"):
            parallel_matmul(a, b, c, n)
        
        with nvtx.annotate("CPU_Manual_MatMul_Validation", color="cyan"):
            checksum = np.sum(c[0, :10])
        
        return c, checksum
    
    @staticmethod
    @nvtx.annotate("Task2_GPU_MatMul", color="green")
    def gpu_impl(n):
        """GPU implementation using cuBLAS via CuPy"""
        
        with nvtx.annotate("GPU_MatMul_H2D_Transfer", color="yellow"):
            a = cp.random.rand(n, n, dtype=cp.float32)
            b = cp.random.rand(n, n, dtype=cp.float32)
        
        with nvtx.annotate("GPU_cuBLAS_MatMul", color="green"):
            # CuPy uses highly optimized cuBLAS
            c = cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
        
        with nvtx.annotate("GPU_MatMul_D2H_Transfer", color="yellow"):
            checksum = float(cp.sum(c[0, :10]))
            result = cp.asnumpy(c)
        
        return result, checksum


# =============================================================================
# TASK 3: PARALLEL REDUCTION (Communication-intensive operation)
# =============================================================================

class Task3_ParallelReduction:
    """
    Parallel Reduction: Sum of all array elements
    Tests: Thread synchronization, reduction patterns, memory access
    """
    
    @staticmethod
    @nvtx.annotate("Task3_CPU_Reduction", color="blue")
    def cpu_impl(n):
        """CPU implementation with OpenMP reduction"""
        
        @njit(parallel=True, fastmath=True)
        def parallel_reduction(arr):
            # OpenMP reduction clause
            total = 0.0
            for i in prange(len(arr)):
                total += arr[i]
            return total
        
        with nvtx.annotate("CPU_Reduction_Allocation", color="cyan"):
            a = np.random.rand(n).astype(np.float32)
        
        with nvtx.annotate("CPU_Reduction_Compute", color="blue"):
            result = parallel_reduction(a)
        
        return result
    
    @staticmethod
    @nvtx.annotate("Task3_CPU_TwoStage_Reduction", color="blue")
    def cpu_impl_twostage(n):
        """CPU implementation with two-stage reduction (more complex)"""
        
        @njit(parallel=True, fastmath=True)
        def two_stage_reduction(arr, n_chunks):
            chunk_size = len(arr) // n_chunks
            partial_sums = np.zeros(n_chunks, dtype=np.float32)
            
            # Stage 1: Parallel chunk reduction
            for chunk_id in prange(n_chunks):
                start = chunk_id * chunk_size
                end = start + chunk_size if chunk_id < n_chunks - 1 else len(arr)
                local_sum = 0.0
                for i in range(start, end):
                    local_sum += arr[i]
                partial_sums[chunk_id] = local_sum
            
            # Stage 2: Final reduction (serial)
            total = 0.0
            for i in range(n_chunks):
                total += partial_sums[i]
            
            return total
        
        with nvtx.annotate("CPU_TwoStage_Allocation", color="cyan"):
            a = np.random.rand(n).astype(np.float32)
            n_chunks = os.cpu_count() or 8
        
        with nvtx.annotate("CPU_TwoStage_Compute", color="blue"):
            result = two_stage_reduction(a, n_chunks)
        
        return result
    
    @staticmethod
    @nvtx.annotate("Task3_GPU_Reduction", color="green")
    def gpu_impl(n):
        """GPU implementation using CuPy reduction"""
        
        with nvtx.annotate("GPU_Reduction_H2D_Transfer", color="yellow"):
            a = cp.random.rand(n, dtype=cp.float32)
        
        with nvtx.annotate("GPU_Reduction_Compute", color="green"):
            # CuPy uses optimized reduction kernel
            result = cp.sum(a)
            cp.cuda.Stream.null.synchronize()
        
        with nvtx.annotate("GPU_Reduction_D2H_Transfer", color="yellow"):
            result = float(result)
        
        return result
    
    @staticmethod
    @nvtx.annotate("Task3_GPU_Custom_Reduction", color="green")
    def gpu_impl_custom(n):
        """GPU implementation with custom reduction kernel"""
        
        @cuda.jit
        def reduction_kernel(arr, partial_sums):
            # Shared memory for thread block reduction
            shared = cuda.shared.array(256, dtype=np.float32)
            
            tid = cuda.threadIdx.x
            bid = cuda.blockIdx.x
            idx = bid * cuda.blockDim.x + tid
            
            # Load data into shared memory
            if idx < arr.shape[0]:
                shared[tid] = arr[idx]
            else:
                shared[tid] = 0.0
            
            cuda.syncthreads()
            
            # Tree-based reduction in shared memory
            stride = cuda.blockDim.x // 2
            while stride > 0:
                if tid < stride and idx + stride < arr.shape[0]:
                    shared[tid] += shared[tid + stride]
                cuda.syncthreads()
                stride //= 2
            
            # Write block result
            if tid == 0:
                partial_sums[bid] = shared[0]
        
        with nvtx.annotate("GPU_Custom_Reduction_Allocation", color="yellow"):
            a = cp.random.rand(n, dtype=cp.float32)
            
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            partial_sums = cp.zeros(blocks_per_grid, dtype=cp.float32)
        
        with nvtx.annotate("GPU_Custom_Reduction_Kernel", color="green"):
            reduction_kernel[blocks_per_grid, threads_per_block](a, partial_sums)
            cuda.synchronize()
        
        with nvtx.annotate("GPU_Custom_Final_Sum", color="green"):
            # Final reduction on CPU (small array)
            result = float(cp.sum(partial_sums))
        
        return result


# =============================================================================
# PROFILING AND BENCHMARKING UTILITIES
# =============================================================================

class TaskProfiler:
    """Profile tasks across different sizes for ML training data collection"""
    
    @staticmethod
    @nvtx.annotate("Profiler_Benchmark", color="red")
    def benchmark_task(task_func, n, device, warmup=2, runs=5):
        """
        Benchmark a task with warmup and multiple runs
        
        Args:
            task_func: Function to benchmark
            n: Input size
            device: 'CPU' or 'GPU'
            warmup: Number of warmup runs
            runs: Number of timed runs
            
        Returns:
            (mean_time, std_time, result)
        """
        # Warmup runs
        for _ in range(warmup):
            _ = task_func(n)
        
        # Timed runs
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            result = task_func(n)
            end = time.perf_counter()
            times.append(end - start)
        
        return np.mean(times), np.std(times), result
    
    @staticmethod
    @nvtx.annotate("Profiler_CollectData", color="red")
    def collect_profiling_data(task_name, cpu_func, gpu_func, size_range, output_file):
        """
        Collect profiling data for a task across size range
        
        Usage with NSight:
            nsys profile -t cuda,nvtx,openmp --openmp-tracking=true \\
                 -o profile_data python this_script.py
        """
        results = []
        
        print(f"\n{'='*60}")
        print(f"Profiling {task_name}")
        print(f"{'='*60}")
        
        for n in size_range:
            with nvtx.annotate(f"Profile_Size_{n}", color="red"):
                print(f"\nSize: {n:,}")
                
                # CPU profiling
                try:
                    cpu_time, cpu_std, cpu_result = TaskProfiler.benchmark_task(
                        cpu_func, n, 'CPU', warmup=2, runs=3
                    )
                    print(f"  CPU: {cpu_time*1000:.3f} ± {cpu_std*1000:.3f} ms")
                except Exception as e:
                    print(f"  CPU: Error - {e}")
                    cpu_time = float('inf')
                    cpu_std = 0
                
                # GPU profiling
                try:
                    gpu_time, gpu_std, gpu_result = TaskProfiler.benchmark_task(
                        gpu_func, n, 'GPU', warmup=2, runs=3
                    )
                    print(f"  GPU: {gpu_time*1000:.3f} ± {gpu_std*1000:.3f} ms")
                except Exception as e:
                    print(f"  GPU: Error - {e}")
                    gpu_time = float('inf')
                    gpu_std = 0
                
                # Determine optimal device
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                optimal = 'GPU' if gpu_time < cpu_time else 'CPU'
                label = 1 if optimal == 'GPU' else 0
                
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Optimal: {optimal}")
                
                results.append({
                    'task': task_name,
                    'size': n,
                    'cpu_time_ms': cpu_time * 1000,
                    'cpu_std_ms': cpu_std * 1000,
                    'gpu_time_ms': gpu_time * 1000,
                    'gpu_std_ms': gpu_std * 1000,
                    'speedup': speedup,
                    'optimal_device': optimal,
                    'label': label
                })
        
        # Save to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved profiling data to {output_file}")
        
        return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution for profiling all three tasks"""
    
    print("\n" + "="*60)
    print("GPU TASK SCHEDULER - PROFILING MODE")
    print("="*60)
    print(f"NumPy threads: {np.__config__.show()}")
    print(f"Numba threads: {os.environ.get('NUMBA_NUM_THREADS', 'auto')}")
    
    # Define size ranges for each task
    vector_sizes = [10**i for i in range(4, 9)]  # 10K to 100M
    matrix_sizes = [2**i for i in range(6, 11)]  # 64 to 1024
    reduction_sizes = [10**i for i in range(4, 9)]  # 10K to 100M
    
    # Task 1: Vector Addition
    TaskProfiler.collect_profiling_data(
        task_name="VectorAddition",
        cpu_func=Task1_VectorAddition.cpu_impl,
        gpu_func=Task1_VectorAddition.gpu_impl,
        size_range=vector_sizes,
        output_file="task1_vector_addition_profile.csv"
    )
    
    # Task 2: Matrix Multiplication
    TaskProfiler.collect_profiling_data(
        task_name="MatrixMultiplication",
        cpu_func=Task2_MatrixMultiplication.cpu_impl,
        gpu_func=Task2_MatrixMultiplication.gpu_impl,
        size_range=matrix_sizes,
        output_file="task2_matrix_multiplication_profile.csv"
    )
    
    # Task 3: Parallel Reduction
    TaskProfiler.collect_profiling_data(
        task_name="ParallelReduction",
        cpu_func=Task3_ParallelReduction.cpu_impl,
        gpu_func=Task3_ParallelReduction.gpu_impl,
        size_range=reduction_sizes,
        output_file="task3_parallel_reduction_profile.csv"
    )
    
    print("\n" + "="*60)
    print("✓ PROFILING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run with NSight Systems for detailed profiling:")
    print("   nsys profile -t cuda,nvtx,openmp --openmp-tracking=true \\")
    print("        -o profile_data python this_script.py")
    print("\n2. Export NSight statistics:")
    print("   nsys stats --report cuda_api_sum,nvtx_sum profile_data.nsys-rep")
    print("\n3. Use the CSV files for ML model training")


if __name__ == "__main__":
    main()