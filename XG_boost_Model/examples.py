"""
Simple example showing how to use the Intelligent Task Scheduler
"""

from ml_task_scheduler import (IntelligentTaskScheduler, Task, TaskType, 
                               ExecutionPolicy)
import numpy as np
import time


def example_1_vector_addition():
    """Example: Vector addition tasks"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Vector Addition")
    print("="*70)
    
    scheduler = IntelligentTaskScheduler()
    scheduler.start_workers()
    
    # Small vector (might go to CPU)
    small_size = 500
    a = np.random.rand(small_size).astype(np.float32)
    b = np.random.rand(small_size).astype(np.float32)
    
    task1 = Task(
        priority=4,
        task_type=TaskType.VECTOR_ADD,
        size=small_size,
        data={'a': a, 'b': b}
    )
    scheduler.submit_task(task1)
    
    # Large vector (will go to GPU)
    large_size = 1000000
    a = np.random.rand(large_size).astype(np.float32)
    b = np.random.rand(large_size).astype(np.float32)
    
    task2 = Task(
        priority=4,
        task_type=TaskType.VECTOR_ADD,
        size=large_size,
        data={'a': a, 'b': b}
    )
    scheduler.submit_task(task2)
    
    time.sleep(1)
    scheduler.shutdown()


def example_2_matrix_multiplication():
    """Example: Matrix multiplication tasks"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Matrix Multiplication")
    print("="*70)
    
    scheduler = IntelligentTaskScheduler()
    scheduler.start_workers()
    
    # Small matrix (will go to CPU)
    small_size = 64
    a = np.random.rand(small_size, small_size).astype(np.float32)
    b = np.random.rand(small_size, small_size).astype(np.float32)
    
    task1 = Task(
        priority=4,
        task_type=TaskType.MATRIX_MUL,
        size=small_size,
        data={'a': a, 'b': b}
    )
    scheduler.submit_task(task1)
    
    # Medium matrix (borderline)
    medium_size = 256
    a = np.random.rand(medium_size, medium_size).astype(np.float32)
    b = np.random.rand(medium_size, medium_size).astype(np.float32)
    
    task2 = Task(
        priority=7,
        task_type=TaskType.MATRIX_MUL,
        size=medium_size,
        data={'a': a, 'b': b}
    )
    scheduler.submit_task(task2)
    
    # Large matrix (will go to GPU)
    large_size = 512
    a = np.random.rand(large_size, large_size).astype(np.float32)
    b = np.random.rand(large_size, large_size).astype(np.float32)
    
    task3 = Task(
        priority=10,
        task_type=TaskType.MATRIX_MUL,
        size=large_size,
        data={'a': a, 'b': b}
    )
    scheduler.submit_task(task3)
    
    time.sleep(1)
    scheduler.shutdown()


def example_3_parallel_reduction():
    """Example: Parallel reduction tasks"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Parallel Reduction (Sum)")
    print("="*70)
    
    scheduler = IntelligentTaskScheduler()
    scheduler.start_workers()
    
    # Different sizes
    sizes = [10000, 100000, 1000000]
    
    for size in sizes:
        arr = np.random.rand(size).astype(np.float32)
        
        task = Task(
            priority=5,
            task_type=TaskType.PARALLEL_REDUCTION,
            size=size,
            data={'arr': arr}
        )
        scheduler.submit_task(task)
    
    time.sleep(1)
    scheduler.shutdown()


def example_4_mixed_workload():
    """Example: Mixed workload with priorities"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Mixed Workload (Priority-based)")
    print("="*70)
    
    scheduler = IntelligentTaskScheduler(policy=ExecutionPolicy.PRIORITY)
    scheduler.start_workers()
    
    # Low priority vector addition
    task1 = Task(
        priority=2,
        task_type=TaskType.VECTOR_ADD,
        size=10000,
        data={'a': np.random.rand(10000).astype(np.float32),
              'b': np.random.rand(10000).astype(np.float32)}
    )
    
    # High priority matrix multiplication
    task2 = Task(
        priority=10,
        task_type=TaskType.MATRIX_MUL,
        size=256,
        data={'a': np.random.rand(256, 256).astype(np.float32),
              'b': np.random.rand(256, 256).astype(np.float32)}
    )
    
    # Medium priority reduction
    task3 = Task(
        priority=6,
        task_type=TaskType.PARALLEL_REDUCTION,
        size=100000,
        data={'arr': np.random.rand(100000).astype(np.float32)}
    )
    
    # Submit in random order (priority queue will handle)
    scheduler.submit_task(task1)  # Low priority
    scheduler.submit_task(task2)  # High priority (should execute first)
    scheduler.submit_task(task3)  # Medium priority
    
    time.sleep(1)
    scheduler.shutdown()


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("INTELLIGENT TASK SCHEDULER - USAGE EXAMPLES")
    print("="*70)
    
    # Run examples
    example_1_vector_addition()
    time.sleep(0.5)
    
    example_2_matrix_multiplication()
    time.sleep(0.5)
    
    example_3_parallel_reduction()
    time.sleep(0.5)
    
    example_4_mixed_workload()
    
    print("\n" + "="*70)
    print("✓ All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
