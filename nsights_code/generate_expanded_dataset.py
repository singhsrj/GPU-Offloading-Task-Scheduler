"""
Generate expanded training dataset (1000+ rows) based on existing profiling data patterns
This is much faster than profiling and provides good training data with realistic noise
"""

import pandas as pd
import numpy as np

def load_existing_data():
    """Load the existing small dataset to learn patterns"""
    df = pd.read_csv('final_training_dataset.csv')
    return df

def generate_expanded_dataset(output_file='expanded_training_dataset.csv', target_rows=1200):
    """
    Generate expanded dataset by interpolating and adding realistic noise
    """
    print("📊 Loading existing profiling data...")
    df_base = load_existing_data()
    
    print(f"   Base dataset: {len(df_base)} rows")
    print(f"   Target: {target_rows} rows")
    
    # Define expanded size ranges
    vector_sizes = [int(10**(3 + i*0.10)) for i in range(40)]  # 40 sizes
    matrix_sizes = list(range(32, 1600, 32)) + list(range(32, 512, 8))  # ~60 sizes
    matrix_sizes = sorted(list(set(matrix_sizes)))[:40]  # Take 40 unique sizes
    reduction_sizes = [int(10**(3 + i*0.10)) for i in range(40)]  # 40 sizes
    
    thread_counts = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
    
    all_data = []
    
    for task_name, sizes in [('VectorAddition', vector_sizes),
                             ('MatrixMultiplication', matrix_sizes),
                             ('ParallelReduction', reduction_sizes)]:
        
        print(f"\n🔧 Generating data for {task_name}...")
        
        # Get base data for this task
        task_base = df_base[df_base['task'] == task_name].copy()
        
        if len(task_base) == 0:
            print(f"   ⚠️  No base data for {task_name}, skipping...")
            continue
        
        for num_threads in thread_counts:
            # Get closest thread count data from base
            thread_base = task_base[task_base['num_cpu_threads'] == num_threads]
            
            if len(thread_base) == 0:
                # Use closest available thread count
                closest_thread = task_base['num_cpu_threads'].iloc[0]
                thread_base = task_base[task_base['num_cpu_threads'] == closest_thread]
            
            for size in sizes:
                # Find nearest size in base data for interpolation
                size_diffs = np.abs(thread_base['size'].values - size)
                nearest_idx = np.argmin(size_diffs)
                base_row = thread_base.iloc[nearest_idx]
                
                # Interpolation factor (how far from nearest base point)
                size_ratio = size / base_row['size']
                
                # Generate CPU time (scales roughly linearly for vectors/reduction, O(n^3) for matrix)
                if task_name == 'MatrixMultiplication':
                    # Matrix multiplication is O(n^3)
                    cpu_time = base_row['cpu_time_ms'] * (size_ratio ** 3)
                    cpu_std = base_row['cpu_std_ms'] * (size_ratio ** 3)
                else:
                    # Vector operations are O(n)
                    cpu_time = base_row['cpu_time_ms'] * size_ratio
                    cpu_std = base_row['cpu_std_ms'] * size_ratio
                
                # GPU time scales better but still increases
                if task_name == 'MatrixMultiplication':
                    gpu_time = base_row['gpu_time_ms'] * (size_ratio ** 2.5)  # Better scaling on GPU
                    gpu_std = base_row['gpu_std_ms'] * (size_ratio ** 2.5)
                else:
                    gpu_time = base_row['gpu_time_ms'] * (size_ratio ** 0.8)  # Sub-linear scaling
                    gpu_std = base_row['gpu_std_ms'] * (size_ratio ** 0.8)
                
                # Add realistic noise (±5-15%)
                noise_factor_cpu = np.random.uniform(0.90, 1.10)
                noise_factor_gpu = np.random.uniform(0.90, 1.10)
                
                cpu_time *= noise_factor_cpu
                cpu_std *= np.random.uniform(0.8, 1.2)
                gpu_time *= noise_factor_gpu
                gpu_std *= np.random.uniform(0.8, 1.2)
                
                # Ensure reasonable values
                cpu_time = max(cpu_time, 0.01)
                gpu_time = max(gpu_time, 0.01)
                cpu_std = max(cpu_std, 0.001)
                gpu_std = max(gpu_std, 0.001)
                
                # Calculate metrics
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                optimal_device = 'GPU' if gpu_time < cpu_time else 'CPU'
                label = 1 if optimal_device == 'GPU' else 0
                
                all_data.append({
                    'task': task_name,
                    'size': size,
                    'cpu_time_ms': cpu_time,
                    'cpu_std_ms': cpu_std,
                    'gpu_time_ms': gpu_time,
                    'gpu_std_ms': gpu_std,
                    'speedup': speedup,
                    'optimal_device': optimal_device,
                    'label': label,
                    'num_cpu_threads': num_threads
                })
    
    # Create DataFrame
    df_expanded = pd.DataFrame(all_data)
    
    print(f"\n✅ Generated {len(df_expanded)} rows")
    print(f"\n📈 Dataset Statistics:")
    print(f"   - Tasks: {df_expanded['task'].nunique()}")
    print(f"   - Size range: {df_expanded['size'].min():,} to {df_expanded['size'].max():,}")
    print(f"   - Thread counts: {sorted(df_expanded['num_cpu_threads'].unique())}")
    print(f"   - CPU optimal: {(df_expanded['label'] == 0).sum()} ({(df_expanded['label'] == 0).sum()/len(df_expanded)*100:.1f}%)")
    print(f"   - GPU optimal: {(df_expanded['label'] == 1).sum()} ({(df_expanded['label'] == 1).sum()/len(df_expanded)*100:.1f}%)")
    
    # Save to CSV
    df_expanded.to_csv(output_file, index=False)
    print(f"\n💾 Saved to {output_file}")
    
    # Show sample
    print(f"\n📋 Sample rows:")
    print(df_expanded.head(10).to_string())
    
    return df_expanded

if __name__ == "__main__":
    df = generate_expanded_dataset(
        output_file='expanded_training_dataset.csv',
        target_rows=1200
    )
