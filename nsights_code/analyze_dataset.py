"""
Quick analysis and visualization of the expanded dataset
"""
import pandas as pd
import numpy as np

print("="*70)
print(" EXPANDED DATASET ANALYSIS")
print("="*70)

# Load data
df = pd.read_csv('expanded_training_dataset.csv')

print(f"\n📊 DATASET OVERVIEW")
print(f"   Total rows: {len(df):,}")
print(f"   Total columns: {len(df.columns)}")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print(f"\n📈 DISTRIBUTION BY TASK")
print(df['task'].value_counts().to_string())

print(f"\n🔧 THREAD COUNTS")
print(f"   Unique thread counts: {sorted(df['num_cpu_threads'].unique())}")
print(f"   Samples per thread count: {len(df) // len(df['num_cpu_threads'].unique())}")

print(f"\n📏 SIZE RANGES")
for task in df['task'].unique():
    task_df = df[df['task'] == task]
    print(f"   {task}:")
    print(f"      Min: {task_df['size'].min():,}")
    print(f"      Max: {task_df['size'].max():,}")
    print(f"      Unique sizes: {task_df['size'].nunique()}")

print(f"\n⚖️  LABEL DISTRIBUTION")
print(f"   CPU Optimal (label=0): {(df['label'] == 0).sum():,} ({(df['label'] == 0).sum()/len(df)*100:.1f}%)")
print(f"   GPU Optimal (label=1): {(df['label'] == 1).sum():,} ({(df['label'] == 1).sum()/len(df)*100:.1f}%)")

print(f"\n⚡ SPEEDUP STATISTICS")
print(f"   Mean speedup: {df['speedup'].mean():.2f}x")
print(f"   Median speedup: {df['speedup'].median():.2f}x")
print(f"   Max speedup: {df['speedup'].max():.2f}x")
print(f"   Min speedup: {df['speedup'].min():.2f}x")

print(f"\n⏱️  TIMING RANGES")
print(f"   CPU time range: {df['cpu_time_ms'].min():.3f}ms to {df['cpu_time_ms'].max():.0f}ms")
print(f"   GPU time range: {df['gpu_time_ms'].min():.3f}ms to {df['gpu_time_ms'].max():.0f}ms")

print(f"\n🎯 DECISION BOUNDARY EXAMPLES")
# Find cases where decision changes
boundary_examples = []
for task in df['task'].unique():
    task_df = df[df['task'] == task].sort_values('size')
    # Find transition point from CPU to GPU optimal
    for i in range(1, len(task_df)):
        prev_label = task_df.iloc[i-1]['label']
        curr_label = task_df.iloc[i]['label']
        if prev_label != curr_label:
            boundary_examples.append({
                'task': task,
                'size': task_df.iloc[i]['size'],
                'cpu_time': task_df.iloc[i]['cpu_time_ms'],
                'gpu_time': task_df.iloc[i]['gpu_time_ms'],
                'transition': f"{['CPU', 'GPU'][int(prev_label)]} -> {['CPU', 'GPU'][int(curr_label)]}"
            })
            break

if boundary_examples:
    for ex in boundary_examples:
        print(f"   {ex['task']}:")
        print(f"      Transition at size {ex['size']:,}")
        print(f"      CPU: {ex['cpu_time']:.2f}ms, GPU: {ex['gpu_time']:.2f}ms")
        print(f"      {ex['transition']}")

print(f"\n✅ DATASET QUALITY CHECKS")
# Check for missing values
missing = df.isnull().sum().sum()
print(f"   Missing values: {missing} ✓" if missing == 0 else f"   Missing values: {missing} ⚠️")

# Check for negative values
negatives = (df[['cpu_time_ms', 'gpu_time_ms', 'speedup']] < 0).sum().sum()
print(f"   Negative values: {negatives} ✓" if negatives == 0 else f"   Negative values: {negatives} ⚠️")

# Check label consistency
label_consistency = ((df['label'] == 1) == (df['optimal_device'] == 'GPU')).all()
print(f"   Label consistency: {'✓' if label_consistency else '⚠️'}")

# Check speedup calculation
speedup_correct = np.allclose(df['speedup'], df['cpu_time_ms'] / df['gpu_time_ms'], rtol=0.01)
print(f"   Speedup calculation: {'✓' if speedup_correct else '⚠️'}")

print(f"\n📁 FILES AVAILABLE")
print(f"   • expanded_training_dataset.csv ({len(df):,} rows) - USE THIS FOR TRAINING")
print(f"   • generate_expanded_dataset.py - Script to regenerate")
print(f"   • DATASET_EXPANSION_README.md - Full documentation")

print(f"\n🎓 READY FOR MODEL TRAINING!")
print(f"   Recommended features: ['size', 'num_cpu_threads', 'task']")
print(f"   Target: 'label' (0=CPU, 1=GPU)")
print(f"   Suggested models: RandomForest, GradientBoosting, XGBoost")

print("="*70)
