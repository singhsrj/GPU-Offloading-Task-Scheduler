#!/bin/bash
#chmod +x run_profiling.sh

# --- Activate Virtual Environment ---
echo "🔧 Activating virtual environment..."
source ../gpu_env/bin/activate

# --- Configuration ---
# The name of your Python profiler script
PROFILER_SCRIPT="file.py" 
# Directory to store the raw CSV data
OUTPUT_DIR="training_data"
# The final combined CSV file for ML training
FINAL_CSV="final_training_dataset.csv"
# List of CPU thread counts to use for profiling
# Expanded list for more diverse dataset to reach 1000+ rows
# With 35 sizes per task (105 total) x 10 thread counts = 1050 rows
THREAD_COUNTS=(1 2 3 4 6 8 10 12 14 16)

# --- Script Start ---
echo "🚀 Starting data collection for ML model training..."

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# 1. Run the profiler for each thread count
for threads in "${THREAD_COUNTS[@]}"; do
    echo "-----------------------------------------------------"
    echo "📊 Profiling with NUMBA_NUM_THREADS=${threads}"
    echo "-----------------------------------------------------"
    
    # Set the environment variable for numba/OpenMP
    export NUMBA_NUM_THREADS=${threads}
    
    # Run your python script
    python3 "${PROFILER_SCRIPT}"
    
    # Move and rename the output files to prevent overwriting
    echo "   -> Saving output files..."
    mv task1_vector_addition_profile.csv "${OUTPUT_DIR}/task1_profile_${threads}threads.csv"
    mv task2_matrix_multiplication_profile.csv "${OUTPUT_DIR}/task2_profile_${threads}threads.csv"
    mv task3_parallel_reduction_profile.csv "${OUTPUT_DIR}/task3_profile_${threads}threads.csv"
done

# 2. Combine all generated CSVs into a single master file
echo "-----------------------------------------------------"
echo "✨ Combining all data into ${FINAL_CSV}..."
echo "-----------------------------------------------------"

HEADER_WRITTEN=0
# Loop through all the CSV files in the output directory
for f in ${OUTPUT_DIR}/*.csv; do
  # Extract the thread count from the filename
  num_threads=$(echo "$f" | grep -o -E '[0-9]+threads' | sed 's/threads//')
  
  if [ $HEADER_WRITTEN -eq 0 ]; then
    # Write the header from the first file and add our new column header
    head -n 1 "$f" | sed 's/$/,num_cpu_threads/' > "$FINAL_CSV"
    HEADER_WRITTEN=1
  fi
  
  # Append the data from the current file (skipping its header) and add the thread count to each line
  tail -n +2 "$f" | sed "s/$/,$num_threads/" >> "$FINAL_CSV"
done

echo "✅ All done! Your training data is ready in ${FINAL_CSV}"