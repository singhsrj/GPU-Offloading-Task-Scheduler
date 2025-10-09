# Simple GPU Offloading Predictor

A straightforward XGBoost model to predict whether tasks should run on CPU or GPU.

## Files

- **`simple_train.py`** - Train the XGBoost model (100% accuracy)
- **`simple_predict.py`** - Load model and make predictions
- **`xgboost_model.pkl`** - Trained model (ready to use)

## How It Works

### Input Features (3 simple inputs):
1. **Task Name**: VectorAddition, MatrixMultiplication, or ParallelReduction
2. **Size**: Problem size (number of elements or matrix dimension)
3. **Threads**: Number of CPU threads available

### Output:
- **Device**: CPU or GPU
- **Confidence**: Prediction probability (0-100%)

## Usage

### Train Model
```bash
python3 simple_train.py
```
This creates `xgboost_model.pkl` file.

### Make Predictions
```python
from simple_predict import load_model, predict

# Load trained model
model_data = load_model('xgboost_model.pkl')

# Predict
device, confidence = predict(model_data, 'VectorAddition', 1000000, 8)
print(f"Use {device} with {confidence:.0%} confidence")
# Output: Use GPU with 100% confidence
```

Or run the demo:
```bash
python3 simple_predict.py
```

## Examples

| Task | Size | Threads | Result | Confidence |
|------|------|---------|--------|------------|
| VectorAddition | 1,000 | 4 | GPU | 100% |
| VectorAddition | 1,000,000 | 8 | GPU | 100% |
| MatrixMultiplication | 64 | 4 | **CPU** | 99% |
| MatrixMultiplication | 512 | 8 | GPU | 100% |
| ParallelReduction | 10,000 | 2 | GPU | 100% |
| ParallelReduction | 10,000,000 | 16 | GPU | 100% |

## Model Details

- **Algorithm**: XGBoost Classifier
- **Accuracy**: 100% on test set
- **Training Data**: 1,200 samples
- **Features**: 6 engineered features (size, size_log, size_category, threads, threads_log, task_encoded)
- **Training Time**: ~1 second
- **Prediction Time**: < 1 millisecond

## Integration Example

```python
import pickle
import numpy as np

# Load model once at startup
with open('xgboost_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
task_encoder = model_data['task_encoder']

# Use in your scheduler
def decide_device(task_name, size, num_threads):
    task_encoded = task_encoder.transform([task_name])[0]
    features = np.array([[
        size,
        np.log10(size + 1),
        2 if size > 100000 else (1 if size > 1000 else 0),
        num_threads,
        np.log10(num_threads),
        task_encoded
    ]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return 'GPU' if prediction == 1 else 'CPU'

# Use it
device = decide_device('MatrixMultiplication', 512, 8)
print(f"Run on: {device}")  # Output: Run on: GPU
```

## Requirements

```bash
pip install xgboost scikit-learn pandas numpy
```

All requirements already installed in `gpu_env`.
