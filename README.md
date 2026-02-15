# EEG Benchmark Framework

Benchmarking framework using the **MOABB** library, primarily designed to evaluate the custom **BrainBotDataset** (16 channels, 5 classes).

It compares `BrainBot` performance against standard datasets (PhysioNet, Weibo), which are also benchmarked with channel reduction to assess the impact of sparsity on classification accuracy.

## 📊 Results

All benchmarks results can be found in **[results.ipynb](./results.ipynb)**

## 🛠️ How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Benchmarks**:
   - **Traditional ML (CPU)**: `python cpu_only.py`
   - **Deep Learning (TF)**: `python tf_only.py`
