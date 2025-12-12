># 🚀 Bio-GPT: Quick Start & Step-by-Step Guide

This guide provides detailed instructions to set up and run the Bio-GPT project. 

---

## 1. Environment Setup

### Step 1: Clone the Repository
This command downloads the project files from GitHub to your local machine.
```bash
# Clone the project repository
git clone https://github.com/Egekocaslqn00/bio-gpt-cancer.git

# Navigate into the project directory
cd bio-gpt-cancer
```

### Step 2: Create a Virtual Environment
A virtual environment is a private workspace that keeps the project's dependencies separate from your system's other Python projects. This is a best practice.
```bash
# Create a virtual environment named 'venv'
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment
Activating the environment means all `pip` installations will be contained within this project.
```bash
# On Linux/Mac
source venv/bin/activate

# On Windows
# venv\Scripts\activate
```
*Your terminal prompt should now show `(venv)` at the beginning.*

### Step 4: Install Dependencies
This command reads the `requirements.txt` file and installs all the necessary Python libraries.
```bash
# Install all required packages
pip install -r requirements.txt
```
*This may take a few minutes as it downloads PyTorch and other large libraries.*

---

## 2. Running the Project (Step-by-Step)

Each script is designed to be run sequentially. 

### Step 5: Run Data Preparation (Synthetic Data)
This script generates the clean, synthetic dataset used for initial model training. It's fast and ensures the model works.
```bash
# Description: Creates a synthetic scRNA-seq dataset for rapid prototyping.
# Expected output: Visuals in `results/` and data files in `data/`.
# Duration: ~2-3 minutes
python 01_data_preparation.py
```

### Step 6: Train the Transformer Model
This script trains the Transformer model on the synthetic data and evaluates its performance.
```bash
# Description: Trains the model and saves the best version.
# Expected output: A trained model in `models/` and training plots in `results/`.
# Duration: ~5-10 minutes (on CPU)
python 02_transformer_model.py
```

### Step 7: Run Attention Analysis
This script interprets the trained model, analyzing which genes it found most important.
```bash
# Description: Analyzes the model's attention weights for biological insights.
# Expected output: Gene importance plots and stats in `results/`.
# Duration: ~2-3 minutes
python 03_attention_analysis.py
```

---

## 3. (Optional but Recommended) Running with Real-World Data

To make the project more impressive, we will now run it on a **real-world cancer dataset** from the GEO database. This demonstrates the model's robustness.

### Step 8: Run Real Data Analysis
This new script downloads a real dataset, preprocesses it, and evaluates our pre-trained model on it.
```bash
# Description: Downloads and tests the model on a real-world dataset (GSE184393).
# Expected output: New visualizations and results for real data in `results/`.
# Duration: ~15-30 minutes (includes data download)
python 04_real_data_analysis.py
```

---

## ⚙️ System Requirements

| Requirement | Minimum | Recommended |
|-----------|---------|----------|
| **RAM** | 8 GB | 16 GB |
| **Disk** | 1 GB | 2 GB |
| **CPU** | 4 cores | 8+ cores |
| **Python** | 3.8+ | 3.10+ |

---

## 🔧 Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'torch'`
**Solution:** Ensure your virtual environment is active and run `pip install -r requirements.txt` again.

### Problem: `CUDA out of memory`
**Solution:** The scripts are configured to run on CPU by default. If you have a GPU and run into memory issues, try reducing the `batch_size` in `02_transformer_model.py`.

### Problem: `Permission denied` (Linux/Mac)
**Solution:** Make the scripts executable with `chmod +x *.py`.

---

**Success! You have now run the entire project.** 🚀
