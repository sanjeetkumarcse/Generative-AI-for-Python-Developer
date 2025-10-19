# Assignment: CKD Classification and Prediction using PyTorch & TensorFlow

## Paper Reference
Metherall, B., Berryman, A. K., & Brennan, G. S. (2025).  
*Machine Learning for Classifying Chronic Kidney Disease and Predicting Creatinine Levels Using At-Home Measurements.*  
**Scientific Reports, 15**, 4364.  
[DOI: 10.1038/s41598-025-88631-y](https://doi.org/10.1038/s41598-025-88631-y)

---

## Objective
Reproduce and improve the CKD classification and creatinine prediction tasks from the paper using **two deep learning frameworks**:

1. **PyTorch Implementation**
2. **TensorFlow / Keras Implementation**

You will:
- Use the same **UCI Chronic Kidney Disease (CKD) dataset**.  
- Build and train **ANN models** for both classification and regression.  
- Compare performance of both frameworks with **Random Forest baselines**.  
- Apply data preprocessing, visualization, and feature analysis.

---

## Dataset
**Source:** [UCI Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)  
**Samples:** 400  
**Features:** 25  
**Targets:**  
- `classification`: CKD vs Not CKD  
- `regression`: Serum creatinine level

---

## Tasks Overview

### 1. Data Preprocessing
Perform these steps in both implementations:
- Load dataset using `pandas`.
- Handle missing values (remove rows missing `serum creatinine`).
- Impute missing numeric values using **k-NN imputation (k=5)**.
- One-hot encode categorical columns.
- Standardize numeric features (mean=0, std=1).
- Log-transform `serum creatinine` for regression.
- Split data into **train/validation/test** using **10-fold cross-validation**.

### 2. Feature Grouping
Use three subsets from the paper:

| Feature Set | Description |
|--------------|--------------|
| **At-home** | Easy to measure (Age, Sex, Hypertension, etc.) |
| **Monitoring** | Clinic test features (Hemoglobin, Sodium, etc.) |
| **Laboratory** | Full set including urine and lab test results |

---

### 3. Data Visualization (with `pandas`, `matplotlib`, `seaborn`)
- Show missing values heatmap.  
- Plot histograms/boxplots for numerical features (e.g., BP, hemoglobin).  
- Correlation heatmap for numeric features.  
- CKD vs Not CKD count plot.  
- Feature distributions by CKD status.

---

### 4. Baseline Models
Implement **Random Forest** classifiers and regressors using `scikit-learn` for comparison.  
Report metrics: Accuracy, AUC (classification) and MSE, R² (regression).

---

## Deep Learning Implementations

### A. PyTorch
- Build a simple feedforward neural network:


- Use:
- **Binary Cross-Entropy** for classification  
- **MSE** for regression  
- Optimizer: Adam  
- Include early stopping and validation loss tracking.  
- Plot training vs validation loss.  
- Save best model per feature set.

### B. TensorFlow / Keras
- Recreate the same model architecture in TensorFlow.
- Use `ModelCheckpoint`, `EarlyStopping`, and `TensorBoard` for logging.
- Train and evaluate using same feature subsets and folds.
- Compare results directly with PyTorch.

---

## Evaluation
For each model (RF, PyTorch, TensorFlow):

| Task | Metrics |
|------|----------|
| Classification | Accuracy, AUC, TPR, TNR |
| Regression | MSE, MAE, R² |

- Plot ROC curves and confusion matrix for classification.
- Plot predicted vs actual creatinine (scatter plot) for regression.
- Compare PyTorch vs TensorFlow results side-by-side.

---

## Improvement Ideas (Optional)
Try one or more improvements:
- Add Batch Normalization or Dropout tuning.
- Use deeper networks (more layers or neurons).
- Combine classification + regression into one **dual-output model**.
- Use SMOTE or class weights for imbalanced data.
- Apply SHAP for explainability (top feature importance).

---

## Deliverables

| File | Description |
|------|--------------|
| `ckd_pytorch.ipynb` | PyTorch implementation notebook |
| `ckd_tensorflow.ipynb` | TensorFlow/Keras implementation notebook |
| `report.pdf` | 2–4 page summary of results & comparisons |
| `README.md` | How to run notebooks and reproduce results |

**Report should include:**
- Method summary  
- Results tables (RF, PyTorch, TensorFlow)  
- ROC and regression plots  
- Short discussion on differences between frameworks  

---


## Submission
 
- Submit as a zipped folder containing:
- Two notebooks  
- Report  
- Any generated plots or model files

---

## Expected Results
- PyTorch and TensorFlow models should both outperform the RF baseline for full laboratory features.  
- Results should improve from previous assignment through better preprocessing and tuning.  
- R² for regression expected to reach **~0.7**, classification accuracy **>95%** for laboratory features.
