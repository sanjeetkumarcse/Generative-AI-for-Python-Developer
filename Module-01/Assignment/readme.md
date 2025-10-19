# Assignment: Machine Learning for Chronic Kidney Disease Classification and Creatinine Prediction

## Objective
Reproduce and extend the experiments from the paper  
**“Machine Learning for Classifying Chronic Kidney Disease and Predicting Creatinine Levels Using At-Home Measurements”**  
(Metherall, Berryman & Brennan, *Scientific Reports*, 2025).

You will:
1. Implement CKD classification and creatinine regression using **Random Forest (RF)** and **Artificial Neural Network (ANN)** models.  
2. Evaluate model performance on three feature subsets — *at-home*, *monitoring*, and *laboratory* — using the **UCI Chronic Kidney Disease (CKD) dataset**.  
3. Perform **data preprocessing, feature engineering, and exploratory data visualization** using Python libraries.  
4. Compare your results with those reported in the paper and interpret feature importance and clinical relevance.

---

## Dataset
**Source:** [UCI Machine Learning Repository – Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)

dataset is avilabe in same folder.

**Description:**  
400 patient records with 25 clinical and demographic features.

**Labels:**  
- `classification` → CKD vs. Not CKD  
- `regression` → Serum creatinine level

---

## Tasks

### 1. Data Loading and Exploration
Use **pandas** to load and explore the dataset.

**Tasks:**
- Load the dataset (`.csv` or `.data` format) into a pandas DataFrame.  
- Display dataset shape, feature types, and missing value counts.  
- Show summary statistics (`.describe()` and `.info()`).  
- Check label distribution (CKD vs not CKD).  
- Identify and document categorical vs numerical features.

**Visualization:**
- Plot missing value heatmap using `matplotlib` or `seaborn`.  
- Visualize class imbalance (bar chart for CKD vs Not CKD).  
- Plot histograms and boxplots for key numerical features (e.g., blood pressure, hemoglobin, creatinine).  
- Create pairplots or correlation heatmaps to observe feature relationships.

---

### 2. Data Preprocessing

Perform the following steps to clean and prepare the dataset:

- Remove rows with missing `serum creatinine` values (≈17 samples).
- Impute missing **numerical** features using **k-NN imputation** (`k = 5`).
- Handle **categorical** features using **one-hot encoding**.
- Standardize numerical features (zero mean, unit variance).
- Log-transform `serum creatinine` (for regression task).
- Split the dataset into train and test folds using **10-fold cross-validation**.

**Deliverables:**
- Cleaned dataset summary (number of rows/columns, missing values).  
- Example of one-hot encoded feature output.  
- Standardization verification (mean ≈ 0, std ≈ 1).

---

### 3. Feature Engineering

Group the dataset features into the three sets described in the paper:

| Feature Set | Description | Example Features |
|--------------|--------------|------------------|
| **At-home** | Easily measurable at home or self-reported | Age, Sex, Race, Hypertension, Diabetes, Appetite, Pedal edema, Anemia, Blood pressure |
| **Monitoring** | Clinic-level basic test data | All at-home features + Blood urea, Blood glucose, Hemoglobin, Sodium, Potassium, RBC/WBC counts |
| **Laboratory** | Full set (comprehensive medical tests) | All monitoring features + Albumin, Sugar, Specific gravity, Pus cell, Packed cell volume, etc. |

**Tasks:**
- Create Python lists for each feature set.  
- Subset your DataFrame into the three corresponding groups.  
- Verify the feature counts match the paper (At-home: ~18, Monitoring: ~27, Laboratory: ~54 after encoding).  

---

### 4. Model Implementation

For each feature set (**At-home**, **Monitoring**, **Laboratory**):

#### (a) Classification Task
Predict whether a patient has **CKD** or not.

- **Algorithms:** Random Forest, Artificial Neural Network (Keras/TensorFlow)  
- **Loss function:** Binary Cross-Entropy  
- **Validation:** 10-fold Cross-Validation (90/10 split per fold, 20% of train used for validation)  
- **Metrics:** Accuracy, True Positive Rate (TPR), True Negative Rate (TNR), False Positive Rate (FPR), False Negative Rate (FNR), AUC  

#### (b) Regression Task
Predict **Serum Creatinine** levels.

- **Algorithms:** Random Forest Regressor, ANN Regressor  
- **Loss function:** Mean Squared Error (MSE)  
- **Metrics:** MSE, MAE, R²  

---

### 5. Hyperparameter Tuning

Use **Randomized Search** or **GridSearchCV** for parameter optimization.

| Model | Key Hyperparameters | Suggested Range |
|--------|----------------------|------------------|
| **ANN** | Hidden units, dropout rate, learning rate | 4–64, 0–0.5, 1e-4 – 1e-2 |
| **RF** | n_estimators, max_depth, min_samples_split, max_features, bootstrap | 100–2000, 10–200, {2, 5, 10}, {sqrt, log2, all}, {True, False} |

---

### 6. Evaluation & Visualization

#### Classification:
- Plot **ROC curves** for each feature set and model (use `matplotlib`).
- Calculate **AUC** for each curve.
- Compare your metrics to Table 3 from the paper.

#### Regression:
- Plot **predicted vs actual creatinine levels** (scatter plot).  
- Show residuals distribution (histogram).  
- Compare your metrics to Table 4 from the paper.

#### Feature Importance:
- Plot **top 10 important features** from Random Forest (bar chart).  
- Discuss the clinical meaning of the top predictors (e.g., hemoglobin, blood urea, hypertension).

---

## Deliverables

| Deliverable | Description |
|--------------|-------------|
| **Notebook (.ipynb)** | End-to-end implementation: preprocessing → feature engineering → training → evaluation → visualization |
| **Report (.pdf)** | 3–4 pages summarizing methods, results, and comparison with the paper |
| **Plots** | Correlation heatmap, ROC curves, feature importance bar charts, training vs validation curves |
| **Code Appendix** | Clean, well-commented Python code with explanations |

---

## Expected Outcomes
- RF should outperform ANN for **at-home** features (~92% accuracy).  
- Both models should reach **>98% accuracy** for **monitoring/laboratory** features.  
- Regression R² should improve from ≈0.38 (at-home) → ≈0.70 (laboratory).  
- Hemoglobin, blood urea, hypertension, and diabetes should emerge as key predictors.

---


## Submission
- **Deadline:** _(Instructor to specify)_  
- **Format:** Submit a zipped folder containing:  
  - Jupyter Notebook (`.ipynb`)  
  - Report (`.pdf`)  
  - Supporting files (`.csv`, generated plots, etc.)

---

### Reference
Metherall, B., Berryman, A. K., & Brennan, G. S. (2025).  
*Machine Learning for Classifying Chronic Kidney Disease and Predicting Creatinine Levels Using At-Home Measurements.*  
Scientific Reports, 15, 4364.  
[https://doi.org/10.1038/s41598-025-88631-y](https://doi.org/10.1038/s41598-025-88631-y)
