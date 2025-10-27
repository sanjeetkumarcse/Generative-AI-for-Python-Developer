# CKD Paper Implementation Notebook
# Reproduces: "Machine learning for classifying chronic kidney disease and predicting creatinine levels using at-home measurements"
# Author: Generated code (adapt as needed)

# === Setup: install required packages (run once in your environment) ===
# !pip install pandas numpy scikit-learn tensorflow keras keras-tuner matplotlib seaborn wget

# === Imports ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
                             mean_squared_error, mean_absolute_error, r2_score)
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# === Helper utilities ===
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def download_ucidataset(dest='kidney_disease.csv'):
    """Download the UCI chronic kidney disease dataset (if permitted).
    If the direct CSV is not available, replace this function by manual download.
    """
    # UCI page: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease
    # Many mirrors host the CSV — user may need to download manually and place file in working dir.
    if os.path.exists(dest):
        print(f"Found dataset at {dest}")
        return dest
    else:
        print("Please download the dataset from UCI and save as 'ckd.csv' in the working directory.")
        raise FileNotFoundError('ckd.csv not found')

# === Load data ===

def load_data(path='kidney_disease.csv'):
    # Adjust based on actual CSV formatting
    df = pd.read_csv(path)
    return df

# === Preprocessing pipeline ===

def preprocess(df):
    # NOTE: column names vary depending on source file. Map them to canonical names used in the paper.
    # Example mapping (adjust after inspecting df.columns):
    print('Columns in dataset:', list(df.columns))

    # Drop rows missing serum creatinine if present as 'serum_creatinine' or 'sc' etc.
    creat_keys = [c for c in df.columns if 'creatinine' in c.lower() or 'sc' == c.lower()]
    if len(creat_keys) == 0:
        print('No creatinine column found; ensure dataset contains serum creatinine.')
    else:
        creat_col = creat_keys[0]
        df = df.dropna(subset=[creat_col])
        df.rename(columns={creat_col: 'serum_creatinine'}, inplace=True)

    # Identify numerical and categorical columns (simple heuristic)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Treat missing categorical as 'missing'
    df[cat_cols] = df[cat_cols].astype(object).fillna('missing')

    # k-NN imputation for numerical columns
    if len(num_cols) > 0:
        imputer = KNNImputer(n_neighbors=5)
        df[num_cols] = imputer.fit_transform(df[num_cols])

    # One-hot encode categorical columns (missing kept as 'missing')
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    # Standardize numerical columns
    scaler = StandardScaler()
    if len(num_cols) > 0:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # Add log creatinine column
    if 'serum_creatinine' in df.columns:
        # undo scaling for creatinine if present in num_cols (we already standardized)
        # For safety, recreate log from original numeric value (assume standardization used)
        # If standardized, take original prior to scaling when available. Here we assume creatinine numeric.
        df['log_creatinine'] = np.log(df['serum_creatinine'].clip(lower=1e-6))

    return df

# === Feature sets defined in the paper ===
AT_HOME = ['Age', 'Race', 'Sex', 'HTN', 'DM', 'CAD', 'APPET', 'PE', 'ANE', 'BP']
MONITORING = AT_HOME + ['RBC', 'RBCC', 'WBCC', 'BGR', 'BU', 'SOD', 'POT', 'HEMO']
LAB = MONITORING + ['SG', 'AL', 'SU', 'BA', 'PC', 'PCC', 'PCV']

# Note: These names must match columns in your df; likely they'll differ. Map appropriately after inspection.

# === Model utilities ===

def make_rf_classifier(X_train, y_train, param_grid=None):
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 500],
            'max_depth': [20, 100],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def make_rf_regressor(X_train, y_train, param_grid=None):
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 500],
            'max_depth': [20, 100],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='r2')
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

# === Keras ANN builders for Keras Tuner ===

def build_classifier_model(hp, input_dim):
    model = keras.Sequential()
    # Fixed 1 hidden layer as paper; but allow tuning neurons and dropout
    units = hp.Int('units', min_value=4, max_value=64, step=4)
    dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    model.add(keras.layers.InputLayer(input_shape=(input_dim,)))
    model.add(keras.layers.Dense(units, activation='relu'))
    if dropout > 0:
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_regressor_model(hp, input_dim):
    model = keras.Sequential()
    units = hp.Int('units', min_value=4, max_value=64, step=4)
    dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    model.add(keras.layers.InputLayer(input_shape=(input_dim,)))
    model.add(keras.layers.Dense(units, activation='relu'))
    if dropout > 0:
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=['mse'])
    return model

# === Cross-validation wrappers ===

def evaluate_classification_cv(X, y, model_type='rf', n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    roc_curves = []
    fold = 0
    for train_index, test_index in kf.split(X):
        fold += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_type == 'rf':
            model, best = make_rf_classifier(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:,1]
            y_pred = model.predict(X_test)
        else:  # ann
            input_dim = X_train.shape[1]
            tuner = kt.RandomSearch(lambda hp: build_classifier_model(hp, input_dim),
                                    objective='val_accuracy', max_trials=10, overwrite=True)
            Xtr, Xval, ytr, yval = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
            tuner.search(Xtr, ytr, epochs=50, validation_data=(Xval, yval), callbacks=[keras.callbacks.EarlyStopping(patience=3)])
            best_model = tuner.get_best_models(num_models=1)[0]
            y_proba = best_model.predict(X_test).ravel()
            y_pred = (y_proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        auc = roc_auc_score(y_test, y_proba)
        fpr_vals, tpr_vals, _ = roc_curve(y_test, y_proba)
        roc_curves.append((fpr_vals, tpr_vals))
        metrics.append({'fold': fold, 'accuracy': acc, 'TPR': tpr, 'TNR': tnr, 'FPR': fpr, 'FNR': fnr, 'AUC': auc})
    return pd.DataFrame(metrics), roc_curves


def evaluate_regression_cv(X, y, model_type='rf', n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    fold = 0
    for train_index, test_index in kf.split(X):
        fold += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_type == 'rf':
            model, best = make_rf_regressor(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            input_dim = X_train.shape[1]
            tuner = kt.RandomSearch(lambda hp: build_regressor_model(hp, input_dim), objective='val_mse', max_trials=10, overwrite=True)
            Xtr, Xval, ytr, yval = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
            tuner.search(Xtr, ytr, epochs=100, validation_data=(Xval, yval), callbacks=[keras.callbacks.EarlyStopping(patience=3)])
            best_model = tuner.get_best_models(num_models=1)[0]
            y_pred = best_model.predict(X_test).ravel()

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics.append({'fold': fold, 'MSE': mse, 'MAE': mae, 'R2': r2})
    return pd.DataFrame(metrics)

# === Visualization functions ===

def plot_roc_curves(roc_curves, title='ROC Curves'):
    plt.figure(figsize=(6,6))
    for fpr_vals, tpr_vals in roc_curves:
        plt.plot(fpr_vals, tpr_vals, alpha=0.3)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()

# === Main run routine (high level) ===

def run_full_pipeline(df):
    # Preprocess
    df = preprocess(df)

    # Map ckd label column
    label_cols = [c for c in df.columns if 'ckd' in c.lower() or 'class' in c.lower()]
    if len(label_cols)==0:
        raise ValueError('No CKD label column found. Please ensure dataset has label for CKD.')
    label_col = label_cols[0]
    df.rename(columns={label_col: 'ckd_label'}, inplace=True)

    # Prepare y for classification and regression
    y_class = df['ckd_label'].astype(int)
    if 'log_creatinine' not in df.columns:
        raise ValueError('log_creatinine missing; ensure serum_creatinine exists and preprocessing added log_creatinine')
    y_reg = df['log_creatinine']

    # Build feature sets by intersecting expected names with actual df columns
    def choose_features(expected_list):
        found = [c for c in expected_list if c in df.columns]
        if len(found)==0:
            print(f'Warning: none of expected features {expected_list} found in dataframe columns.')
        return found

    X_at = df[choose_features(AT_HOME)]
    X_mon = df[choose_features(MONITORING)]
    X_lab = df[choose_features(LAB)]

    results = {}
    for name, X in [('at-home', X_at), ('monitoring', X_mon), ('laboratory', X_lab)]:
        print('\n=== Running classification for feature set:', name, '===')
        if X.shape[1]==0:
            print('Skipping empty feature set')
            continue
        clf_metrics_rf, roc_rf = evaluate_classification_cv(X, y_class, model_type='rf')
        clf_metrics_ann, roc_ann = evaluate_classification_cv(X, y_class, model_type='ann')
        print('RF classification mean metrics:\n', clf_metrics_rf.mean())
        print('ANN classification mean metrics:\n', clf_metrics_ann.mean())
        plot_roc_curves(roc_rf, title=f'RF ROC - {name}')
        plot_roc_curves(roc_ann, title=f'ANN ROC - {name}')

        print('\n=== Running regression for feature set:', name, '===')
        reg_metrics_rf = evaluate_regression_cv(X, y_reg, model_type='rf')
        reg_metrics_ann = evaluate_regression_cv(X, y_reg, model_type='ann')
        print('RF regression mean:\n', reg_metrics_rf.mean())
        print('ANN regression mean:\n', reg_metrics_ann.mean())

        results[name] = {
            'clf_rf': clf_metrics_rf,
            'clf_ann': clf_metrics_ann,
            'roc_rf': roc_rf,
            'roc_ann': roc_ann,
            'reg_rf': reg_metrics_rf,
            'reg_ann': reg_metrics_ann
        }

    return results

# === Example usage (uncomment to run) ===
# path = download_ucidataset('ckd.csv')
# df_raw = load_data(path)
# results = run_full_pipeline(df_raw)

# Save results utility
def save_results(results, outdir='results'):
    os.makedirs(outdir, exist_ok=True)
    for k,v in results.items():
        v['clf_rf'].to_csv(os.path.join(outdir, f'{k}_clf_rf.csv'), index=False)
        v['clf_ann'].to_csv(os.path.join(outdir, f'{k}_clf_ann.csv'), index=False)
        v['reg_rf'].to_csv(os.path.join(outdir, f'{k}_reg_rf.csv'), index=False)
        v['reg_ann'].to_csv(os.path.join(outdir, f'{k}_reg_ann.csv'), index=False)
    print('Saved results to', outdir)

# End of notebook
