# ============================================================
# HEART DISEASE PREDICTION — PUBLICATION CODE (WITH SHAP)
# Optimized for Reliability, Speed, and Standard Metrics
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Standard Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, auc)
import xgboost as xgb
import optuna
import shap  # <--- NEW IMPORT

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_PATH = "/content/cardio_train.csv" # Check this path
RANDOM_STATE = 42
POP_SIZE = 15           # HHO Population
MAX_ITER = 10           # HHO Iterations
ALPHA_PENALTY = 0.005   # Feature selection penalty
OPTUNA_TRIALS = 30      # Trials
MIN_FEATURES = 6
np.random.seed(RANDOM_STATE)

# ============================================================
# 2. DATA LOADING & SCIENTIFIC CLEANING
# ============================================================
print("--- 1. Data Prep & Cleaning ---")
data = pd.read_csv(DATA_PATH, sep=';')

# Drop ID
if "id" in data.columns:
    data = data.drop(columns=["id"])

# Convert Age
data["age"] = data["age"] / 365.25

# Feature Engineering
data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']

# Filter Outliers (Physiologically Impossible Values)
mask_bp = (data['ap_hi'] >= 50) & (data['ap_hi'] <= 250) & \
          (data['ap_lo'] >= 30) & (data['ap_lo'] <= 160) & \
          (data['ap_hi'] > data['ap_lo'])
mask_body = (data['height'] >= 100) & (data['height'] <= 250) & \
            (data['weight'] >= 40) & (data['weight'] <= 200)

data = data[mask_bp & mask_body]
print(f"Cleaned Data Shape: {data.shape}")

# Target & Features
target_col = "cardio"
feature_names = [c for c in data.columns if c != target_col]
X = data[feature_names].values
y = data[target_col].astype(int).values

# ============================================================
# 3. EXTERNAL VALIDATION SPLIT
# ============================================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. BASELINE MODELS
# ============================================================
print("\n--- 2. Training Baseline Models ---")
models = {}

# A. Logistic Regression
lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
lr.fit(X_train_scaled, y_train_full)
models['Logistic Regression'] = lr

# B. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_train_full)
models['Random Forest'] = rf

# C. XGBoost (Default)
xgb_def = xgb.XGBClassifier(objective="binary:logistic", random_state=RANDOM_STATE, verbosity=0)
xgb_def.fit(X_train_scaled, y_train_full)
models['XGBoost (Default)'] = xgb_def

# ============================================================
# 5. FEATURE SELECTION (HHO)
# ============================================================
print("\n--- 3. HHO Feature Selection ---")
# Subset for speed
sample_idx = np.random.choice(len(X_train_scaled), 10000, replace=False)
X_hho = X_train_scaled[sample_idx]
y_hho = y_train_full[sample_idx]

def fitness(mask, Xsub, y):
    if mask.sum() < 2: return 0
    Xs = Xsub[:, mask == 1]
    clf = xgb.XGBClassifier(n_estimators=30, max_depth=3, verbosity=0, n_jobs=-1, random_state=RANDOM_STATE)
    score = cross_val_score(clf, Xs, y, cv=3, scoring="roc_auc").mean()
    penalty = ALPHA_PENALTY * (mask.sum() / len(mask))
    return score - penalty

n_feat = X_hho.shape[1]
pop = np.random.randint(0, 2, (POP_SIZE, n_feat))
fitnesses = np.array([fitness(p, X_hho, y_hho) for p in pop])
best_idx = np.argmax(fitnesses)
best_rabbit = pop[best_idx].copy()
best_fit = fitnesses[best_idx]

for t in range(MAX_ITER):
    E1 = 2 * (1 - (t / MAX_ITER))
    for i in range(POP_SIZE):
        E0 = 2 * np.random.rand() - 1
        Escaping_Energy = E1 * E0

        if abs(Escaping_Energy) >= 1:
            q = np.random.rand()
            rand_hawk = pop[np.random.randint(0, POP_SIZE)]
            if q < 0.5:
                pop[i] = rand_hawk - np.random.rand() * abs(rand_hawk - 2 * np.random.rand() * pop[i])
            else:
                pop[i] = (best_rabbit - pop[i].mean(axis=0)) - np.random.rand() * ((np.random.rand()) + 0)
        else:
            diff = best_rabbit - pop[i]
            pop[i] = best_rabbit - Escaping_Energy * diff

        s = 1 / (1 + np.exp(-10 * (pop[i] - 0.5)))
        pop[i] = (np.random.rand(n_feat) < s).astype(int)

        new_fit = fitness(pop[i], X_hho, y_hho)
        if new_fit > fitnesses[i]:
            fitnesses[i] = new_fit
            if new_fit > best_fit:
                best_fit = new_fit
                best_rabbit = pop[i].copy()

sel_indices = np.where(best_rabbit == 1)[0]
final_features = [feature_names[i] for i in sel_indices]
for f in ['ap_hi', 'age', 'bmi']:
    if f not in final_features: final_features.append(f)
print(f"Selected Features: {final_features}")

feat_indices = [feature_names.index(f) for f in final_features]
X_train_sel = X_train_scaled[:, feat_indices]
X_test_sel = X_test_scaled[:, feat_indices]

# ============================================================
# 6. OPTUNA TUNING
# ============================================================
print("\n--- 4. Optuna Tuning ---")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
    }
    clf = xgb.XGBClassifier(objective="binary:logistic", verbosity=0, n_jobs=-1, random_state=RANDOM_STATE, **params)
    return cross_val_score(clf, X_train_sel, y_train_full, cv=3, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=OPTUNA_TRIALS)

best_xgb = xgb.XGBClassifier(objective="binary:logistic", verbosity=0, random_state=RANDOM_STATE, **study.best_params)
best_xgb.fit(X_train_sel, y_train_full)
models['Proposed Method (HHO+Optuna)'] = best_xgb

# ============================================================
# 7. METRICS & PLOTTING (WITH SHAP)
# ============================================================
print("\n--- 5. Generating Results & Images ---")

results_list = []
plt.figure(figsize=(10, 8))

for name, model in models.items():
    if name == 'Proposed Method (HHO+Optuna)':
        X_eval = X_test_sel
    else:
        X_eval = X_test_scaled

    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    rec = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)

    results_list.append({
        "Model": name, "Accuracy": acc, "AUC-ROC": auc_score,
        "Sensitivity": rec, "Specificity": spec, "F1-Score": f1
    })

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

# 1. Save ROC Curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('roc_curve.png', dpi=300)
print("Saved: roc_curve.png")
plt.show()

# 2. Save Confusion Matrix
plt.figure(figsize=(6, 5))
conf_matrix = confusion_matrix(y_test, best_xgb.predict(X_test_sel))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix (Proposed Method)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('confusion_matrix.png', dpi=300)
print("Saved: confusion_matrix.png")
plt.show()

# 3. Save Optuna Optimization History
plt.figure(figsize=(10, 6))
trials = study.trials_dataframe()
plt.plot(trials['number'], trials['value'], marker='o', linestyle='-', color='tab:orange')
plt.title('Optuna Optimization History (ROC-AUC)')
plt.xlabel('Number of Trials')
plt.ylabel('Objective Value (ROC-AUC)')
plt.grid(True, alpha=0.5)
plt.savefig('optuna_history.png', dpi=300)
print("Saved: optuna_history.png")
plt.show()

# ============================================================
# 8. SHAP EXPLAINABILITY (NEW SECTION)
# ============================================================
print("\n--- 6. Generating SHAP Diagrams ---")

# Convert Test Set to DataFrame for better labeling in plots
X_test_shap_df = pd.DataFrame(X_test_sel, columns=final_features)

# Create Explainer
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test_shap_df)

# A. SHAP Summary Plot (Beeswarm)
plt.figure()
plt.title("SHAP Summary Plot")
shap.summary_plot(shap_values, X_test_shap_df, show=False)
plt.savefig('shap_summary.png', bbox_inches='tight', dpi=300)
print("Saved: shap_summary.png")
plt.close()

# B. SHAP Importance Bar Plot
plt.figure()
plt.title("SHAP Feature Importance")
shap.summary_plot(shap_values, X_test_shap_df, plot_type="bar", show=False)
plt.savefig('shap_importance.png', bbox_inches='tight', dpi=300)
print("Saved: shap_importance.png")
plt.close()

# Print Final Table
results_df = pd.DataFrame(results_list).sort_values(by="AUC-ROC", ascending=False)
print("\n=== FINAL RESULTS TABLE ===")
print(results_df.round(4).to_string(index=False))
