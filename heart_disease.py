# ==================================================================================
# IMPROVED PIPELINE — v2 + SHAP EXPLAINABILITY
# ==================================================================================

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

# ============================================================
# CONFIG
# ============================================================
N_RUNS        = 30
NP_POPULATION = 30
MAX_ITER      = 40
OPTUNA_TRIALS = 25

KAGGLE_CSV_PATH = "/content/cardio_train.csv"

# ============================================================
# SEMANTIC FEATURE BRIDGE
# ============================================================
UCI_TO_KAGGLE = {
    0: 0,   # age      ↔ age
    1: 1,   # sex      ↔ gender
    3: 4,   # trestbps ↔ ap_hi
    4: 6,   # chol     ↔ cholesterol
    5: 7,   # fbs      ↔ gluc
}
BRIDGE_UCI_COLS    = sorted(UCI_TO_KAGGLE.keys())
BRIDGE_KAGGLE_COLS = [UCI_TO_KAGGLE[k] for k in BRIDGE_UCI_COLS]
N_BRIDGE           = len(BRIDGE_UCI_COLS)

# Human-readable feature names for SHAP plots
UCI_FEATURE_NAMES = [
    "age", "sex", "chest pain type", "resting BP",
    "cholesterol", "fasting blood sugar", "resting ECG",
    "max heart rate", "exercise angina", "ST depression",
    "ST slope", "vessels colored", "thal"
]

# ============================================================
# DATA LOADERS
# ============================================================
def load_kaggle_data(path=KAGGLE_CSV_PATH):
    df = pd.read_csv(path, sep=";")
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    if df["age"].max() > 1000:
        df["age"] = (df["age"] / 365.25).round(1)
    df = df.apply(pd.to_numeric, errors="coerce")
    y = df["cardio"].values.astype(int)
    X = df.drop(columns=["cardio"]).values
    print(f"  Kaggle loaded: {X.shape[0]} rows, {X.shape[1]} features.")
    return X, y


def load_uci_data():
    try:
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "heart-disease/processed.cleveland.data")
        df = pd.read_csv(url, header=None)
    except Exception as e:
        print(f"⚠️  UCI unreachable ({e}). Falling back to mirror...")
        url = ("https://raw.githubusercontent.com/reinaldoq/processing-heart-disease-dataset/"
               "master/processed.cleveland.data")
        df = pd.read_csv(url, header=None)
    df = df.replace("?", np.nan)
    df.columns = ["age","sex","cp","trestbps","chol","fbs","restecg",
                  "thalach","exang","oldpeak","slope","ca","thal","target"]
    df["target"] = df["target"].apply(lambda x: 1 if int(x) > 0 else 0)
    df = df.apply(pd.to_numeric, errors="coerce")
    y = df["target"].values
    X = df.drop(columns=["target"]).values
    print(f"  UCI loaded:    {X.shape[0]} rows, {X.shape[1]} features.")
    return X, y


# ============================================================
# FITNESS FUNCTION
# ============================================================
def teacher_guided_fitness(mask, X, y, teacher_probs):
    if mask.sum() < 2:
        return 999
    sel = np.where(mask == 1)[0]
    X_sel = X[:, sel]
    model = xgb.XGBClassifier(n_estimators=30, max_depth=4, verbosity=0,
                               eval_metric="logloss")
    student_probs = cross_val_predict(
        model, X_sel, y, cv=3, method="predict_proba"
    )[:, 1]
    bce      = log_loss(y, student_probs)
    sparsity = mask.sum() / len(mask)
    temp     = 2
    distill  = mean_squared_error(
        teacher_probs ** (1/temp), student_probs ** (1/temp)
    )
    diversity = np.mean(np.abs(student_probs - teacher_probs))
    return 0.6*bce + 0.1*sparsity + 0.2*distill + 0.1*diversity


# ============================================================
# HHO (Lévy-flight enhanced)
# ============================================================
def levy_flight(n, beta=1.5):
    num   = math.gamma(1+beta) * np.sin(np.pi*beta/2)
    denom = math.gamma((1+beta)/2) * beta * 2**((beta-1)/2)
    sigma = (num/denom) ** (1/beta)
    u = np.random.randn(n) * sigma
    v = np.abs(np.random.randn(n))
    return u / (v ** (1/beta))


def run_hho(X, y, teacher_probs, seed):
    np.random.seed(seed)
    n   = X.shape[1]
    pop = np.random.randint(0, 2, (NP_POPULATION, n))
    for i in range(pop.shape[0]):
        if pop[i].sum() == 0:
            pop[i, np.random.randint(0, n)] = 1

    fitness  = np.array([teacher_guided_fitness(p, X, y, teacher_probs) for p in pop])
    best_idx = np.argmin(fitness)
    rabbit   = pop[best_idx].copy()

    for t in range(MAX_ITER):
        E0 = 2 * np.random.rand() - 1
        E  = 2 * E0 * (1 - t / MAX_ITER)

        for i in range(NP_POPULATION):
            new = pop[i].copy().astype(float)

            if abs(E) >= 1:
                q = np.random.rand()
                if q >= 0.5:
                    rand_hawk = pop[np.random.randint(0, NP_POPULATION)].astype(float)
                    new = rand_hawk - np.random.rand() * abs(
                        rand_hawk - 2 * np.random.rand() * pop[i])
                else:
                    new = (rabbit.astype(float)
                           - np.mean(pop, axis=0)
                           - np.random.rand() * (0.1 + np.random.rand() * 0.9)
                           * levy_flight(n))
            else:
                r = np.random.rand()
                if r >= 0.5 and abs(E) >= 0.5:
                    delta = rabbit.astype(float) - pop[i].astype(float)
                    new   = delta - E * abs(np.random.rand(n) * rabbit - pop[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    new = rabbit.astype(float) - E * abs(
                        rabbit.astype(float) - pop[i].astype(float))
                elif r < 0.5 and abs(E) >= 0.5:
                    Y = rabbit.astype(float) - E * abs(
                        np.random.rand(n) * rabbit - pop[i])
                    Z = Y + np.random.rand(n) * levy_flight(n)
                    new = Y if teacher_guided_fitness(
                        (Y > 0.5).astype(int), X, y, teacher_probs
                    ) < teacher_guided_fitness(
                        (Z > 0.5).astype(int), X, y, teacher_probs) else Z
                else:
                    Y = rabbit.astype(float) - E * abs(
                        np.random.rand(n) * rabbit - pop[i])
                    Z = Y + np.random.rand(n) * levy_flight(n)
                    new = Y if teacher_guided_fitness(
                        (Y > 0.5).astype(int), X, y, teacher_probs
                    ) < teacher_guided_fitness(
                        (Z > 0.5).astype(int), X, y, teacher_probs) else Z

            new_bin = (new > 0.5).astype(int)
            if new_bin.sum() == 0:
                new_bin[np.random.randint(0, n)] = 1

            f_new = teacher_guided_fitness(new_bin, X, y, teacher_probs)
            if f_new < fitness[i]:
                pop[i]     = new_bin
                fitness[i] = f_new
                if f_new < fitness[best_idx]:
                    rabbit   = new_bin.copy()
                    best_idx = i

    return rabbit


# ============================================================
# NGO
# ============================================================
def run_ngo(X, y, teacher_probs, seed):
    np.random.seed(seed + 100)
    n    = X.shape[1]
    mask = np.random.randint(0, 2, n)
    for _ in range(MAX_ITER):
        noise = np.random.rand(n)
        mask  = np.where(noise > 0.7, 1 - mask, mask)
    if mask.sum() == 0:
        mask[np.random.randint(0, n)] = 1
    return mask


# ============================================================
# OOA
# ============================================================
def run_ooa(X, y, teacher_probs, seed):
    np.random.seed(seed + 200)
    n    = X.shape[1]
    mask = np.random.randint(0, 2, n)
    for _ in range(MAX_ITER):
        idx       = np.random.choice(n, 2)
        mask[idx] = 1 - mask[idx]
    if mask.sum() == 0:
        mask[np.random.randint(0, n)] = 1
    return mask


# ============================================================
# DYNAMIC FUSION
# ============================================================
def dynamic_fusion(p_student, p_teacher, gate_threshold=0.25):
    teacher_conf   = np.abs(p_teacher - 0.5)
    student_uncert = 1 - 2 * np.abs(p_student - 0.5)
    gate  = (teacher_conf > gate_threshold).astype(float)
    alpha = gate * student_uncert * 0.35
    return (1 - alpha) * p_student + alpha * p_teacher


# ============================================================
# OPTUNA STUDENT TUNING
# ============================================================
def tune_student(X_tr, y_tr, X_val, y_val, seed):
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 400),
            max_depth        = trial.suggest_int("max_depth", 3, 7),
            learning_rate    = trial.suggest_float("lr", 0.01, 0.15, log=True),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample", 0.6, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 3.0),
            verbosity=0, eval_metric="logloss", seed=seed,
        )
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr)
        return log_loss(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess(X_train, X_val, X_test, y_train, seed):
    imp     = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_val   = imp.transform(X_val)
    X_test  = imp.transform(X_test)
    sm               = SMOTE(random_state=seed)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    sc      = RobustScaler()
    X_train = sc.fit_transform(X_train)
    X_val   = sc.transform(X_val)
    X_test  = sc.transform(X_test)
    return X_train, X_val, X_test, y_train, imp, sc


# ============================================================
# SHAP ANALYSIS
# Called once after the 30-run loop using the last run's models.
# Saves 3 plots per algorithm to /content/:
#   1. shap_summary_{name}.png   — bar chart + beeswarm side by side
#   2. shap_waterfall_{name}.png — single patient breakdown
# Also prints a cross-algorithm consensus table to console.
# ============================================================
def run_shap_analysis(shap_store):
    print("\n" + "="*62)
    print("SHAP EXPLAINABILITY")
    print("="*62)

    for name, store in shap_store.items():
        sv         = store["shap_values"]   # (n_test, n_selected_features)
        X_te       = store["X_te"]
        y_test     = store["y_test"]
        feat_names = store["feat_names"]
        fusion_p   = store["fusion_probs"]
        explainer  = store["explainer"]
        n_sel      = len(feat_names)

        print(f"\n── {name}  ({n_sel} features selected) ──")
        print(f"   Features: {feat_names}")

        # ── 1. Summary plot: bar + beeswarm ──────────────────────
        mean_abs = np.abs(sv).mean(axis=0)
        order    = np.argsort(mean_abs)[::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4, n_sel * 0.6 + 1.5)))
        fig.suptitle(f"SHAP — {name} student  |  {n_sel} features selected",
                     fontsize=13, fontweight="bold", y=1.01)

        # Left: mean |SHAP| bar chart
        ax1.barh(
            [feat_names[i] for i in order],
            [mean_abs[i]   for i in order],
            color="#1D9E75", edgecolor="none", height=0.6
        )
        ax1.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax1.set_title("Global feature importance", fontsize=11)
        ax1.invert_yaxis()
        ax1.spines[["top","right"]].set_visible(False)
        ax1.tick_params(labelsize=9)

        # Right: beeswarm — each dot = one patient
        # Color = normalised raw feature value (red=high, green=low)
        for rank, fi in enumerate(order):
            vals    = sv[:, fi]
            fv_norm = (X_te[:, fi] - X_te[:, fi].min()) / (np.ptp(X_te[:, fi]) + 1e-9)
            ax2.scatter(
                vals,
                np.full_like(vals, rank) + np.random.uniform(-0.2, 0.2, len(vals)),
                c=fv_norm, cmap="RdYlGn_r", alpha=0.65, s=18, linewidths=0
            )
        ax2.set_yticks(range(len(order)))
        ax2.set_yticklabels([feat_names[i] for i in order], fontsize=9)
        ax2.axvline(0, color="#888", lw=0.8, linestyle="--")
        ax2.set_xlabel("SHAP value  (+ = pushes toward disease)", fontsize=10)
        ax2.set_title("Per-patient impact\n(red=high value, green=low value)", fontsize=11)
        ax2.spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        path_summary = f"/content/shap_summary_{name}.png"
        fig.savefig(path_summary, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {path_summary}")

        # ── 2. Waterfall — most confident correct prediction ─────
        preds       = (fusion_p > 0.5).astype(int)
        correct_idx = np.where(preds == y_test)[0]
        if len(correct_idx) > 0:
            chosen = correct_idx[np.argmax(np.abs(fusion_p[correct_idx] - 0.5))]
        else:
            chosen = 0

        true_label   = "disease" if y_test[chosen] == 1 else "healthy"
        fusion_score = fusion_p[chosen]

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])

        exp = shap.Explanation(
            values        = sv[chosen],
            base_values   = base_val,
            data          = X_te[chosen],
            feature_names = feat_names,
        )

        fig2, _ = plt.subplots(figsize=(9, max(4, n_sel * 0.6 + 2)))
        shap.waterfall_plot(exp, max_display=n_sel, show=False)
        fig2.suptitle(
            f"{name} — patient #{chosen}  |  true: {true_label}  "
            f"|  fusion score: {fusion_score:.3f}",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        path_wf = f"/content/shap_waterfall_{name}.png"
        fig2.savefig(path_wf, dpi=130, bbox_inches="tight")
        plt.close(fig2)
        print(f"   Saved: {path_wf}")

        # ── 3. Console importance ranking ────────────────────────
        print(f"\n   Feature ranking by mean |SHAP|:")
        for rank, fi in enumerate(order, 1):
            direction = "↑ risk" if sv[:, fi].mean() > 0 else "↓ risk"
            print(f"   {rank}. {feat_names[fi]:<22}  "
                  f"|SHAP|={mean_abs[fi]:.4f}   avg direction: {direction}")

    # ── Cross-algorithm consensus table ──────────────────────────
    print("\n" + "="*62)
    print("CROSS-ALGORITHM FEATURE SELECTION CONSENSUS")
    print("="*62)
    all_names = sorted({f for s in shap_store.values() for f in s["feat_names"]})
    rows = []
    for fname in all_names:
        row = {"feature": fname}
        for alg, store in shap_store.items():
            if fname in store["feat_names"]:
                fi  = store["feat_names"].index(fname)
                imp = float(np.abs(store["shap_values"][:, fi]).mean())
                row[alg] = f"YES  ({imp:.3f})"
            else:
                row[alg] = "—"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("feature").to_string())
    print("\nAll SHAP plots saved to /content/")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("Loading datasets...")
    X_kaggle, y_kaggle = load_kaggle_data()
    X_uci,    y_uci    = load_uci_data()

    # Preprocess Kaggle (teacher)
    imp_kag  = SimpleImputer(strategy="median")
    X_kaggle = imp_kag.fit_transform(X_kaggle)
    sm_kag             = SMOTE(random_state=42)
    X_kaggle, y_kaggle = sm_kag.fit_resample(X_kaggle, y_kaggle)
    sc_kag   = RobustScaler()
    X_kaggle = sc_kag.fit_transform(X_kaggle)

    # Train teacher
    print("\nTraining teacher on Kaggle data...")
    teacher_base = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=42, verbosity=0, eval_metric="logloss"
    )
    teacher = CalibratedClassifierCV(teacher_base, method="isotonic", cv=3)
    teacher.fit(X_kaggle, y_kaggle)
    print("  Teacher ready.\n")

    algorithms = {"HHO": run_hho, "NGO": run_ngo, "OOA": run_ooa}
    results    = {k: [] for k in algorithms}

    # Populated on the final run for SHAP analysis
    shap_store = {}

    for run in range(N_RUNS):
        print(f"RUN {run + 1}/{N_RUNS}")

        # Split UCI
        X_tr_full, X_test, y_tr_full, y_test = train_test_split(
            X_uci, y_uci, test_size=0.2, stratify=y_uci, random_state=run
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=0.25,
            stratify=y_tr_full, random_state=run
        )

        # Preprocess UCI splits
        X_train, X_val, X_test, y_train, imp_uci, sc_uci = preprocess(
            X_train, X_val, X_test, y_train, run
        )

        # Semantic bridge: project UCI → Kaggle feature space
        def project_uci_to_kaggle(X_uci_scaled):
            X_proj = np.zeros((X_uci_scaled.shape[0], X_kaggle.shape[1]))
            for uci_col, kag_col in zip(BRIDGE_UCI_COLS, BRIDGE_KAGGLE_COLS):
                X_proj[:, kag_col] = X_uci_scaled[:, uci_col]
            return X_proj

        p_teach_train = teacher.predict_proba(project_uci_to_kaggle(X_train))[:, 1]
        p_teach_test  = teacher.predict_proba(project_uci_to_kaggle(X_test))[:,  1]

        # Feature selection + student
        for name, algo in algorithms.items():

            mask = algo(X_train, y_train, p_teach_train, run)
            idx  = np.where(mask == 1)[0]
            if len(idx) == 0:
                idx = np.arange(X_train.shape[1])

            X_tr = X_train[:, idx]
            X_te = X_test[:,  idx]
            X_v  = X_val[:,   idx]

            best_params = tune_student(X_tr, y_train, X_v, y_val, seed=run)

            student = xgb.XGBClassifier(
                n_estimators     = best_params["n_estimators"],
                max_depth        = best_params["max_depth"],
                learning_rate    = best_params["lr"],
                subsample        = best_params["subsample"],
                colsample_bytree = best_params["colsample"],
                min_child_weight = best_params["min_child_weight"],
                reg_alpha        = best_params["reg_alpha"],
                reg_lambda       = best_params["reg_lambda"],
                verbosity=0, eval_metric="logloss", seed=run,
            )
            student.fit(X_tr, y_train)

            p_stud_test   = np.clip(student.predict_proba(X_te)[:, 1], 1e-6, 1-1e-6)
            p_teach_test_ = np.clip(p_teach_test, 1e-6, 1-1e-6)
            fusion        = dynamic_fusion(p_stud_test, p_teach_test_)

            acc = accuracy_score(y_test, (fusion > 0.5).astype(int))
            results[name].append(acc)

            if run == N_RUNS - 1:
                stud_acc  = accuracy_score(y_test, (p_stud_test   > 0.5))
                teach_acc = accuracy_score(y_test, (p_teach_test_  > 0.5))
                print(f"  {name:3s} | features={idx.shape[0]:2d} | "
                      f"Student={stud_acc:.4f}  "
                      f"Teacher(bridge)={teach_acc:.4f}  "
                      f"Fusion={acc:.4f}")

                # Save everything needed for SHAP
                explainer = shap.TreeExplainer(student)
                shap_store[name] = {
                    "explainer"   : explainer,
                    "shap_values" : explainer.shap_values(X_te),
                    "X_te"        : X_te,
                    "y_test"      : y_test,
                    "feat_names"  : [UCI_FEATURE_NAMES[i] for i in idx],
                    "fusion_probs": fusion,
                }

    print("\n" + "="*62)
    print("FINAL RESULTS  (accuracy over 30 runs)")
    print("="*62)
    df_res = pd.DataFrame(results)
    print(df_res.describe().round(4))
    print()
    print("Improvement over v1 baseline (HHO mean 0.7279):")
    for col in df_res.columns:
        delta = df_res[col].mean() - 0.7279
        sign  = "+" if delta >= 0 else ""
        print(f"  {col}: {df_res[col].mean():.4f}  ({sign}{delta:.4f})")

    # Run SHAP analysis on the final run's models
    run_shap_analysis(shap_store)
