# ==================================================================================
# IMPROVED PIPELINE — v2
# Fixes:  1) Confidence-gated fusion  (teacher only helps when it's actually sure)
#         2) Proper HHO               (energy-based exploration / exploitation)
#         3) Optuna student tuning    (per-run hyperparameter search)
#         4) Semantic feature bridge  (map UCI ↔ Kaggle by meaning, not position)
# ==================================================================================

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import xgboost as xgb
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
NP_POPULATION = 30        # ↑ from 20
MAX_ITER      = 40        # ↑ from 15
OPTUNA_TRIALS = 25        # student tuning trials per run

KAGGLE_CSV_PATH = "/content/cardio_train.csv"

# ============================================================
# SEMANTIC FEATURE BRIDGE
# ============================================================
# UCI has 13 features. Kaggle (cardio) has 11 features.
# Instead of blindly aligning by column position, we hand-map
# the semantically closest features between the two datasets.
#
# UCI columns (index → name):
#   0  age         1  sex        2  cp (chest pain)   3  trestbps (rest BP)
#   4  chol        5  fbs        6  restecg            7  thalach (max HR)
#   8  exang       9  oldpeak   10  slope             11  ca      12  thal
#
# Kaggle (cardio) columns (index → name, after dropping id):
#   0  age   1  gender   2  height   3  weight   4  ap_hi (systolic BP)
#   5  ap_lo 6  cholesterol  7  gluc   8  smoke   9  alco  10  active
#
# Best semantic matches  (UCI_idx → Kaggle_idx):
UCI_TO_KAGGLE = {
    0: 0,   # age         ↔ age
    1: 1,   # sex         ↔ gender
    3: 4,   # trestbps    ↔ ap_hi (systolic BP)
    4: 6,   # chol        ↔ cholesterol
    5: 7,   # fbs         ↔ gluc (both binary metabolic markers)
}
# UCI features NOT in the bridge are excluded from the teacher projection
# (they have no semantic match in cardio_train).
BRIDGE_UCI_COLS    = sorted(UCI_TO_KAGGLE.keys())        # [0,1,3,4,5]
BRIDGE_KAGGLE_COLS = [UCI_TO_KAGGLE[k] for k in BRIDGE_UCI_COLS]

N_BRIDGE = len(BRIDGE_UCI_COLS)   # 5 shared features


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
# FIX 2 — PROPER HHO
# (energy-based soft/hard besiege + Lévy flight exploration)
# ============================================================
def levy_flight(n, beta=1.5):
    """Generate a Lévy-distributed step for n dimensions."""
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

    # Guarantee no all-zero rows
    for i in range(pop.shape[0]):
        if pop[i].sum() == 0:
            pop[i, np.random.randint(0, n)] = 1

    fitness = np.array([teacher_guided_fitness(p, X, y, teacher_probs)
                        for p in pop])
    best_idx = np.argmin(fitness)
    rabbit   = pop[best_idx].copy()

    for t in range(MAX_ITER):
        # Escaping energy decreases from 2 → 0 over iterations
        E0 = 2 * np.random.rand() - 1        # initial energy
        E  = 2 * E0 * (1 - t / MAX_ITER)     # current energy

        for i in range(NP_POPULATION):
            new = pop[i].copy().astype(float)

            if abs(E) >= 1:
                # ── EXPLORATION (far from rabbit) ──────────────────
                q = np.random.rand()
                if q >= 0.5:
                    # Random jump
                    rand_hawk = pop[np.random.randint(0, NP_POPULATION)].astype(float)
                    new = rand_hawk - np.random.rand() * abs(
                        rand_hawk - 2 * np.random.rand() * pop[i]
                    )
                else:
                    # Perch near rabbit with Lévy step
                    new = (rabbit.astype(float)
                           - np.mean(pop, axis=0)
                           - np.random.rand() * (
                               0.1 + np.random.rand() * 0.9   # [0.1, 1.0] range
                           ) * levy_flight(n))

            else:
                # ── EXPLOITATION (close to rabbit) ─────────────────
                r = np.random.rand()
                if r >= 0.5 and abs(E) >= 0.5:
                    # Soft besiege
                    delta = rabbit.astype(float) - pop[i].astype(float)
                    new   = delta - E * abs(
                        np.random.rand(n) * rabbit - pop[i]
                    )
                elif r >= 0.5 and abs(E) < 0.5:
                    # Hard besiege
                    new = rabbit.astype(float) - E * abs(
                        rabbit.astype(float) - pop[i].astype(float)
                    )
                elif r < 0.5 and abs(E) >= 0.5:
                    # Soft besiege with rapid dives (Lévy)
                    Y = rabbit.astype(float) - E * abs(
                        np.random.rand(n) * rabbit - pop[i]
                    )
                    Z = Y + np.random.rand(n) * levy_flight(n)
                    new = Y if teacher_guided_fitness(
                        (Y > 0.5).astype(int), X, y, teacher_probs
                    ) < teacher_guided_fitness(
                        (Z > 0.5).astype(int), X, y, teacher_probs
                    ) else Z
                else:
                    # Hard besiege with rapid dives
                    Y = rabbit.astype(float) - E * abs(
                        np.random.rand(n) * rabbit - pop[i]
                    )
                    Z = Y + np.random.rand(n) * levy_flight(n)
                    new = Y if teacher_guided_fitness(
                        (Y > 0.5).astype(int), X, y, teacher_probs
                    ) < teacher_guided_fitness(
                        (Z > 0.5).astype(int), X, y, teacher_probs
                    ) else Z

            # Binarise
            new_bin = (new > 0.5).astype(int)
            if new_bin.sum() == 0:
                new_bin[np.random.randint(0, n)] = 1

            f_new = teacher_guided_fitness(new_bin, X, y, teacher_probs)
            if f_new < fitness[i]:
                pop[i]    = new_bin
                fitness[i] = f_new
                if f_new < fitness[best_idx]:
                    rabbit   = new_bin.copy()
                    best_idx = i

    return rabbit


# ============================================================
# NGO  (unchanged — reasonable baseline)
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
# OOA  (unchanged — reasonable baseline)
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
# FIX 1 — CONFIDENCE-GATED FUSION
# Problem: teacher is ~0.59 cross-domain → blind blending HURTS.
# Fix:     only blend when teacher is genuinely confident AND
#          student is genuinely uncertain. Elsewhere trust student.
# ============================================================
def dynamic_fusion(p_student, p_teacher, gate_threshold=0.25):
    """
    gate_threshold: teacher must be at least this far from 0.5
                    to be allowed to influence the prediction.
    """
    teacher_conf   = np.abs(p_teacher - 0.5)          # 0 = uncertain, 0.5 = sure
    student_uncert = 1 - 2 * np.abs(p_student - 0.5)  # 1 = uncertain, 0 = sure

    # Gate: 1 where teacher is confident, 0 where it is not
    gate  = (teacher_conf > gate_threshold).astype(float)

    # Alpha: how much teacher weight — capped at 0.35 even when gated
    alpha = gate * student_uncert * 0.35

    return (1 - alpha) * p_student + alpha * p_teacher


# ============================================================
# FIX 3 — OPTUNA STUDENT TUNING
# ============================================================
def tune_student(X_tr, y_tr, X_val, y_val, seed):
    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 100, 400),
            max_depth       = trial.suggest_int("max_depth", 3, 7),
            learning_rate   = trial.suggest_float("lr", 0.01, 0.15, log=True),
            subsample       = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree= trial.suggest_float("colsample", 0.6, 1.0),
            min_child_weight= trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha       = trial.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda      = trial.suggest_float("reg_lambda", 0.5, 3.0),
            verbosity=0, eval_metric="logloss", seed=seed,
        )
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_val)[:, 1]
        return log_loss(y_val, p)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params


# ============================================================
# PREPROCESSING HELPER
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
# MAIN
# ============================================================
if __name__ == "__main__":

    print("Loading datasets...")
    X_kaggle, y_kaggle = load_kaggle_data()
    X_uci,    y_uci    = load_uci_data()

    # ── Preprocess Kaggle (teacher) ───────────────────────
    imp_kag  = SimpleImputer(strategy="median")
    X_kaggle = imp_kag.fit_transform(X_kaggle)
    sm_kag             = SMOTE(random_state=42)
    X_kaggle, y_kaggle = sm_kag.fit_resample(X_kaggle, y_kaggle)
    sc_kag   = RobustScaler()
    X_kaggle = sc_kag.fit_transform(X_kaggle)

    # ── Train teacher ─────────────────────────────────────
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

        # ── FIX 4: Semantic bridge for teacher projection ──
        # Extract only the semantically matched UCI columns,
        # reorder them to match the Kaggle column order the
        # teacher was trained on, pad remaining Kaggle columns to 0.
        def project_uci_to_kaggle(X_uci_scaled):
            n_samples  = X_uci_scaled.shape[0]
            X_proj     = np.zeros((n_samples, X_kaggle.shape[1]))
            for uci_col, kag_col in zip(BRIDGE_UCI_COLS, BRIDGE_KAGGLE_COLS):
                X_proj[:, kag_col] = X_uci_scaled[:, uci_col]
            return X_proj

        X_train_proj = project_uci_to_kaggle(X_train)
        X_test_proj  = project_uci_to_kaggle(X_test)

        p_teach_train = teacher.predict_proba(X_train_proj)[:, 1]
        p_teach_test  = teacher.predict_proba(X_test_proj)[:,  1]

        # ── Feature selection + student ───────────────────
        for name, algo in algorithms.items():

            mask = algo(X_train, y_train, p_teach_train, run)
            idx  = np.where(mask == 1)[0]
            if len(idx) == 0:
                idx = np.arange(X_train.shape[1])

            X_tr = X_train[:, idx]
            X_te = X_test[:,  idx]
            X_v  = X_val[:,   idx]

            # Tune student with Optuna
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

            p_stud_test  = np.clip(student.predict_proba(X_te)[:, 1], 1e-6, 1-1e-6)
            p_teach_test_ = np.clip(p_teach_test, 1e-6, 1-1e-6)

            # Confidence-gated fusion
            fusion = dynamic_fusion(p_stud_test, p_teach_test_)

            acc = accuracy_score(y_test, (fusion > 0.5).astype(int))
            results[name].append(acc)

            if run == N_RUNS - 1:
                stud_acc  = accuracy_score(y_test, (p_stud_test   > 0.5))
                teach_acc = accuracy_score(y_test, (p_teach_test_  > 0.5))
                n_feats   = idx.shape[0]
                print(f"  {name:3s} | features={n_feats:2d} | "
                      f"Student={stud_acc:.4f}  "
                      f"Teacher(bridge)={teach_acc:.4f}  "
                      f"Fusion={acc:.4f}")

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
