#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_best3_search.py

This script:
- Loads multiple subject-level predictors and outcomes (ITR, peak count, mu variance, etc.).
- Merges them into a single dataset.
- Selects top 5 predictors by univariate correlation with ITR.
- Tests all 3-predictor combinations from that top 5 using leave-one-out regression.
- Outputs the best-performing combination (by CV R²), predictions, and a result plot.
"""

import os
import itertools
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define input and output paths
base_dir = r"D:\radboud\courses\Thesis"
derivatives_dir = os.path.join(base_dir, "sourcedata.tar", "derivatives")
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

peak_counts_csv = os.path.join(figures_dir, "peak_counts.csv")
mu_variance_csv = os.path.join(figures_dir, "mu_peak_variance.csv")
participants_tsv = os.path.join(base_dir, "scripts", "participants.tsv")
group_itr_csv = os.path.join(base_dir, "sourcedata.tar", "group_itr_per_subject.csv")
ppfactor_beta_csv = os.path.join(figures_dir, "ppfactor_relative_beta.csv")

# Load all subject folders
subjects = [
    name for name in os.listdir(derivatives_dir)
    if os.path.isdir(os.path.join(derivatives_dir, name))
]
if not subjects:
    raise RuntimeError(f"No subject folders found in {derivatives_dir!r}")

# Extract selected model parameters and features from each subject's .npz file
records = []
for subj in subjects:
    npz_path = os.path.join(derivatives_dir, subj, f"{subj}_rs_task-rstate_open_run-001.npz")
    if not os.path.exists(npz_path):
        print(f"Skipping {subj}: NPZ not found.")
        continue
    data = np.load(npz_path)
    records.append({
        "subject": subj,
        "lamb": float(data["lamb"]),
        "mu1": float(data["mu1"]),
        "mu2": float(data["mu2"]),
        "sigma1": float(data["sigma1"]),
        "sigma2": float(data["sigma2"]),
        "k1": float(data["k1"]),
        "k2": float(data["k2"]),
        "feat": float(data["feat1"])
    })

df_rs = pd.DataFrame(records)
if df_rs.empty:
    raise RuntimeError("No RS predictor records loaded from NPZ files.")

# Load and merge all other relevant data
df_peak = pd.read_csv(peak_counts_csv)
df = df_rs.merge(df_peak, on="subject", how="inner")

df_mu = pd.read_csv(mu_variance_csv)
df = df.merge(df_mu, on="subject", how="inner")

df_part = pd.read_csv(participants_tsv, sep="\t").rename(columns={"participant_id": "subject"})
df = df.merge(df_part[["subject", "sex", "age"]], on="subject", how="inner")

df_itr = pd.read_csv(group_itr_csv).drop(columns=[col for col in ["best_time", "best_accuracy"] if col in df_itr.columns])
df = df.merge(df_itr, on="subject", how="inner")

if not os.path.exists(ppfactor_beta_csv):
    raise FileNotFoundError(f"{ppfactor_beta_csv!r} not found")
df_pp = pd.read_csv(ppfactor_beta_csv)
df = df.merge(df_pp, on="subject", how="inner")

if df.shape[0] == 0:
    raise RuntimeError("No subjects remained after merging all predictors + ITR + new metrics")

# One-hot encode sex and prepare features
df_encoded = pd.get_dummies(df, columns=["sex"], drop_first=True)
y = df_encoded["best_itr"].values.astype(float)
subjects_list = df_encoded["subject"].tolist()
X = df_encoded.drop(columns=["subject", "best_itr"])
feature_names = X.columns.tolist()
X_mat = X.values.astype(float)

# Normalize features to [0, 1] range
scaler = MinMaxScaler()
X_mat = scaler.fit_transform(X_mat)

print("Full set of predictors available:")
for name in feature_names:
    print(f"  - {name}")
print()

# Compute univariate Pearson correlations
univ_stats = []
for idx, fname in enumerate(feature_names):
    xi = X_mat[:, idx]
    r, p = pearsonr(xi, y)
    univ_stats.append((fname, abs(r), r, p))

univ_stats.sort(key=lambda tup: tup[1], reverse=True)
print("Univariate correlations (sorted by |r|):")
for fname, abs_r, r, p in univ_stats:
    print(f"  {fname}:  r = {r:.3f},  p = {p:.4f}")
print()

# Select top 5 predictors by absolute correlation
top5 = [fname for fname, abs_r, r, p in univ_stats[:5]]
print("Top 5 predictors by |r|:")
for t in top5:
    print(f"  - {t}")
print()

# Evaluate all 3-feature combinations using leave-one-out regression
loo = LeaveOneOut()
best_result = {"combo": None, "cv_r2": -np.inf, "pearson_r": None, "pearson_p": None}

print("Testing all combinations of 3 out of the top 5:\n")
for combo in itertools.combinations(top5, 3):
    cols_idx = [feature_names.index(c) for c in combo]
    X_sub = X_mat[:, cols_idx]
    y_pred = np.empty_like(y)
    for train_idx, test_idx in loo.split(X_sub):
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train = y[train_idx]
        model = LinearRegression().fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)
    cv_r2 = r2_score(y, y_pred)
    r_val, p_val = pearsonr(y, y_pred)
    print(f"Combo {combo}:  CV R² = {cv_r2:.4f},  Pearson r(true vs pred) = {r_val:.4f}, p = {p_val:.4f}")
    if cv_r2 > best_result["cv_r2"]:
        best_result = {
            "combo": combo,
            "cv_r2": cv_r2,
            "pearson_r": r_val,
            "pearson_p": p_val
        }

# Output results from best combo
print("\n=== Best 3‐Predictor Combination ===")
print(f"Predictors: {best_result['combo']}")
print(f"CV R²     : {best_result['cv_r2']:.4f}")
print(f"Pearson r : {best_result['pearson_r']:.4f}")
print(f"Pearson p : {best_result['pearson_p']:.4e}\n")

# Run LOO again for best combination to get predictions
best_combo = best_result["combo"]
cols_idx = [feature_names.index(c) for c in best_combo]
X_best = X_mat[:, cols_idx]
y_pred_best = np.empty_like(y)
for train_idx, test_idx in loo.split(X_best):
    X_train, X_test = X_best[train_idx], X_best[test_idx]
    y_train = y[train_idx]
    model = LinearRegression().fit(X_train, y_train)
    y_pred_best[test_idx] = model.predict(X_test)

# Plot true vs predicted ITR for best 3-predictor model
plt.figure(figsize=(6, 4))
plt.scatter(y, y_pred_best, color="C0", label="Subjects")
lims = [min(y.min(), y_pred_best.min()), max(y.max(), y_pred_best.max())]
plt.plot(lims, lims, "k--", label="Ideal y = x")
plt.xlabel("True itr")
plt.ylabel("Predicted itr")
plt.title(f"Best 3‐Predictor LOO‐CV: {best_combo}")
rval, pval = pearsonr(y, y_pred_best)
cv_r2_best = r2_score(y, y_pred_best)
plt.text(0.05, 0.95, f"Pearson r = {rval:.2f}\np = {pval:.3f}\nCV R² = {cv_r2_best:.2f}", transform=plt.gca().transAxes, va="top")
plt.legend(loc="lower right")
plt.tight_layout()

out_png = os.path.join(figures_dir, "best3_vs_itr.png")
if os.path.exists(out_png):
    try:
        os.remove(out_png)
    except PermissionError:
        pass
plt.savefig(out_png)
plt.close()
print(f"Saved best‐3‐predictor plot → {out_png}\n")

# Save prediction results to CSV
df_out_best = pd.DataFrame({
    "subject": subjects_list,
    "itr_true": y,
    "itr_pred": y_pred_best
})
out_csv_best = os.path.join(figures_dir, "loo_best3_vs_itr.csv")
df_out_best.to_csv(out_csv_best, index=False)
print(f"Saved best‐3‐predictor predictions CSV → {out_csv_best}")
