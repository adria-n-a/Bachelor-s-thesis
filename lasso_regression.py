#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_lasso_regression.py

This script performs Lasso regression using leave-one-out cross-validation
to predict best ITR from a set of resting-state and behavioral predictors.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define paths
base_dir = r"D:\radboud\courses\Thesis"
derivatives_dir = os.path.join(base_dir, "sourcedata.tar", "derivatives")
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

peak_counts_csv = os.path.join(figures_dir, "peak_counts.csv")
mu_variance_csv = os.path.join(figures_dir, "mu_peak_variance.csv")
participants_tsv = os.path.join(base_dir, "scripts", "participants.tsv")
group_itr_csv = os.path.join(base_dir, "sourcedata.tar", "group_itr_per_subject.csv")
ppfactor_beta_csv = os.path.join(figures_dir, "ppfactor_relative_beta.csv")

# Get list of subject IDs
subjects = [
    name for name in os.listdir(derivatives_dir)
    if os.path.isdir(os.path.join(derivatives_dir, name))
]
if not subjects:
    raise RuntimeError(f"No subject folders found in {derivatives_dir}")

# Load model parameters and features from each subject's .npz file
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
    raise RuntimeError("No RS predictor records loaded.")

# Merge all additional predictors and outcome
df_peak = pd.read_csv(peak_counts_csv)
df = df_rs.merge(df_peak, on="subject", how="inner")

df_mu = pd.read_csv(mu_variance_csv)
df = df.merge(df_mu, on="subject", how="inner")

df_part = pd.read_csv(participants_tsv, sep="\t").rename(columns={"participant_id": "subject"})
df = df.merge(df_part[["subject", "sex", "age"]], on="subject", how="inner")

df_itr = pd.read_csv(group_itr_csv)
df_itr = df_itr.drop(columns=[col for col in ["best_time", "best_accuracy"] if col in df_itr.columns])
df = df.merge(df_itr, on="subject", how="inner")

df_pp = pd.read_csv(ppfactor_beta_csv)
df = df.merge(df_pp, on="subject", how="inner")

if df.shape[0] == 0:
    raise RuntimeError("No data available after merging all inputs.")

# One-hot encode 'sex' and prepare feature matrix
df_encoded = pd.get_dummies(df, columns=["sex"], drop_first=True)
y = df_encoded["best_itr"].values.astype(float)
subjects_list = df_encoded["subject"].tolist()
X = df_encoded.drop(columns=["subject", "best_itr"])
feature_names = X.columns.tolist()
X_mat = X.values.astype(float)

print("Predictors included in the analysis:")
for name in feature_names:
    print(f"  - {name}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_mat)

# Use LassoCV with Leave-One-Out to select regularization strength
loo = LeaveOneOut()
alphas = np.logspace(-4, 0, 50)
lasso_cv = LassoCV(alphas=alphas, cv=loo, max_iter=10000).fit(X_scaled, y)
best_alpha = lasso_cv.alpha_
print(f"\nChosen Lasso alpha (LOO-CV): {best_alpha:.5f}")

# Perform leave-one-out prediction using selected alpha
y_pred = np.empty_like(y)
for train_idx, test_idx in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train = y[train_idx]
    model = Lasso(alpha=best_alpha, max_iter=10000).fit(X_train, y_train)
    y_pred[test_idx] = model.predict(X_test)

# Compute performance metrics
cv_r2 = r2_score(y, y_pred)
r_val, p_val = pearsonr(y, y_pred)

# Save predictions to CSV
df_out = pd.DataFrame({
    "subject": subjects_list,
    "itr_true": y,
    "itr_pred": y_pred
})
out_csv = os.path.join(figures_dir, "loo_lasso_vs_itr.csv")
df_out.to_csv(out_csv, index=False)
print(f"\nSaved predictions CSV → {out_csv}")

# Generate scatter plot of true vs predicted ITR
plt.figure(figsize=(6, 4))
plt.scatter(y, y_pred, color="C0", label="Subjects")
lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
plt.plot(lims, lims, "k--", label="Ideal y = x")
plt.xlabel("True itr")
plt.ylabel("Predicted itr")
plt.text(
    0.05, 0.95,
    f"Pearson r = {r_val:.2f}\np = {p_val:.3f}\nCV R² = {cv_r2:.2f}",
    transform=plt.gca().transAxes,
    va="top"
)
plt.legend(loc="lower right")
plt.tight_layout()

out_png = os.path.join(figures_dir, "loo_lasso_vs_itr.png")
if os.path.exists(out_png):
    try:
        os.remove(out_png)
        print(f"Deleted old plot at: {out_png}")
    except PermissionError:
        print(f"Warning: Could not delete old plot: {out_png}")

plt.savefig(out_png)
plt.close()
print(f"Saved new plot to: {out_png}")

# Fit final model on full dataset to get coefficients
final_model = Lasso(alpha=best_alpha, max_iter=10000).fit(X_scaled, y)
coef = final_model.coef_

print("\nSelected features and coefficients:")
for name, c in zip(feature_names, coef):
    if abs(c) > 1e-6:
        print(f"  {name}: {c:.4f}")

# Print summary statistics
print("\n=== LASSO LOO Regression Results ===")
print(f"  Pearson r    = {r_val:.4f}")
print(f"  p-value      = {p_val:.4e}")
print(f"  CV R²        = {cv_r2:.4f}")
