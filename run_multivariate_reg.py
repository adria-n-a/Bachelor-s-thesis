#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_multivariate_regression.py

Performs multivariate leave-one-out regression to predict ITR using multiple predictors:
- Model parameters from resting-state NPZ files (lamb, mu1, mu2, sigma1, sigma2, feat1)
- Additional CSV-derived features (e.g., peak counts, mu variance)
- Demographic metadata (age, sex)

Outputs:
- Predicted ITRs (CSV)
- Scatter plot of predicted vs. true ITR
- Final OLS model coefficients
"""

import os
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Step 1: Define file paths
base_dir = r"D:\radboud\courses\Thesis"
derivatives_dir = os.path.join(base_dir, "sourcedata.tar", "derivatives")
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

peak_counts_csv = os.path.join(figures_dir, "peak_counts.csv")
mu_variance_csv = os.path.join(figures_dir, "mu_peak_variance.csv")
participants_tsv = os.path.join(base_dir, "scripts", "participants.tsv")
group_itr_csv = os.path.join(base_dir, "sourcedata.tar", "group_itr_per_subject.csv")

# Step 2: Get subject IDs
subjects = [
    name for name in os.listdir(derivatives_dir)
    if os.path.isdir(os.path.join(derivatives_dir, name))
]
if not subjects:
    raise RuntimeError(f"No subject folders found in {derivatives_dir!r}")

# Step 3: Load NPZ data for each subject
records = []
for subj in subjects:
    npz_path = os.path.join(
        derivatives_dir,
        subj,
        f"{subj}_rs_task-rstate_open_run-001.npz"
    )
    if not os.path.exists(npz_path):
        print(f"Skipping {subj}: NPZ not found.")
        continue
    data = np.load(npz_path)
    records.append({
        "subject": subj,
        "lamb":    float(data["lamb"]),
        "mu1":     float(data["mu1"]),
        "mu2":     float(data["mu2"]),
        "sigma1":  float(data["sigma1"]),
        "sigma2":  float(data["sigma2"]),
        "feat":    float(data["feat1"])
    })

df_rs = pd.DataFrame(records)
if df_rs.empty:
    raise RuntimeError("No RS predictor records loaded from NPZ files.")

# Step 4–7: Merge additional predictors
df_peak = pd.read_csv(peak_counts_csv)
df = df_rs.merge(df_peak, on="subject", how="inner")

df_mu = pd.read_csv(mu_variance_csv)
df = df.merge(df_mu, on="subject", how="inner")

df_part = pd.read_csv(participants_tsv, sep="\t")
df_part = df_part.rename(columns={"participant_id": "subject"})
df = df.merge(df_part[["subject", "sex", "age"]], on="subject", how="inner")

df_itr = pd.read_csv(group_itr_csv)
df_itr = df_itr.drop(columns=[col for col in ["best_time", "best_accuracy"] if col in df_itr.columns])
df = df.merge(df_itr, on="subject", how="inner")

if df.shape[0] == 0:
    raise RuntimeError("No subjects remained after merging all predictors + ITR")

# Step 8: One-hot encode 'sex'
df_encoded = pd.get_dummies(df, columns=["sex"], drop_first=True)

# Step 9: Extract predictors and targets
y = df_encoded["best_itr"].values.astype(float)
subjects_list = df_encoded["subject"].tolist()

X = df_encoded.drop(columns=["subject", "best_itr"])
feature_names = X.columns.tolist()
X_mat = X.values.astype(float)

print("Predictors included in the analysis:")
for name in feature_names:
    print(f"  - {name}")

# Step 10: Normalize X to [0, 1]
scaler = MinMaxScaler()
X_mat = scaler.fit_transform(X_mat)

# Step 11: Leave-One-Out Cross-Validation
loo = LeaveOneOut()
n = X_mat.shape[0]
y_pred = np.empty(n, dtype=float)

for train_idx, test_idx in loo.split(X_mat):
    X_train, X_test = X_mat[train_idx], X_mat[test_idx]
    y_train = y[train_idx]
    model = LinearRegression().fit(X_train, y_train)
    y_pred[test_idx] = model.predict(X_test)

# Step 12: Compute and print CV results
cv_r2 = r2_score(y, y_pred)
r_val, p_val = pearsonr(y, y_pred)

# Step 13: Save predictions to CSV
df_out = pd.DataFrame({
    "subject":  subjects_list,
    "itr_true": y,
    "itr_pred": y_pred
})
out_csv = os.path.join(figures_dir, "loo_multivariate_vs_itr.csv")
df_out.to_csv(out_csv, index=False)
print(f"\nSaved predictions CSV → {out_csv}")

# Step 14: Plot predicted vs. true ITR
plt.figure(figsize=(6, 4))
plt.scatter(y, y_pred, color="C0", label="Subjects")
lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
plt.plot(lims, lims, "k--", label="Ideal y = x")
plt.xlabel("True itr")
plt.ylabel("Predicted itr")
plt.title("Multivariate LOO-CV: True vs. Predicted ITR")
plt.text(
    0.05, 0.95,
    f"Pearson r = {r_val:.2f}\np = {p_val:.3f}\nCV R² = {cv_r2:.2f}",
    transform=plt.gca().transAxes,
    va="top"
)
plt.legend(loc="lower right")
plt.tight_layout()

out_png = os.path.join(figures_dir, "loo_multivariate_vs_itr.png")
try:
    if os.path.exists(out_png):
        os.remove(out_png)
except PermissionError:
    print(f"Could not delete existing plot file: {out_png}")
print(f"Saving plot to: {out_png}")
plt.savefig(out_png)
plt.close()

# Step 15: Final OLS coefficients
final_model = LinearRegression().fit(X_mat, y)
coefs = final_model.coef_

print("\n=== MULTIVARIATE LOO Regression Results ===")
print(f"  Pearson r    = {r_val:.4f}")
print(f"  p-value      = {p_val:.4e}")
print(f"  CV R²        = {cv_r2:.4f}")

print("\nFinal OLS coefficients (feature → coefficient):")
for name, c in zip(feature_names, coefs):
    print(f"  {name}: {c:.4f}")
