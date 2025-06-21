#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_param_regressions_per_subject_ITR.py

1) Loads each subject's .npz file for task="rstate_open", run=1.
2) Extracts parametric model features (10 total: 9 parameters + feat).
3) Loads group-level ITR CSV (columns: subject, best_itr).
4) Merges both datasets on 'subject'.
5) Runs leave-one-out regression: ITR ~ each parameter.
6) Saves one scatter plot PNG per parameter to the figures folder.
"""

import os
import glob
import numpy as np
import pandas as pd

# Input paths
derivatives_dir = r"D:\radboud\courses\Thesis\sourcedata.tar\derivatives"
itr_csv         = r"D:\radboud\courses\Thesis\sourcedata.tar\group_itr_per_subject.csv"
figures_dir     = r"D:\radboud\courses\Thesis\figures"
os.makedirs(figures_dir, exist_ok=True)

# Step 1: Extract parametric model features for each subject
records = []
for subj in os.listdir(derivatives_dir):
    subj_path = os.path.join(derivatives_dir, subj)
    if not os.path.isdir(subj_path):
        continue

    npz_path = os.path.join(subj_path, f"{subj}_rs_task-rstate_open_run-001.npz")
    if not os.path.isfile(npz_path):
        continue

    try:
        data = np.load(npz_path)
        records.append({
            "subject": subj,
            "lamb":    float(data["lamb"]),
            "mu1":     float(data["mu1"]),
            "mu2":     float(data["mu2"]),
            "sigma1":  float(data["sigma1"]),
            "sigma2":  float(data["sigma2"]),
            "k1":      float(data["k1"]),
            "k2":      float(data["k2"]),
            "k3":      float(data["k3"]),
            "k4":      float(data["k4"]),
            "feat":    float(data["feat1"])
        })
    except KeyError as e:
        raise KeyError(f"Missing key {e} in file: {npz_path}")

params_df = pd.DataFrame(records)
if params_df.empty:
    raise RuntimeError(f"No valid .npz files found under: {derivatives_dir}")

# Step 2: Load ITR data and merge
itr_df = pd.read_csv(itr_csv)
if not {"subject", "best_itr"}.issubset(itr_df.columns):
    raise ValueError("ITR CSV must contain 'subject' and 'best_itr' columns")

merged = pd.merge(params_df, itr_df[["subject", "best_itr"]], on="subject", how="inner")
if merged.shape[0] != params_df.shape[0]:
    missing = set(params_df["subject"]) - set(itr_df["subject"])
    print(f"Warning: {len(missing)} subject(s) had no ITR entry:\n  {missing}")

# Step 3: Regression
try:
    from linear_regression import run_regression, plot_regression_results
except ImportError:
    raise ImportError("Could not import required regression utilities.")

param_names = ["lamb", "mu1", "mu2", "sigma1", "sigma2", "k1", "k2", "k3", "k4", "feat"]
pretty_labels = {
    "mu1": r"$\mu_1$",
    "mu2": r"$\mu_2$",
    "sigma1": r"$\sigma_1$",
    "sigma2": r"$\sigma_2$",
    "lamb": r"$\lambda$",
    "k1": r"$k_1$",
    "k2": r"$k_2$",
    "k3": r"$k_3$",
    "k4": r"$k_4$",
    "feat": r"PBC"
}

for pname in param_names:
    X = merged[pname].values.astype(float)
    y = merged["best_itr"].values.astype(float)
    subjects = merged["subject"].tolist()

    df_out, r_val, p_val, cv_r2 = run_regression(X, y, subjects)

    print(f"\nParameter: {pname}")
    print(f"  Pearson r          = {r_val:.4f}")
    print(f"  p-value            = {p_val:.4e}")
    print(f"  CV R² (LOO)        = {cv_r2:.4f}")

    output_png = os.path.join(figures_dir, f"{pname}_vs_best_itr.png")
    plot_regression_results(
        df_out=df_out,
        feature_name=pretty_labels.get(pname, pname),
        itr_name="itr",
        r_val=r_val,
        p_val=p_val,
        cv_r2=cv_r2,
        output_png=output_png
    )
    print(f"  → Saved plot to: {output_png}")

print("\nFinished. All parameter regression plots are in:")
print(f"  {figures_dir}")
