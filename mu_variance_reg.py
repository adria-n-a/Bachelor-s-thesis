#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_mu_variance_regression.py

Performs leave-one-subject-out linear regression to predict ITR from μ₁ variance.
"""

import os
import pandas as pd
from linear_regression import run_regression, plot_regression_results

if __name__ == "__main__":
    # Define file paths
    base_dir = r"D:\radboud\courses\Thesis"
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    mu_csv = os.path.join(figures_dir, "mu_peak_variance.csv")
    itr_csv = os.path.join(base_dir, "sourcedata.tar", "group_itr_per_subject.csv")

    # Load variance and ITR data
    df_mu = pd.read_csv(mu_csv)
    df_itr = pd.read_csv(itr_csv)

    # Check required columns
    for col in ["subject", "mu_variance"]:
        if col not in df_mu.columns:
            raise ValueError(f"Column '{col}' missing from {mu_csv}")
    for col in ["subject", "best_itr"]:
        if col not in df_itr.columns:
            raise ValueError(f"Column '{col}' missing from {itr_csv}")

    # Merge data on subject ID
    merged = pd.merge(df_mu, df_itr, on="subject", how="inner")
    if merged.shape[0] < df_mu.shape[0]:
        missing = set(df_mu["subject"]) - set(df_itr["subject"])
        print(f"Warning: {len(missing)} subject(s) had no ITR entry and were dropped:\n  {missing}")

    # Extract inputs for regression
    subjects = merged["subject"].tolist()
    feature_values = merged["mu_variance"].values.astype(float)
    itr_values = merged["best_itr"].values.astype(float)

    # Run leave-one-out regression
    df_out, r_val, p_val, cv_r2 = run_regression(
        feature_values=feature_values,
        itr_values=itr_values,
        subjects=subjects
    )

    # Save prediction results
    out_csv = os.path.join(figures_dir, "loo_mu_variance_vs_itr.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved predictions CSV → {out_csv}")

    # Create and save regression plot
    out_png = os.path.join(figures_dir, "loo_mu_variance_vs_itr.png")
    plot_regression_results(
        df_out,
        feature_name=r"$\mu_1$ variability",
        itr_name="best_itr",
        r_val=r_val,
        p_val=p_val,
        cv_r2=cv_r2,
        output_png=out_png
    )
    print(f"Saved regression plot → {out_png}\n")

    # Print summary statistics
    print("=== LOO Regression Results: ITR ~ mu_variance ===")
    print(f"  Pearson r    = {r_val:.4f}")
    print(f"  p-value      = {p_val:.4e}")
    print(f"  CV R²        = {cv_r2:.4f}")
