#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_peakcount_regression.py

This script:
1. Computes 25th percentile thresholds for k3 and k4 features across subjects.
2. Uses these thresholds to count the number of significant peaks (0–2) in each subject's PSD.
3. Runs leave-one-subject-out regression: ITR as a function of peak count.
4. Saves results to CSV and plots the regression output.
"""

import os
import numpy as np
import pandas as pd

from linear_regression import run_regression, plot_regression_results

def main():
    # Define input/output paths
    base_dir  = r"D:\radboud\courses\Thesis"
    group_csv = os.path.join(base_dir, "sourcedata.tar", "group_itr_per_subject.csv")
    deriv_dir = os.path.join(base_dir, "sourcedata.tar", "derivatives")
    out_dir   = os.path.join(base_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    itr_col = "best_itr"

    # Load group-level ITR data
    df_itr = pd.read_csv(group_csv)
    subjects = df_itr["subject"].tolist()
    itr_vals = df_itr[itr_col].values

    # Extract k3 and k4 features across all subjects
    all_k3 = []
    all_k4 = []
    for subj in subjects:
        npz_path = os.path.join(deriv_dir, subj, f"{subj}_rs_task-rstate_open_run-001.npz")
        data = np.load(npz_path)
        all_k3.append(float(data["feat1"]))
        all_k4.append(float(data["feat2"]))

    all_k3 = np.array(all_k3)
    all_k4 = np.array(all_k4)

    # Compute 25th percentile thresholds for both features
    thresh_k3 = np.percentile(all_k3, 50)
    thresh_k4 = np.percentile(all_k4, 50)
    print(f"Thresholds: k3 = {thresh_k3:.2f}, k4 = {thresh_k4:.2f}")

    # Count how many k-values exceed the thresholds for each subject
    peak_counts = []
    for subj in subjects:
        data = np.load(os.path.join(deriv_dir, subj, f"{subj}_rs_task-rstate_open_run-001.npz"))
        k3_val = float(data["feat1"])
        k4_val = float(data["feat2"])
        count = int(k3_val > thresh_k3) + int(k4_val > thresh_k4)
        peak_counts.append(count)

    peak_counts = np.array(peak_counts)

    # Save peak count values to CSV
    df_peaks = pd.DataFrame({
        "subject": subjects,
        "peak_count": peak_counts
    })
    peaks_csv = os.path.join(out_dir, "peak_counts.csv")
    df_peaks.to_csv(peaks_csv, index=False)
    print(f"Saved peak counts CSV → {peaks_csv}")

    # Run leave-one-subject-out regression of ITR on peak count
    df_out, r_val, p_val, cv_r2 = run_regression(
        feature_values=peak_counts,
        itr_values=itr_vals,
        subjects=subjects
    )

    # Save regression output (true vs predicted ITR)
    csv_path = os.path.join(out_dir, "loo_peakcount_vs_itr.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Saved predictions CSV → {csv_path}")

    # Generate and save the regression plot
    png_path = os.path.join(out_dir, "loo_peakcount_vs_itr.png")
    plot_regression_results(
        df_out,
        feature_name="peak_count",
        itr_name="itr",
        r_val=r_val,
        p_val=p_val,
        cv_r2=cv_r2,
        output_png=png_path
    )
    print(f"Saved regression plot → {png_path}")

if __name__ == "__main__":
    main()
