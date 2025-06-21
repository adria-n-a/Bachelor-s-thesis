#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_age_itr_regression.py

Loads group-level ITR values and participant metadata.
Extracts "age" as a predictor and fits leave-one-out regression:
    ITR ~ Age

Saves:
- CSV of predictions
- Scatter plot with fit and confidence interval
"""

import os
import numpy as np
import pandas as pd

from linear_regression import run_regression, plot_regression_results


def main():
    # Input paths
    base_dir         = r"D:\radboud\courses\Thesis"
    group_itr_csv    = os.path.join(base_dir, "sourcedata.tar", "group_itr_per_subject.csv")
    participants_tsv = os.path.join(base_dir, "scripts", "participants.tsv")

    # Output paths
    out_dir          = os.path.join(base_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    output_csv       = os.path.join(out_dir, "age_vs_itr_predictions.csv")
    output_png       = os.path.join(out_dir, "age_vs_itr_plot.png")

    # Load group ITR values
    df_itr = pd.read_csv(group_itr_csv)
    if "subject" not in df_itr or "best_itr" not in df_itr:
        raise ValueError("group_itr_per_subject.csv must contain 'subject' and 'best_itr' columns")

    subjects   = df_itr["subject"].tolist()
    itr_values = df_itr["best_itr"].values

    # Load participant metadata
    df_part = pd.read_csv(participants_tsv, sep="\t")
    if "participant_id" not in df_part or "age" not in df_part:
        raise ValueError("participants.tsv must contain 'participant_id' and 'age' columns")

    # Match and filter valid entries
    ages, valid_subjs, valid_itrs = [], [], []
    for subj, itr_val in zip(subjects, itr_values):
        if np.isnan(itr_val):
            continue

        match = df_part[df_part["participant_id"] == subj]
        if match.empty or pd.isna(match["age"].values[0]):
            continue

        age = match["age"].values[0]
        ages.append(age)
        valid_itrs.append(itr_val)
        valid_subjs.append(subj)

    if not valid_subjs:
        raise RuntimeError("No valid subject entries with age and ITR data.")

    feature_values = np.array(ages)
    itr_values     = np.array(valid_itrs)

    # Run leave-one-out regression
    df_out, r_val, p_val, cv_r2 = run_regression(
        feature_values=feature_values,
        itr_values=itr_values,
        subjects=valid_subjs
    )

    # Save predictions
    df_out.to_csv(output_csv, index=False)
    print(f"Saved predictions CSV → {output_csv}")

    # Save regression plot
    plot_regression_results(
        df_out=df_out,
        feature_name="Age (years)",
        itr_name="itr",
        r_val=r_val,
        p_val=p_val,
        cv_r2=cv_r2,
        output_png=output_png
    )
    print(f"Saved regression plot → {output_png}")


if __name__ == "__main__":
    main()
