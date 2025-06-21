#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_ppfactor_relative_beta.py

1) Loads each subject's eyes-open PSD from .npz files.
2) Computes:
   - PPfactor = (alpha + beta) / (theta + gamma from 30–40 Hz)
   - RelativeBeta = 10 * log10(beta / total power from 4–40 Hz)
3) Saves a CSV with subject, PPfactor, and RelativeBeta (in dB).
4) Merges with group-level ITR data.
5) Runs leave-one-out regression of best ITR on each metric separately.
6) Saves two scatter plots showing fit and metrics.
"""

import os
import glob
import numpy as np
import pandas as pd

# Frequency band definitions
bands = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 40),  # gamma band limited by PSD range
    "total": (4, 40)
}

def band_power(freq, psd_db, band_name):
    fmin, fmax = bands[band_name]
    mask = (freq >= fmin) & (freq < fmax)
    if not np.any(mask):
        return 0.0

    psd_lin = 10 ** (psd_db[mask] / 10.0)
    df = np.mean(np.diff(freq[mask])) if len(freq[mask]) > 1 else 1.0
    return np.sum(psd_lin) * df

def compute_metrics_from_npz(npz_path):
    data = np.load(npz_path)
    freq = data["freq"]
    psd_db = data["psd_db"]

    theta = band_power(freq, psd_db, "theta")
    alpha = band_power(freq, psd_db, "alpha")
    beta = band_power(freq, psd_db, "beta")
    gamma = band_power(freq, psd_db, "gamma")
    total = band_power(freq, psd_db, "total")

    denom = theta + gamma
    ppfactor = (alpha + beta) / denom if denom > 0 else np.nan
    rel_beta = 10.0 * np.log10(beta / total) if total > 0 and beta > 0 else np.nan

    return float(ppfactor), float(rel_beta)

if __name__ == "__main__":
    # Location of derivatives with PSD .npz files
    base_dir = r"D:\radboud\courses\Thesis\sourcedata.tar\derivatives"
    pattern = os.path.join(base_dir, "*", "*rstate_open_run-001*.npz")

    records = []
    for npz_path in glob.glob(pattern):
        fname = os.path.basename(npz_path)
        subject = fname.split("_")[0]
        try:
            ppf, rb = compute_metrics_from_npz(npz_path)
        except Exception as e:
            print(f"Skipping {subject} due to error: {e}")
            continue

        records.append({
            "subject": subject,
            "ppfactor": ppf,
            "relative_beta_dB": rb
        })
        print(f"{subject}: PPfactor={ppf:.4f}, RelativeBeta={rb:.4f} dB")

    if not records:
        raise RuntimeError("No metrics were computed. No valid data files found.")

    metrics_df = pd.DataFrame(records).sort_values("subject").reset_index(drop=True)

    out_dir = r"D:\radboud\courses\Thesis\figures"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ppfactor_relative_beta.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV with computed metrics to: {csv_path}")

    # Load group ITRs and merge with metrics
    itr_csv = r"D:\radboud\courses\Thesis\sourcedata.tar\group_itr_per_subject.csv"
    itr_df = pd.read_csv(itr_csv)
    for col in ["subject", "best_itr"]:
        if col not in itr_df.columns:
            raise ValueError(f"Missing column '{col}' in: {itr_csv}")
    itr_df = itr_df[["subject", "best_itr"]]
    merged = pd.merge(metrics_df, itr_df, on="subject", how="inner")

    if merged.shape[0] != metrics_df.shape[0]:
        missing = set(metrics_df["subject"]) - set(itr_df["subject"])
        print(f"Warning: {len(missing)} subjects had no ITR entry and were skipped:\n  {missing}")

    # Run regression for each metric
    try:
        from linear_regression import run_regression, plot_regression_results
    except ImportError:
        raise ImportError("Could not import run_regression or plot_regression_results. Ensure 'leave_one_out_regression.py' is accessible.")

    predictors = ["ppfactor", "relative_beta_dB"]
    for pname in predictors:
        X = merged[pname].values.astype(float)
        y = merged["best_itr"].values.astype(float)
        subjects = merged["subject"].tolist()

        if np.all(np.isnan(X)) or np.nanstd(X) == 0:
            print(f"Skipping regression on '{pname}': all values are NaN or constant.")
            continue

        df_out, r_val, p_val, cv_r2 = run_regression(
            feature_values=X,
            itr_values=y,
            subjects=subjects
        )

        print(f"\nResults for regression on {pname}:")
        print(f"  Pearson r        = {r_val:.4f}")
        print(f"  p-value          = {p_val:.4e}")
        print(f"  CV R² (LOO)      = {cv_r2:.4f}")

        output_png = os.path.join(out_dir, f"{pname}_vs_best_itr.png")
        plot_regression_results(
            df_out=df_out,
            feature_name=pname,
            itr_name="itr",
            r_val=r_val,
            p_val=p_val,
            cv_r2=cv_r2,
            output_png=output_png
        )
        print(f"Figure saved: {output_png}")

    print("\nFinished. All regression results and figures are in:")
    print(f"  {out_dir}")
