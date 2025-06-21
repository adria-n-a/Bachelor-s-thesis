#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_mu_variance.py

For each subject:
1. Load the 120 s “eyes-open” EEG recording at channel Oz.
2. Extract the continuous Oz signal.
3. Split into overlapping 10 s windows (3 s overlap).
4. For each window, compute the PSD, fit a parametric model, and extract μ₁.
5. Compute the variance of all valid μ₁ estimates.
6. Save all results as a CSV file.
"""

import os
import numpy as np
import pandas as pd
import mne
import pyxdf
from scipy.optimize import curve_fit
from mnelab_read_raw import read_raw_xdf  # Custom loader used previously

# Parametric spectral model components
def g1(f, lamb, k1, k2):
    return k1 + k2 / (f ** lamb)

def g2(f, mu, sigma, k):
    return k * (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * (f - mu)**2 / (sigma**2))

def g(f, lamb, mu1, mu2, sigma1, sigma2, k1, k2, k3, k4):
    return g1(f, lamb, k1, k2) + g2(f, mu1, sigma1, k3) + g2(f, mu2, sigma2, k4)

# Fit g(f) to the PSD from a 10 s window of Oz data, return estimated μ₁
def extract_mu1_from_epoch(epoch_data, sfreq):
    psd, freqs = mne.time_frequency.psd_array_welch(
        epoch_data,
        sfreq=sfreq,
        fmin=0.5,
        fmax=30.0,
        n_fft=2048,
        verbose=False
    )
    psd_uv2 = psd * 1e12
    psd_db = 10.0 * np.log10(psd_uv2)

    y = psd_db
    f = freqs
    p0 = [0.2, 10, 20, 5, 5, 1e-6, 25, 10, 5]
    bounds_lower = [0, 6, 16, 0, 0, 0, 0, 0, 0]
    bounds_upper = [np.inf, 14, 24, 20, 20, np.inf, np.inf, np.inf, np.inf]

    try:
        popt, _ = curve_fit(g, f, y, p0=p0, bounds=(bounds_lower, bounds_upper))
    except RuntimeError:
        return np.nan

    mu1 = popt[1]
    return mu1

# Load one subject's data, extract Oz signal, compute μ₁ for each window, return variance
def compute_mu_peak_variance_for_Oz(subject, data_dir):
    raw_fname = os.path.join(
        data_dir, "sourcedata", f"sub-{subject}", "ses-S001", "eeg",
        f"sub-{subject}_ses-S001_task-rstate_open_run-001_eeg.xdf"
    )
    if not os.path.exists(raw_fname):
        raise FileNotFoundError(f"XDF not found: {raw_fname}")

    streams = pyxdf.resolve_streams(raw_fname)
    names = [s["name"] for s in streams]
    if "BioSemi" not in names:
        raise RuntimeError(f"No BioSemi stream in {raw_fname}")
    bio_idx = names.index("BioSemi")
    bio_id = streams[bio_idx]["stream_id"]

    raw = read_raw_xdf(raw_fname, stream_ids=[bio_id])

    raw = raw.drop_channels([f"EX{i}" for i in range(1, 9)] + [f"AIB{i}" for i in range(1, 33)])
    montage = mne.channels.make_standard_montage("biosemi64")
    rename_map = {
        old: new
        for old, new in zip(
            [f"A{i}" for i in range(1, 33)] + [f"B{i}" for i in range(1, 33)],
            montage.ch_names
        )
    }
    raw = raw.rename_channels(rename_map)

    raw._data[0] -= raw._data[0].min()
    raw._data[0, raw._data[0] > 0] = 1
    events = mne.find_events(raw, stim_channel="Trig1", verbose=False)

    epo = mne.Epochs(
        raw, events, tmin=0.0, tmax=120.0,
        baseline=None, picks="eeg",
        preload=True, verbose=False
    )
    epo.set_montage(montage, on_missing="ignore")
    epo = mne.preprocessing.compute_current_source_density(epo, verbose=False)

    if "Oz" not in epo.ch_names:
        raise RuntimeError(f"'Oz' not found in channels for subject {subject}")

    data = epo.copy().pick("Oz").get_data()
    oz_arr = data[0, 0, :]
    sfreq = epo.info["sfreq"]

    win_len = int(10.0 * sfreq)
    step = int((10.0 - 3.0) * sfreq)

    n_samples = oz_arr.shape[0]
    mu1_list = []

    start = 0
    while start + win_len <= n_samples:
        epoch_data = oz_arr[start:start + win_len]
        mu1_val = extract_mu1_from_epoch(epoch_data, sfreq)
        mu1_list.append(mu1_val)
        start += step

    mu1_arr = np.array(mu1_list)
    mu1_arr = mu1_arr[~np.isnan(mu1_arr)]

    if mu1_arr.size < 2:
        raise RuntimeError(f"Not enough valid μ₁ windows for {subject} (got {mu1_arr.size}).")

    mu_variance = float(np.var(mu1_arr, ddof=0))
    return mu_variance

# Loop through all subjects, compute μ₁ variance for Oz, and save results to CSV
if __name__ == "__main__":
    base_dir = r"D:\radboud\courses\Thesis\sourcedata.tar"
    subjects = [
        "VPpdca", "VPpdcb", "VPpdcc", "VPpdcd", "VPpdce", "VPpdcf", "VPpdcg",
        "VPpdch", "VPpdci", "VPpdcj", "VPpdck", "VPpdcl", "VPpdcm", "VPpdcn",
        "VPpdco", "VPpdcp", "VPpdcq", "VPpdcr", "VPpdcs", "VPpdct", "VPpdcu",
        "VPpdcv", "VPpdcw", "VPpdcx", "VPpdcy", "VPpdcz"
    ]

    out_records = []
    for subj in subjects:
        try:
            mv = compute_mu_peak_variance_for_Oz(subj, base_dir)
        except Exception as e:
            print(f"Skipping {subj}: {e}")
            continue

        out_records.append({"subject": subj, "mu_variance": mv})
        print(f"{subj}: mu_variance = {mv:.4f}")

    df_mu = pd.DataFrame(out_records)
    if df_mu.empty:
        raise RuntimeError("No mu_variance values computed for any subject.")

    df_mu = df_mu.sort_values("subject").reset_index(drop=True)

    figures_dir = r"D:\radboud\courses\Thesis\figures"
    os.makedirs(figures_dir, exist_ok=True)
    csv_path = os.path.join(figures_dir, "mu_peak_variance.csv")
    df_mu.to_csv(csv_path, index=False)
    print(f"\nSaved mu_peak_variance CSV → {csv_path}")
