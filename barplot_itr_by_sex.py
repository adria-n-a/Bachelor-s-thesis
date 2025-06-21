#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leave_one_out_regression_by_sex.py

Loads per‐subject ITRs and participant sex, then plots group means ± SD with individual
dots jittered, counts included in the legend, and p-value annotated above the bars.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Define input file paths
data_dir = r"D:\radboud\courses\Thesis\sourcedata.tar"
participants_file = r"D:\radboud\courses\Thesis\scripts\participants.tsv"

# Read ITR values per subject from CSV and store in a dictionary
itr_df = pd.read_csv(os.path.join(data_dir, "group_itr_per_subject.csv"))
itr_dict = dict(zip(itr_df["subject"], itr_df["best_itr"]))

# Load participant demographic info (including sex)
df = pd.read_csv(participants_file, sep="\t")

# Group ITR values by participant sex
male_itrs = []
female_itrs = []

for subj_id, itr_value in itr_dict.items():
    if np.isnan(itr_value):
        continue  # Skip entries with missing ITR values
    row = df[df["participant_id"] == subj_id]
    if row.empty:
        continue  # Skip if participant metadata is missing
    sex = row["sex"].values[0]
    if sex == "m":
        male_itrs.append(itr_value)
    elif sex == "f":
        female_itrs.append(itr_value)

# Compute mean and standard deviation for each group
means = [np.mean(female_itrs), np.mean(male_itrs)]
stds = [np.std(female_itrs, ddof=1), np.std(male_itrs, ddof=1)]
labels = ["Female", "Male"]
x_pos = np.arange(len(labels))

# Perform independent t-test between groups
tval, pval = ttest_ind(female_itrs, male_itrs, equal_var=False)
print(f"T-test: t = {tval:.2f}, p = {pval:.4f}")

# Initialize plot
plt.figure(figsize=(6, 5))

# Plot group means as bars with standard deviation as error bars
plt.bar(
    x_pos, means, yerr=stds, capsize=8,
    color=["#f2b5d4", "#a4c8f0"],
    edgecolor="black", width=0.6,
    label="Group Mean ± SD"
)

# Add individual subject points with jitter to avoid overlap
rng = np.random.default_rng(seed=42)
female_x = rng.normal(loc=x_pos[0], scale=0.05, size=len(female_itrs))
male_x = rng.normal(loc=x_pos[1], scale=0.05, size=len(male_itrs))

plt.scatter(
    female_x, female_itrs,
    color="darkred", alpha=0.8, s=60, edgecolor='k',
    label=f"Female subjects (n={len(female_itrs)})"
)
plt.scatter(
    male_x, male_itrs,
    color="darkblue", alpha=0.8, s=60, edgecolor='k',
    label=f"Male subjects (n={len(male_itrs)})"
)

# Display p-value above the bars
y_max = max(max(female_itrs), max(male_itrs))
plt.text(
    0.5, y_max + (0.05 * y_max),
    f"p = {pval:.3f}",
    ha='center', fontsize=10, fontweight='bold'
)

# Configure axis labels and grid
plt.xticks(x_pos, labels, fontsize=11)
plt.ylabel("Best ITR (bits/min)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Add legend positioned outside the plot area
plt.legend(
    fontsize=9,
    loc='upper left',
    bbox_to_anchor=(1.02, 1)
)

plt.tight_layout()
plt.show()
