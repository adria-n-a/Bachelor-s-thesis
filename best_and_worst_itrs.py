import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Convert subject ID to a readable label like "Subject A"
def get_subject_label(subj):
    return f"Subject {subj[-1].upper()}"

# Load ITR values per subject
itr_df = pd.read_csv(r"D:\radboud\courses\Thesis\sourcedata.tar\group_itr_per_subject.csv")

# Select top 3 and bottom 3 subjects based on best ITR score
top3 = itr_df.nlargest(3, 'best_itr')['subject'].tolist()
bottom3 = itr_df.nsmallest(3, 'best_itr')['subject'].tolist()

# Create side-by-side subplots for top and bottom performers
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Define color palettes for the two groups
green_shades = ["#A8D5BA", "#6BBF59", "#3C8031"]
red_shades = ["#F7C6C7", "#EF6F6C", "#B03A2E"]

# Load spectral data and plot each subject on the given axes
def plot_subjects(ax, subj_list, title, color_list):
    for i, subj in enumerate(subj_list):
        pattern = rf"D:\radboud\courses\Thesis\sourcedata.tar\derivatives\{subj}\{subj}_rs_task-rstate_open_run-001.npz"
        files = glob.glob(pattern)
        if not files:
            print(f"No file found for {subj}")
            continue
        data = np.load(files[0])
        label = get_subject_label(subj)
        color = color_list[i % len(color_list)]
        ax.plot(data['freq'], data['model_db'], label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.legend(loc='upper right', fontsize=9)

# Plot spectra for the top and bottom groups
plot_subjects(ax1, top3, "Top 3 ITR Subjects", green_shades)
plot_subjects(ax2, bottom3, "Bottom 3 ITR Subjects", red_shades)

# Add shared y-axis label and title
ax1.set_ylabel("Power (dB μV²/Hz)")
fig.suptitle("Parametric Models by ITR Rank")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
