import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === SETTINGS ===
base_dir   = r"D:\radboud\courses\Thesis\sourcedata.tar"
csv_path   = os.path.join(base_dir, "group_itr_per_subject.csv")
subjects   = ["VPpdcx", "VPpdcg", "VPpdcb"]
accuracy_thresh = 0.95

# Helper: converts subject ID → "Subject A", etc.
def get_subject_label(subj_id):
    return f"Subject {subj_id[-1].upper()}"

# === 1) Load ITR summary CSV ===
df = pd.read_csv(csv_path).set_index("subject")

# === 2) Load time–accuracy–ITR curves for selected subjects ===
subject_curves = {}
for subj in subjects:
    npz_path = os.path.join(base_dir, "derivatives", subj, f"{subj}_itr_curve_mseq.npz")
    data = np.load(npz_path)
    subject_curves[subj] = {
        "time":     data["time_windows"],
        "accuracy": data["accuracy"],
        "itr":      data["itr"]
    }

# === 3) Determine global axis limits ===
itr_values = [subject_curves[s]["itr"] for s in subjects]
acc_values = [subject_curves[s]["accuracy"] for s in subjects]

itr_min = min(np.min(arr) for arr in itr_values)
itr_max = max(np.max(arr) for arr in itr_values)
acc_min = 0.0
acc_max = 1.05

# === 4) Plot 1×3 grid (ITR + Accuracy per subject) ===
fig, axes = plt.subplots(1, len(subjects), figsize=(5 * len(subjects), 4), sharey=True)

for ax, subj in zip(axes, subjects):
    curves = subject_curves[subj]
    t   = curves["time"]
    acc = curves["accuracy"]
    itr = curves["itr"]

    t_best   = df.loc[subj, "best_time"]
    itr_best = df.loc[subj, "best_itr"]

    # Left axis: ITR
    l1, = ax.plot(t, itr, color='green', alpha=0.5, label="ITR (bits/min)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title(get_subject_label(subj))
    ax.set_ylim(itr_min, itr_max)
    ax.axvline(t_best, color='black', linestyle="-", alpha=0.2)
    ax.text(0.55, 0.95, f"{itr_best:.2f} bits/min",
            transform=ax.transAxes,
            ha='right', va='center',
            fontsize=10, color='black')

    # Right axis: Accuracy
    ax2 = ax.twinx()
    l2, = ax2.plot(t, acc, color='purple', alpha=0.5, label="Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(acc_min, acc_max)
    ax2.axhline(accuracy_thresh, color='black', linestyle="--", alpha=1)

    # Add accuracy threshold label
    xmax = t.max()
    ax2.text(xmax, accuracy_thresh, "95% accuracy", ha='right', va='top', fontsize=9)

    # Shared legend
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="lower right", fontsize=8)

plt.tight_layout()
plt.show()
