import os
import numpy as np
import matplotlib.pyplot as plt

# Define subjects, task, and run configuration
subjects = ["VPpdca", "VPpdcb", "VPpdcc"]
task = "rstate_open"
run = 1

# Define paths for data and output figure
base_dir = r"D:\radboud\courses\Thesis\sourcedata.tar"
figures_dir = r"D:\radboud\courses\Thesis\figures"
os.makedirs(figures_dir, exist_ok=True)
output_path = os.path.join(figures_dir, "mu_sigma_shading.png")

# Generate readable label from subject ID
def get_label(subj):
    return f"Subject {subj[-1].upper()}"

# Load model and frequency data for each subject
data_dict = {}
valid_subjects = []
for subj in subjects:
    fname = f"{subj}_rs_task-{task}_run-{run:03d}.npz"
    fullpath = os.path.join(base_dir, "derivatives", subj, fname)
    if not os.path.isfile(fullpath):
        print(f"File not found for subject '{subj}':\n  {fullpath}")
        continue
    arrs = np.load(fullpath)
    data_dict[subj] = {
        "freq": arrs["freq"],
        "model_db": arrs["model_db"],
        "mu1": float(arrs["mu1"]),
        "mu2": float(arrs["mu2"]),
        "sigma1": float(arrs["sigma1"]),
        "sigma2": float(arrs["sigma2"]),
    }
    valid_subjects.append(subj)

if not valid_subjects:
    raise RuntimeError("No valid .npz files found. Check configuration or file paths.")

# Create subplots for each subject
num_plots = len(valid_subjects)
fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4), sharey=True)
if num_plots == 1:
    axes = [axes]

for ax, subj in zip(axes, valid_subjects):
    dd = data_dict[subj]
    freq = dd["freq"]
    model = dd["model_db"]
    mu1, mu2 = dd["mu1"], dd["mu2"]
    sigma1, sigma2 = dd["sigma1"], dd["sigma2"]

    ax.plot(freq, model, color="tab:blue", lw=2, label="Parametric Model")

    # Shade ±σ regions around mu1 and mu2
    ax.axvspan(mu1 - sigma1, mu1 + sigma1, color="tab:red", alpha=0.2,
               label="σ₁ band (μ₁±σ₁)")
    ax.axvspan(mu2 - sigma2, mu2 + sigma2, color="tab:green", alpha=0.2,
               label="σ₂ band (μ₂±σ₂)")

    ax.set_title(get_label(subj))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB µV²/Hz)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close(fig)

print(f"\nFigure with shaded sigma bands saved to:\n  {output_path}")
