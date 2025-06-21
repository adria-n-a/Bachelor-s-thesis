import os
import numpy as np
import matplotlib.pyplot as plt

# Set subject, task, and run configuration
BASE_DIR    = r"D:\radboud\courses\Thesis\sourcedata.tar"
PARTICIPANT = "VPpdca"
TASK        = "rstate_open"
RUN         = 1

# Convert participant ID to label format
subject_label = f"Subject {PARTICIPANT[-1].upper()}"

# Build the full path to the .npz file
npz_name = f"{PARTICIPANT}_rs_task-{TASK}_run-{RUN:03d}.npz"
npz_path = os.path.join(BASE_DIR, "derivatives", PARTICIPANT, npz_name)
if not os.path.isfile(npz_path):
    raise FileNotFoundError(f"Could not find:\n  {npz_path}")

# Load frequency-domain data and model components
data        = np.load(npz_path)
f           = data["freq"]
psd_db      = data["psd_db"]
bg_db       = data["background_db"]
g1_db       = data["gauss1_db"]
g2_db       = data["gauss2_db"]
model_db    = data["model_db"]
mu1, mu2    = data["mu1"], data["mu2"]
feat1, feat2 = data["feat1"], data["feat2"]
k1_db       = data["k1"]
k2          = data["k2"]
lamb        = data["lamb"]

# Convert all dB values to linear scale (μV²/Hz)
psd_lin   = 10 ** (psd_db / 10)
bg_lin    = 10 ** (bg_db / 10)
g1_lin    = 10 ** (g1_db / 10)
g2_lin    = 10 ** (g2_db / 10)
model_lin = 10 ** (model_db / 10)
k1_lin    = k1_db  # already scalar

# Compute theoretical 1/f power: k2 / f^lambda
f_nonzero = np.where(f == 0, 1e-6, f)
background_power = k2 / (f_nonzero ** lamb)

# Initialize the figure
plt.figure(figsize=(7, 4))

# Plot full spectrum and model components
plt.plot(f, psd_lin, linewidth=1.5, label="Full spectrum")
plt.plot(f, model_lin, color="blue", linewidth=2, label="Model fit")
plt.plot(f, bg_lin, color="gray", linestyle="--", linewidth=1.5, label="1/f background")
plt.plot(f, background_power, linestyle="--", color="orange", linewidth=1.5, label=r"$k_2 / f^\lambda$")
plt.axhline(y=k1_lin, color="green", linestyle="--", linewidth=1.5, label="Offset k₁")
plt.axvline(x=mu1, color="red", linestyle="--", linewidth=1.5, label=f"μ₁ = {mu1:.2f} Hz")
plt.axvline(x=mu2, color="purple", linestyle="--", linewidth=1.5, label=f"μ₂ = {mu2:.2f} Hz")

# Annotate peak bump contributions at mu1 and mu2
i1 = np.argmin(np.abs(f - mu1))
i2 = np.argmin(np.abs(f - mu2))
y1 = model_lin[i1]
y2 = model_lin[i2]
plt.text(mu1, y1, f"PBC ≈ {feat1:.2f}", color="red", ha="center", va="bottom")
plt.text(mu2, y2, f"PBC ≈ {feat2:.2f}", color="purple", ha="center", va="bottom")

# Axes configuration
plt.xlim(0, 30)
plt.ylim(0, 115)
plt.xticks(np.arange(0, 31, 5))
plt.yticks([20, 40, 60, 80, 100])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (a.u.)")
plt.title(f"Power Spectrum Model for {subject_label}")

plt.legend(loc="upper right", fontsize="small")
plt.grid(True, which='both', linestyle='-', alpha=0.5)
plt.tight_layout()
plt.show()
