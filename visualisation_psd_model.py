import os
import numpy as np
import matplotlib.pyplot as plt

# === 1) Define model components in dB ===

def g1_db(f, lamb, k1, k2):
    """1/f background component in dB."""
    return k1 + k2 / (f ** lamb)

def gaussian_db(f, mu, sigma, k):
    """Gaussian bump component in dB."""
    return k * np.exp(-0.5 * ((f - mu) / sigma) ** 2)

def full_model_db(f, lamb, mu1, mu2, sigma1, sigma2, k1, k2, k3, k4):
    """Full parametric model: background + two Gaussian bumps."""
    bg  = g1_db(f, lamb, k1, k2)
    a1  = gaussian_db(f, mu1, sigma1, feat1)
    a2  = gaussian_db(f, mu2, sigma2, feat2)
    return bg + a1 + a2

# === 2) Load PSD and parameters for one subject ===

subject = "VPpdca"  # change to desired subject
subject_label = f"Subject {subject[-1].upper()}"

data_dir = r"D:\radboud\courses\Thesis\sourcedata.tar"
fn = os.path.join(data_dir, "derivatives", subject, f"{subject}_rs_task-rstate_open_run-001.npz")
data = np.load(fn)

# PSD data
f = data["freq"]
psd_db = data["psd_db"]

# Fitted model parameters
lamb, mu1, mu2 = data["lamb"], data["mu1"], data["mu2"]
sigma1, sigma2 = data["sigma1"], data["sigma2"]
k1, k2, k3, k4 = data["k1"], data["k2"], data["k3"], data["k4"]
feat1 = data["feat1"]
feat2 = data["feat2"]

# === 3) Rebuild model and components ===

model_db = full_model_db(f, lamb, mu1, mu2, sigma1, sigma2, k1, k2, feat1, feat2)
bg_db    = g1_db(f, lamb, k1, k2)
alpha_db = gaussian_db(f, mu1, sigma1, feat1)
beta_db  = gaussian_db(f, mu2, sigma2, feat2)

# === 4) Plot PSD and model ===

plt.figure(figsize=(8, 4))

# Plot empirical PSD
plt.fill_between(f, psd_db, color="lightgray", alpha=0.6, label="Empirical PSD (dB)")

# Plot model and components
plt.plot(f, model_db, color="blue", lw=2, label="Full model (dB)")
plt.plot(f, bg_db,    color="black", ls="--", label="1/f background")
plt.plot(f, alpha_db, color="green", ls="-.", label="α bump")
plt.plot(f, beta_db,  color="purple", ls="-.", label="β bump")

# Plot k1 offset line
plt.axhline(k1, color="darkgreen", ls=":", label="k₁ offset")

# Annotate peak frequencies and PBCs
y1 = np.interp(mu1, f, model_db)
y2 = np.interp(mu2, f, model_db)
plt.axvline(mu1, color="red", ls="--", label=f"μ₁ = {mu1:.2f} Hz")
plt.axvline(mu2, color="purple", ls="--", label=f"μ₂ = {mu2:.2f} Hz")
plt.text(mu1, y1 + 1, f"PBC ≈ {feat1:.2f} dB", color="black", ha="center", va="bottom")
plt.text(mu2, y2 + 1, f"PBC ≈ {feat2:.2f} dB", color="black", ha="center", va="bottom")

# Axis and layout settings
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.title(f"{subject_label}: PSD & Model Components (dB)")
plt.legend(loc="upper right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
