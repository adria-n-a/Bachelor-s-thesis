import numpy as np
import matplotlib.pyplot as plt
import os

# === Settings ===
subject = "VPpdca"
data_dir = r"D:\radboud\courses\Thesis\sourcedata.tar"
npz_path = os.path.join(data_dir, "derivatives", subject, f"{subject}_rs_task-rstate_closed_run-002.npz")

# === Load subject data ===
data = np.load(npz_path)
mu1 = data["mu1"].item()
mu2 = data["mu2"].item()
sigma1 = data["sigma1"].item()
sigma2 = data["sigma2"].item()
amp1 = data["k3"].item()
amp2 = data["k4"].item()
k1 = data["k1"].item()
k2 = data["k2"].item()
lamb = data["lamb"].item()

# === Frequency range and Gaussian ===
f = np.linspace(1, 30, 1000)

def gaussian(f, mu, sigma, amp):
    return amp * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((f - mu) / sigma) ** 2)

# === Compute components ===
background = k1 + k2 / f**lamb
peak1 = gaussian(f, mu1, sigma1, amp1)
peak2 = gaussian(f, mu2, sigma2, amp2)
spectrum = background + peak1 + peak2

# === Compute feats ===
f_idx1 = np.argmin(np.abs(f - mu1))
f_idx2 = np.argmin(np.abs(f - mu2))
feat1 = spectrum[f_idx1] - background[f_idx1]
feat2 = spectrum[f_idx2] - background[f_idx2]

# === Plot ===
plt.figure(figsize=(10, 6))
plt.plot(f, spectrum, label="Full Spectrum", color="navy")
plt.plot(f, background, '--', label="1/f Background", color="gray")
plt.axvline(mu1, color="red", linestyle="--", label=f"$\\mu_1$ = {mu1:.2f} Hz")
plt.axvline(mu2, color="purple", linestyle="--", label=f"$\\mu_2$ = {mu2:.2f} Hz")

# Annotate feats
plt.annotate(f"feat₁ ≈ {feat1:.2f}", xy=(mu1 + 0.5, spectrum[f_idx1] + 0.5), fontsize=11, color="darkred")
plt.annotate(f"feat₂ ≈ {feat2:.2f}", xy=(mu2 + 0.5, spectrum[f_idx2] + 0.5), fontsize=11, color="indigo")

# Background components
plt.hlines(k1, xmin=1, xmax=30, colors="green", linestyles="--", label=f"$k_1$ (baseline) = {k1:.2f}")
plt.plot(f, k2 / f**lamb, color="orange", linestyle="--", label="$k_2 / f^\\lambda$")

# === Final touches ===
plt.title(f"Power Spectrum Model with $\\mu_1$, $\\mu_2$, $k_1$, $k_2$ for Subject {subject}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (a.u.)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
