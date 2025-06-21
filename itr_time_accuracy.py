import pandas as pd
import matplotlib.pyplot as plt

# Load per-subject performance data
df = pd.read_csv(r"D:\radboud\courses\Thesis\sourcedata.tar\group_itr_per_subject.csv")

# Extract time to reach best ITR, best ITR value, and accuracy in percentage
times = df['best_time']
itr = df['best_itr']
acc = df['best_accuracy'] * 100  # convert from [0–1] to %

# Create scatter plot: x = time, y = ITR, color-coded by accuracy
plt.figure(figsize=(8, 5))
sc = plt.scatter(times, itr, c=acc, cmap='viridis', s=100, edgecolors='k', vmin=70, vmax=100)
plt.colorbar(sc, label='Best Accuracy (%)')

plt.xlabel('Time to Reach Best Performance (s)')
plt.ylabel('Best ITR (bits/min)')
plt.title('Per-Subject Best ITR vs. Time (color = Best Accuracy)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
