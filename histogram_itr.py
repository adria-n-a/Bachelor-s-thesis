import pandas as pd
import matplotlib.pyplot as plt

# Load ITR values per subject from CSV
csv_path = r"D:\radboud\courses\Thesis\sourcedata.tar\group_itr_per_subject.csv"
df = pd.read_csv(csv_path)

# Get the best_itr column for plotting
best_itrs = df['best_itr']

# Create a histogram of ITR values
plt.figure(figsize=(8, 5))
plt.hist(best_itrs, bins=15, color='lightsteelblue', edgecolor='black')

plt.title("Distribution of Best ITRs Across Subjects")
plt.xlabel("Best ITR (bits/min)")
plt.ylabel("Number of Subjects")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
