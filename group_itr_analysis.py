import numpy as np
import os
import pandas as pd
from pyntbci.classifiers import rCCA
from pyntbci.utilities import itr
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Subject list and general parameters
subjects = [
    "VPpdca", "VPpdcb", "VPpdcc", "VPpdcd", "VPpdce", "VPpdcf", "VPpdcg", "VPpdch", "VPpdci", "VPpdcj", "VPpdck",
    "VPpdcl", "VPpdcm", "VPpdcn", "VPpdco", "VPpdcp", "VPpdcq", "VPpdcr", "VPpdcs", "VPpdct", "VPpdcu", "VPpdcv",
    "VPpdcw", "VPpdcx", "VPpdcy", "VPpdcz"
]
code = "mseq"
data_dir = r"D:\radboud\courses\Thesis\sourcedata.tar"
iti = 1.0
fs_default = 120

# Cross-validation and curve sampling configuration
n_splits = 4
num_windows = 100

# Compute ITR and accuracy curves per subject
def run_itr_for_subject(subject):
    try:
        path = os.path.join(data_dir, "derivatives", subject, f"{subject}_cvep_{code}.npz")
        data = np.load(path)
        X = data["X"]
        y = data["y"]
        V = data["V"]
        fs = int(data["fs"]) if "fs" in data else fs_default
        del data

        time_windows = np.linspace(0.2, 4.2, num_windows)
        accuracy_curve = []
        itr_curve = []

        print(f"Starting ITR curve for {subject} with {num_windows} windows and {n_splits}-fold CV...")
        for i_t, t in enumerate(time_windows, start=1):
            t_samples = int(t * fs)
            X_t = X[:, :, :t_samples]

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            acc_folds = []
            itr_folds = []

            for i_fold, (train_idx, test_idx) in enumerate(kf.split(X_t), start=1):
                print(f"  Time window {i_t}/{num_windows}: Fold {i_fold}/{n_splits}...")
                X_train, X_test = X_t[train_idx], X_t[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                clf = rCCA(
                    stimulus=V,
                    fs=fs,
                    event="refe",
                    onset_event=True,
                    encoding_length=t,
                    gamma_x=1e-3,
                    gamma_m=1e-3
                )
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                acc = np.mean(y_pred == y_test)
                acc_folds.append(acc)
                total_time = t + iti
                itr_folds.append(itr(n=V.shape[0], p=acc, t=total_time))

            mean_acc = np.mean(acc_folds)
            mean_itr = np.mean(itr_folds)
            accuracy_curve.append(mean_acc)
            itr_curve.append(mean_itr)
            print(f"    --> Window {i_t}: mean accuracy={mean_acc:.3f}, mean ITR={mean_itr:.2f} bits/min")

        accuracy_curve = np.array(accuracy_curve)
        itr_curve = np.array(itr_curve)

        # Check when accuracy crosses threshold
        accuracy_thresh = 0.95
        valid_mask = accuracy_curve >= accuracy_thresh
        if np.any(valid_mask):
            first_idx = np.argmax(valid_mask)
            print(f"{subject}: Accuracy reached {accuracy_thresh*100:.0f}% at {time_windows[first_idx]:.2f}s | ITR = {itr_curve[first_idx]:.2f} bits/min")
        else:
            print(f"{subject}: Accuracy never reached {accuracy_thresh*100:.0f}%")

        save_path = os.path.join(data_dir, "derivatives", subject, f"{subject}_itr_curve_{code}.npz")
        np.savez(save_path, time_windows=time_windows, accuracy=accuracy_curve, itr=itr_curve)
        print(f"Saved ITR curve for {subject} to {save_path}\n")

    except Exception as e:
        print(f"Error for {subject}: {e}")

# Run ITR computation across all subjects
for subject in subjects:
    run_itr_for_subject(subject)

# Summarize group-level performance for subjects who reached the accuracy threshold
best_times = []
best_itrs = []
best_accs = []
valid_subjects = []
accuracy_thresh = 0.95

for subject in subjects:
    try:
        npz_path = os.path.join(data_dir, "derivatives", subject, f"{subject}_itr_curve_{code}.npz")
        data = np.load(npz_path)
        time_windows = data["time_windows"]
        accuracy_curve = data["accuracy"]
        itr_curve = data["itr"]

        valid_mask = accuracy_curve >= accuracy_thresh
        if not np.any(valid_mask):
            continue

        best_idx = np.argmax(valid_mask)
        best_times.append(time_windows[best_idx])
        best_itrs.append(itr_curve[best_idx])
        best_accs.append(accuracy_curve[best_idx])
        valid_subjects.append(subject)

    except FileNotFoundError:
        continue
    except Exception as e:
        print(f"Error summarizing {subject}: {e}")

# Compute group-level averages and standard deviations
best_times = np.array(best_times)
best_itrs = np.array(best_itrs)
best_accs = np.array(best_accs)

print("Group Summary:")
print(f" Avg threshold time: {np.mean(best_times):.2f}s (SD {np.std(best_times):.2f})")
print(f" Avg ITR at threshold: {np.mean(best_itrs):.2f} bits/min (SD {np.std(best_itrs):.2f})")

# Save summary results
np.save(os.path.join(data_dir, "group_best_times.npy"), best_times)
np.save(os.path.join(data_dir, "group_best_itrs.npy"), best_itrs)

results_df = pd.DataFrame({
    "subject": valid_subjects,
    "best_time": best_times,
    "best_itr": best_itrs,
    "best_accuracy": best_accs
})
df_path = os.path.join(data_dir, "group_itr_per_subject.csv")
results_df.to_csv(df_path, index=False)
print(f"Saved per-subject results to {df_path}")
