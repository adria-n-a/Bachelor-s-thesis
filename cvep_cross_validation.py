#!/project/2422139.01/venv/bin/python
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)
"""

import os
import numpy as np
import pyntbci  # custom BCI toolbox for classification and ITR

def main(subject, data_dir):
    print(subject)

    # Define which stimulus codes (conditions) to use
    codes = ["mseq"]

    n_folds = 4  # Number of folds for cross-validation
    iti = 1.0  # Inter-trial interval (added to trial duration when computing ITR)

    # Create array to store decoding accuracy for each condition and fold
    accuracy = np.zeros((len(codes), n_folds))

    for i_code, code in enumerate(codes):

        # Load the preprocessed EEG data for this subject and code
        tmp = np.load(os.path.join(r"D:\radboud\courses\Thesis\sourcedata.tar",  "derivatives", subject, f"{subject}_cvep_{code}.npz"))
        X = tmp["X"]  # EEG data: trials × channels × samples
        y = tmp["y"]  # Labels: target (1) / non-target (0)
        V = tmp["V"]  # Reference stimulus codes
        fs = tmp["fs"]  # Sampling frequency
        del tmp  # Clean up memory
        print("X shape:", X.shape, "| # Trials:", X.shape[0])
        print("V shape:", V.shape)
        print("Unique y:", np.unique(y, return_counts=True))

        # Create fold indices for cross-validation
        folds = np.arange(n_folds).repeat(X.shape[0] / n_folds)[:X.shape[0]]

        for i_fold in range(n_folds):
            # Split data into training and test sets for this fold
            X_trn, y_trn = X[folds != i_fold, :, :], y[folds != i_fold]
            X_tst, y_tst = X[folds == i_fold, :, :], y[folds == i_fold]

            # Initialize rCCA classifier from pyntbci
            rcca = pyntbci.classifiers.rCCA(
                stimulus=V,
                fs=fs,
                event="refe",  # reference event
                onset_event=True,
                encoding_length=0.3  # duration of the encoding segment
            )

            # Train classifier
            rcca.fit(X_trn, y_trn)

            # Predict test labels and compute accuracy
            yh_tst = rcca.predict(X_tst)
            accuracy[i_code, i_fold] = np.mean(yh_tst == y_tst)

        # Print mean accuracy for this code
        print(f"\tCondition: {code}: {accuracy[i_code, :].mean()}")

    # Set trial duration (e.g., 4.2 seconds per trial)
    duration = np.full(accuracy.shape, 4.2)

    # Compute ITR using accuracy and trial duration + ITI
    itr = pyntbci.utilities.itr(V.shape[0], accuracy, duration + iti)

    # Save accuracy, duration, and ITR for this subject
    np.savez(os.path.join(data_dir, "derivatives", subject, f"{subject}_cvep_cross_validation.npz"),
             accuracy=accuracy, duration=duration, itr=itr, codes=codes)

# Run the script for all subjects
if __name__ == "__main__":
    subjects = [
        "VPpdca", "VPpdcb", "VPpdcc", "VPpdcd", "VPpdce", "VPpdcf", "VPpdcg", "VPpdch", "VPpdci", "VPpdcj", "VPpdck",
        "VPpdcl", "VPpdcm", "VPpdcn", "VPpdco", "VPpdcp", "VPpdcq", "VPpdcr", "VPpdcs", "VPpdct", "VPpdcu", "VPpdcv",
        "VPpdcw", "VPpdcx", "VPpdcy", "VPpdcz"]
    
    for subject in subjects:
        main(subject, r"D:\radboud\courses\Thesis\sourcedata.tar")
