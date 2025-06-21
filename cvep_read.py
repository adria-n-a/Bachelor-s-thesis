#!/project/2422139.01/venv/bin/python
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)
"""

import os
import sys
#sys.path.append(r"D:\radboud\courses\Thesis\scripts\mnelab_read_raw.py")
#from mnelab_read_raw import read_raw_xdf as read_raw

import mne
from mnelab_read_raw import read_raw_xdf as read_raw
import numpy as np
import pyxdf

# Load EEG data using MNE
#xdf_file = r"D:\radboud\courses\Thesis\sourcedata.tar\sourcedata\sub-VPpdca\ses-S001\eeg\sub-VPpdca_ses-S001_task-cvep_run-001_eeg.xdf"
#streams = pyxdf.load_xdf(xdf_file)
#raw = mne.io.RawArray(xdf_file)  # Alternative method


def main(subject, data_dir, repo_dir):
    print(subject)
    mapping = {
        "mseq_61_shift_2": "mseq",
        "mmseq_61_shift_2": "m-mseq",
        "debruijn_2_6_shift_2": "debruijn",
        "mdebruijn_2_6_shift_2": "m-debruijn",
        "golay_64_1_shift_2": "golay",
        "mgolay_64_1_shift_2": "m-golay",
        "gold_61_6521_shift_2": "gold",
        "mgold_61_6521_shift_2": "m-gold",
        "gold_61_6521": "gold-set",
        "mgold_61_6521": "m-gold-set",
    }

    fs = 120
    pr = 60

    n_runs = 8
    l_freq = 6
    h_freq = 40
    baseline = None
    trial_time_1 = 63 / pr * 4  # 126 / pr * 2 (m-sequence, Gold)
    trial_time_2 = 64 / pr * 4  # 128 / pr * 2 (de Bruijn, Golay)

    # Create output folder
    if not os.path.exists(os.path.join(data_dir, "derivatives", subject)):
        os.makedirs(os.path.join(data_dir, "derivatives", subject))

    eeg = []
    labels = []
    conditions = []
    for i_run in range(n_runs):
        fn = os.path.join(data_dir, "sourcedata", f"sub-{subject}", "ses-S001", "eeg",
                          f"sub-{subject}_ses-S001_task-cvep_run-{1 + i_run:03d}_eeg.xdf")

        # Load EEG
        streams = pyxdf.resolve_streams(fn)
        names = [stream["name"] for stream in streams]
        stream_id = streams[names.index("BioSemi")]["stream_id"]
        raw = read_raw(fn, stream_ids=[stream_id])

        # Adjust marker channel data
        raw._data[0, :] -= np.median(raw._data[0, :])
        raw._data[0, :] = np.diff(np.concatenate((np.zeros(1), raw._data[0, :]))) > 0

        # Read events
        events = mne.find_events(raw, stim_channel="Trig1", verbose=False)

        # Repair missing event VPpdcy run 8 trial 33
        if subject == "VPpdcy" and i_run == 7:
            streams = pyxdf.load_xdf(fn)[0]
            names = [stream["info"]["name"][0] for stream in streams]
            marker_stream = streams[names.index("KeyboardMarkerStream")]
            onsets = np.array([int(onset * raw.info["sfreq"])
                               for onset, marker in zip(marker_stream["time_stamps"], marker_stream["time_series"])
                               if marker[2] == "start_trial"])
            row = np.array([events[32, 0] + np.diff(onsets)[32], 0, 1])[np.newaxis, :]
            events = np.concatenate((events[:33, :], row, events[33:, :]), axis=0)

        # Filtering
        raw = raw.filter(l_freq=l_freq, h_freq=h_freq, picks=np.arange(1, 65), verbose=False)

        # Slicing
        # N.B. add 0.5 sec pre and post trial to capture filtering artefacts of downsampling (removed later on)
        # N.B. Use largest trial time (samples are cut away later)
        epo = mne.Epochs(raw, events=events, tmin=-0.5, tmax=trial_time_2 + 0.5, baseline=baseline, picks="eeg",
                         preload=True, verbose=False)

        # Resampling
        # N.B. Downsampling is done after slicing to maintain accurate stimulus timing
        epo = epo.resample(sfreq=fs, verbose=False)

        # Add EEG to database (trials channels samples)
        eeg.append(epo.get_data(tmin=0, tmax=trial_time_2, copy=True))

        # Load labels and conditions from marker stream
        streams = pyxdf.load_xdf(fn)[0]
        names = [stream["info"]["name"][0] for stream in streams]
        marker_stream = streams[names.index("KeyboardMarkerStream")]
        labels.extend([int(marker[3]) for marker in marker_stream["time_series"] if marker[2] == "target"])
        conditions.extend([marker[3].strip('"') for marker in marker_stream["time_series"] if marker[2] == "code"])

    # Extract data
    eeg = np.concatenate(eeg, axis=0).astype("float32")  # trials channels samples
    labels = np.array(labels).astype("uint8")

    # Loop conditions
    for condition in set(conditions):

        # Select trials
        idx = np.array([x == condition for x in conditions]).astype("bool_")
        X = eeg[idx, :, :].astype("float32")
        y = labels[idx].astype("uint8")

        # Set correct trial length
        if "mseq" in condition or "gold" in condition:
            X = X[:, :, :int(trial_time_1 * fs)]

        # Load codes
        fn = os.path.join(repo_dir, "experiment", "codes", f"{condition}.npz")
        V = np.load(fn)["codes"].T
        if condition == "gold-set":
            V[6, -1] = 0  # Fix (i.e., prevent) code 6 of gold-set creation of 7th event when cycling
        V = np.repeat(V, int(fs / pr), axis=1).astype("uint8")

        # Set condition name
        for old, new in mapping.items():
            if condition == old:
                condition = new

        # Print summary
        print("Condition:", condition)
        print("\tX:", X.shape)
        print("\ty:", y.shape)
        print("\tV:", V.shape)

        # Save data
        np.savez(os.path.join(data_dir, "derivatives", subject, f"{subject}_cvep_{condition}.npz"),
                 X=X, y=y, V=V, fs=fs)

if __name__ == "__main__":
    subjects = [
        "VPpdca", "VPpdcb", "VPpdcc", "VPpdcd", "VPpdce", "VPpdcf", "VPpdcg", "VPpdch", "VPpdci", "VPpdcj", "VPpdck",
        "VPpdcl", "VPpdcm", "VPpdcn", "VPpdco", "VPpdcp", "VPpdcq", "VPpdcr", "VPpdcs", "VPpdct", "VPpdcu", "VPpdcv",
        "VPpdcw", "VPpdcx", "VPpdcy", "VPpdcz"
    ]

    for subject in subjects:
        main(
            subject,
            r"D:\radboud\courses\Thesis\sourcedata.tar",     # <- actual data path
            r"D:\radboud\courses\Thesis"                      
        )

