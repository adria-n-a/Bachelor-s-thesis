#!/project/2422139.01/venv/bin/python
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)
Saves, per subject/task/run:
 - freq:        frequency vector (Hz)
 - psd_db:      empirical PSD in dB(µV²/Hz)
 - model_db:    fitted model g(f) in dB(µV²/Hz)
 - background:  1/f component g1(f) in dB(µV²/Hz)
 - gauss1/gauss2: two Gaussian components in dB
 - parameters: all nine fit parameters plus feat
"""
import sys
sys.path.append(r"D:\radboud\courses\Thesis\scripts")
import mne
from mnelab_read_raw import read_raw_xdf as read_raw
import numpy as np
import os
import pyxdf
from scipy.optimize import curve_fit

def g1(f, lamb, k1, k2):
    return k1 + k2 / f**lamb

def g2(f, mu, sigma, k):
    return k * (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-.5 * (f-mu)**2 / sigma**2)

def g(f, lamb, mu1, mu2, sigma1, sigma2, k1, k2, k3, k4):
    return g1(f, lamb, k1, k2) + g2(f, mu1, sigma1, k3) + g2(f, mu2, sigma2, k4)

def main(subject, task, run, data_dir):
    print(f"Processing {subject} {task} run {run}")

    # 1) Load & preprocess
    fn = os.path.join(data_dir, "sourcedata", f"sub-{subject}", "ses-S001", "eeg",
                      f"sub-{subject}_ses-S001_task-{task}_run-{run:03d}_eeg.xdf")
    streams = pyxdf.resolve_streams(fn)
    bio_id = streams[[s["name"] for s in streams].index("BioSemi")]["stream_id"]
    raw = read_raw(fn, stream_ids=[bio_id])
    raw = raw.drop_channels([f"EX{i}" for i in range(1,9)] + [f"AIB{i}" for i in range(1,33)])
    montage = mne.channels.make_standard_montage("biosemi64")
    raw = raw.rename_channels({old:new for old,new in zip(
        [f"A{i}" for i in range(1,33)] + [f"B{i}" for i in range(1,33)],
        montage.ch_names)})
    # binarize trigger channel
    raw._data[0] -= raw._data[0].min()
    raw._data[0, raw._data[0]>0] = 1

    events = mne.find_events(raw, stim_channel="Trig1", verbose=False)
    epo = mne.Epochs(raw, events, tmin=0, tmax=120, baseline=None,
                     picks="eeg", preload=True, verbose=False)
    epo.set_montage(montage, on_missing="ignore")
    epo = mne.preprocessing.compute_current_source_density(epo, verbose=False)

    # 2) Compute PSD (Welch)
    psd = epo.copy().pick("Oz").compute_psd(fmin=0.5, fmax=30, verbose=False)
    f = psd.freqs                 # shape (n_freqs,)
    psd_uv2 = psd.get_data().flatten() * 1e12   # from V²/Hz to µV²/Hz
    psd_db  = 10 * np.log10(psd_uv2)            # to dB

    # 3) Fit parametric model on dB y
    y = psd_db
    p0 = [0.2, 10, 20, 5, 5, 1e-6, 25, 10, 5]
    bounds = ([0, 6,16, 0,0, 0,0,0,0], [np.inf,14,24,20,20, np.inf,np.inf,np.inf,np.inf])
    popt, _ = curve_fit(g, f, y, p0=p0, bounds=bounds)

    # unpack
    lamb, mu1, mu2, sigma1, sigma2, k1, k2, k3, k4 = popt
    feat1 = g(mu1, *popt) - g1(mu1, lamb, k1, k2)
    feat2 = g(mu2, *popt) - g1(mu2, lamb, k1, k2)

    # 4) build model components (in dB)
    bg      = 10 * np.log10( (g1(f, lamb, k1, k2)) )
    gauss1  = 10 * np.log10( (g2(f, mu1, sigma1, k3)) )
    gauss2  = 10 * np.log10( (g2(f, mu2, sigma2, k4)) )
    full_db = 10 * np.log10( g(f, *popt) )

    # 5) save everything
    outpath = os.path.join(data_dir, "derivatives", subject,
                           f"{subject}_rs_task-{task}_run-{run:03d}.npz")
    np.savez(outpath,
             freq=f,
             psd_db=psd_db,
             background_db=bg,
             gauss1_db=gauss1,
             gauss2_db=gauss2,
             model_db=full_db,
             lamb=lamb, mu1=mu1, mu2=mu2,
             sigma1=sigma1, sigma2=sigma2,
             k1=k1, k2=k2, k3=k3, k4=k4,
             feat1=feat1,feat2=feat2)

    print(f"  saved: {outpath}")

if __name__ == "__main__":
    subjects = ["VPpdca", "VPpdcb", "VPpdcc", "VPpdcd", "VPpdce", "VPpdcf", "VPpdcg", "VPpdch", "VPpdci", "VPpdcj", "VPpdck",
        "VPpdcl", "VPpdcm", "VPpdcn", "VPpdco", "VPpdcp", "VPpdcq", "VPpdcr", "VPpdcs", "VPpdct", "VPpdcu", "VPpdcv",
        "VPpdcw", "VPpdcx", "VPpdcy", "VPpdcz" ]
    tasks    = ["rstate_open", "rstate_closed"]
    runs     = [2, 2]
    for subj in subjects:
        for ti, task in enumerate(tasks):
            for run in range(1, runs[ti] + 1):
                main(subj, task, run, r"D:\radboud\courses\Thesis\sourcedata.tar")
