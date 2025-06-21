#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leave_one_out_regression.py

Provides `run_regression()` and `plot_regression_results()` for leave-one-subject-out
linear regression, including OLS fit with 95% confidence intervals, marginal histograms,
and annotated scatter plots with CV R² and Pearson statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec


def run_regression(
    feature_values: np.ndarray,
    itr_values: np.ndarray,
    subjects: list,
    regressor=None,
    cv_split=None
):
    # Check input lengths
    if not (len(feature_values) == len(itr_values) == len(subjects)):
        raise ValueError("feature_values, itr_values, and subjects must have the same length")

    X = feature_values.reshape(-1, 1)
    y = itr_values

    regressor = regressor or LinearRegression()
    cv_split = cv_split or LeaveOneOut()

    preds = np.empty_like(y)
    for train_idx, test_idx in cv_split.split(X):
        regressor.fit(X[train_idx], y[train_idx])
        preds[test_idx] = regressor.predict(X[test_idx])

    cv_r2 = r2_score(y, preds)
    r_val, p_val = pearsonr(feature_values, y)

    df_out = pd.DataFrame({
        "subject": subjects,
        "feature": feature_values,
        "itr_true": y,
        "itr_pred": preds
    })

    return df_out, r_val, p_val, cv_r2


def plot_regression_results(
    df_out: pd.DataFrame,
    feature_name: str,
    itr_name: str,
    r_val: float,
    p_val: float,
    cv_r2: float,
    output_png: str
):
    x = df_out["feature"].values
    y = df_out["itr_true"].values

    # Fit OLS model for CI
    X_sm = sm.add_constant(x)
    ols_model = sm.OLS(y, X_sm).fit()
    xs_grid = np.linspace(x.min(), x.max(), 200)
    Xg_sm = sm.add_constant(xs_grid)
    ps = ols_model.get_prediction(Xg_sm).summary_frame(alpha=0.05)

    y_mean = ps["mean"].values
    y_ci_low = ps["mean_ci_lower"].values
    y_ci_high = ps["mean_ci_upper"].values

    # Font sizes
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=16)

    # Set up layout
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                  left=0.08, right=0.96, bottom=0.08, top=0.94,
                  wspace=0.05, hspace=0.05)

    # Scatter + fit + CI
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(x, y, color="C0", alpha=0.7, edgecolor="k", linewidth=0.5, s=40)
    ax.plot(xs_grid, y_mean, color="C1", linewidth=2)
    ax.fill_between(xs_grid, y_ci_low, y_ci_high, color="C1", alpha=0.3)
    ax.set_xlabel(feature_name)
    ax.set_ylabel(itr_name)

    # Annotated legend with CV R², Pearson r, and p
    from matplotlib.lines import Line2D
    txt = f"CV R² (LOO) = {cv_r2:.2f}\nPearson r = {r_val:.2f}, p = {p_val:.3f}"
    dummy = Line2D([0], [0], linestyle="none")
    ax.legend([dummy], [txt], loc="lower right", frameon=True, facecolor="white", edgecolor="gray")

    # Marginal histogram (x-axis)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_top.hist(x, bins=20, color="C0", alpha=0.7)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("Count")
    for spine in ["right", "top", "left"]:
        ax_top.spines[spine].set_visible(False)

    # Optional y-marginal histogram
    # ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    # ax_right.hist(y, bins=20, orientation="horizontal", color="C0", alpha=0.7)
    # ax_right.tick_params(axis="y", labelleft=False)
    # ax_right.set_xlabel("Count")
    # for spine in ["right", "top", "bottom"]:
    #     ax_right.spines[spine].set_visible(False)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    fig.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
