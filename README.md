# Script Summaries
This document summarizes the purpose, inputs, and outputs of each script in the repository.

## mnelab_read_raw.py
**Description:**  
Custom version of the MNE XDF loader to read EEG streams, resample data, and convert markers to annotations.
**Input:**  
XDF file path and list of stream IDs.
**Output:**  
Returns an MNE Raw object with loaded and optionally resampled EEG data.

## rs_stats.py
**Description:**  
Fits a parametric model to resting-state EEG from Oz channel, saves spectral components and parameters per subject/task/run.
**Input:**  
XDF EEG files per subject, task, and run.
**Output:**  
NPZ files containing model components, parameters, and power spectra.

## visualize_psd_fit_and_peaks_oz.py
**Description:**  
Plots a subject's power spectrum and fitted model in linear scale, highlighting μ₁/μ₂ peaks and power bump contributions.
**Input:**  
NPZ file with model data and PSD for the Oz channel.
**Output:**  
A figure showing linear PSD with model components and annotations.

## visualisation_psd_model.py
**Description:**  
Visualizes the full PSD model for one subject including background, Gaussian bumps, and annotated peak contributions.
**Input:**  
NPZ file with frequency, PSD, and spectral model parameters.
**Output:**  
A plot showing the empirical PSD and model components with annotations.

## plot_mu1andmu2.py
**Description:**  
Plots the parametric spectrum for selected subjects with vertical lines for μ₁ and μ₂ peak frequencies.
**Input:**  
NPZ files with spectral model data including μ₁ and μ₂.
**Output:**  
PNG figure showing model fits and marked peaks for each subject.

## plot_sigma1and2.py
**Description:**  
Visualizes spectral models with shaded regions for ±σ around μ₁ and μ₂, indicating peak bandwidth.
**Input:**  
NPZ files with model data including frequency, μ, and σ values.
**Output:**  
PNG plot showing shaded bandwidths on spectral curves.

## compute_mu_variance.py
**Description:**  
Calculates the variance of the μ₁ peak frequency from Oz channel EEG data for each subject.
**Input:**  
XDF EEG recordings per subject in a structured directory.
**Output:**  
CSV file with per-subject μ₁ variance saved to disk.

## ppfactor_beta_extract.py
**Description:**  
Computes PPfactor and relative beta power for each subject's PSD, saves the metrics, and evaluates their predictive power for ITR.
**Input:**  
NPZ PSD files and group ITR CSV.
**Output:**  
CSV of computed metrics and PNG plots of regression fits with stats.

## count_peaks_reg.py
**Description:**  
Counts significant peaks in subject PSDs and uses them to predict ITR using leave-one-out regression.
**Input:**  
CSV of subject ITRs and NPZ files with peak features (k3, k4).
**Output:**  
CSV files with peak counts and regression results, plus a regression plot image.

## group_itr_analysis.py
**Description:**  
Computes and summarizes ITR performance curves for all subjects using CVEP decoding and saves best accuracy/ITR values.
**Input:**  
NPZ files with CVEP trials and target codes per subject.
**Output:**  
CSV of best ITR per subject and saved curves per subject.

## histogram_itr.py
**Description:**  
Plots a histogram of best ITR values across all subjects.
**Input:**  
CSV file containing best ITR values per subject.
**Output:**  
A histogram plot showing distribution of ITR scores.

## itr_time_accuracy.py
**Description:**  
Creates a scatter plot showing the relationship between time to best ITR and the ITR value, color-coded by accuracy.
**Input:**  
CSV file with columns: best_time, best_itr, and best_accuracy.
**Output:**  
Displays a scatter plot with color gradient representing accuracy percentage.

## visualization_learning curve.py
**Description:**  
Plots learning curves (accuracy and ITR over time) for selected subjects, highlighting the point of best performance.
**Input:**  
Group-level ITR CSV and NPZ time–accuracy–ITR curve files.
**Output:**  
A multi-subplot figure visualizing learning curves per subject.

## cvep_read.py
**Description:**  
Processes CVEP EEG data for each subject across 8 runs, extracting trials and saving cleaned, labeled datasets per condition.
**Input:**  
XDF EEG recordings and associated stimulus marker files.
**Output:**  
Condition-specific NPZ files containing preprocessed EEG data and labels.

## cvep_cross_validation.py
**Description:**  
Performs 4-fold cross-validation using rCCA classifier for CVEP decoding, then computes and saves ITR metrics per subject.
**Input:**  
Preprocessed EEG NPZ files for each subject and condition.
**Output:**  
NPZ file with cross-validated accuracy and ITR values per subject.

## barplot_itr_by_sex.py
**Description:**  
Plots a bar chart comparing ITR values by sex, including mean ± SD, individual points, counts, and p-value from t-test.
**Input:**  
CSV file with subject ITRs and TSV file with participant sex data.
**Output:**  
Displays a bar plot comparing male and female ITRs with statistical annotation.

## regression_itr_age.py
**Description:**  
Predicts ITR using participant age via leave-one-out regression. Saves predictions and visualization.
**Input:**  
Participant metadata TSV and group ITR CSV.
**Output:**  
CSV and PNG showing regression performance and statistical fit.

## mu_variance_reg.py
**Description:**  
Runs leave-one-out regression to assess the predictive power of μ₁ variance on ITR.
**Input:**  
CSV files with μ₁ variance and best ITR per subject.
**Output:**  
CSV of regression results and a PNG plot of fit with statistical annotation.

## run_linear_regression.py
**Description:**  
Performs univariate leave-one-out regressions to predict ITR using each parameter from the spectral model.
**Input:**  
NPZ files with spectral parameters and a CSV with subject ITRs.
**Output:**  
PNG scatter plots and console output for each parameter vs. ITR.

## linear_regression.py
**Description:**  
Provides helper functions to perform leave-one-out linear regression and generate annotated plots with confidence intervals.
**Input:**  
Feature values, ITR values, and subject IDs (from calling scripts).
**Output:**  
Returns prediction DataFrame; optionally saves scatter plot with confidence band.

## run_multivariate_reg.py
**Description:**  
Runs a multivariate leave-one-out regression to predict ITR from a combination of EEG features, peak stats, and demographics.
**Input:**  
NPZ and CSV files with spectral features, peak data, variance, age, and sex.
**Output:**  
CSV with predicted ITRs, PNG plot, and printed model coefficients.

## lasso_regression.py
**Description:**  
Performs Lasso regression to predict ITR from multiple features using leave-one-out cross-validation. Identifies predictive features and plots results.
**Input:**  
CSV and NPZ files with features like mu/variance, demographics, and beta metrics.
**Output:**  
CSV of predictions, PNG plot of results, and printed model coefficients.

## find_best_combo.py
**Description:**  
Identifies the best 3-feature combination to predict ITR using leave-one-out regression among multiple subject-level predictors.
**Input:**  
CSV files for various predictors (e.g., peak count, μ₁ variance, demographics, beta power).
**Output:**  
CSV and PNG file with best model predictions and plot of true vs predicted ITR.

## best_and_worst_itrs.py
**Description:**  
Visualizes spectral power for the top 3 and bottom 3 subjects based on their ITR scores.
**Input:**  
CSV file with per-subject ITRs and NPZ files with spectral data.
**Output:**  
Displays two side-by-side plots for spectral power of best and worst performers.

## feat_k1_k2.py
**Description:**  
Visualizes the spectral model of one subject, decomposing it into background and peak components and computing features.
**Input:**  
NPZ file containing spectral model parameters for one subject.
**Output:**  
A plot showing full spectrum, components, and annotations for key parameters (e.g., μ₁, μ₂, k₁, k₂).
