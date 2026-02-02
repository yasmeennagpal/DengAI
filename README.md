Dengue Cases Prediction (DengAI)
This repository contains a machine learning pipeline developed for the DengAI: Predicting Disease Spread competition hosted by DrivenData. The goal is to predict the number of dengue fever cases in San Juan and Iquitos on a weekly basis.

Model Overview
The solution utilizes a Triple Ensemble approach to balance trend capturing with error minimization:

HistGradientBoostingRegressor (Poisson Loss): Optimized for count-based data and general trends.

HistGradientBoostingRegressor (MAE Loss): Optimized specifically for the Mean Absolute Error metric.

Random Forest Regressor: Acts as a stabilizer to prevent over-prediction of seasonal spikes.

Data Processing & Engineering
To address the high variance and noise in the climate data, the following steps were implemented:

Log-Target Transformation: Training is performed on log1p(total_cases) to squash outliers and improve focus on the median case frequency.

Biological Lags: Implementation of 4-week and 12-week shifts for humidity and temperature to reflect the mosquito breeding cycle.

Outlier Suppression: Weather variables are clipped at the 2nd and 98th percentiles to mitigate sensor errors.

Dampening Factor: A global multiplier of 0.85 is applied to final predictions to correct for the historical bias between training and test periods.

Repository Structure
Dengue_model.py: Main python script containing data cleaning, feature engineering, and the ensemble model.

submission.csv: The final predicted values formatted for competition entry.

Requirements
Python 3.x

Pandas

NumPy

Scikit-Learn