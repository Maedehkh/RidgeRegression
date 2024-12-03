# RidgeRegression
This repository contains MATLAB code for a predictive modeling framework that employs Ridge regression to predict post-stroke functional deficits, including motor function, executive function, and processing speed. The workflow leverages advanced cross-validation strategies and statistical evaluation to ensure robust and interpretable results.

Key Features:
Ridge Regression: Utilizes L2 regularization to enhance model generalization and prevent overfitting, particularly in the context of lesion-behavior mapping with small sample sizes.
Nested LOOCV Training Strategy: A two-step nested Leave-One-Out Cross-Validation (LOOCV) process optimizes the selection of Principal Components (PCs) to maximize predictive performance (R²).

Performance metrics include:
Coefficients of determination (R² values)
Mean Squared Error (MSE)
Total explained variance

Statistical Significance Testing:
 Functional scores are permuted 10,000 times, and Ridge regression is applied to each permutation. Statistical significance is assessed by comparing observed R² values to the distribution of permuted R² values (p-values < 0.05).
