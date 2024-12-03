# Post-stroke functional defiecit prediction in stroke patients
This repository contains MATLAB code for a predictive modeling framework that employs Ridge regression to predict post-stroke functional deficits, including motor function, executive function, and processing speed. The workflow leverages advanced cross-validation strategies and statistical evaluation to ensure robust and interpretable results.

** Key Features:
Ridge Regression: Utilizes L2 regularization to enhance model generalization and prevent overfitting, particularly in the context of lesion-behavior mapping with small sample sizes.
Nested LOOCV Training Strategy: A two-step nested Leave-One-Out Cross-Validation (LOOCV) process optimizes the selection of Principal Components (PCs) to maximize predictive performance (R²).

** Statistical Significance Testing:
Functional scores are permuted 10,000 times, and Ridge regression is applied to each permutation. Statistical significance is assessed by comparing observed R² values to the distribution of permuted R² values (p-values < 0.05).
 
 
** Instructions for Use
1- Input Data:
Input data is located in the Data folder, organized by deficit type:
MotorL: Left motor deficit
MotorR: Right motor deficit
Exec: Executive function
Speed: Processing speed
%% Each .mat file contains PCA-transformed data (Score1) and z-scored functional scores (Score_zscore).

2- Running the Code:
Open Main_RidgeRegression_LOO.m.
Specify the desired deficit type (MotorL, MotorR, Exec, or Speed).
Run the script in MATLAB. The code processes all feature sets automatically.

3- Output:
Results are saved in the Output folder for each input dataset.
Example results for processing speed (Speed) are already included in the repository.
%% Each Out_*.mat includes:
Coefficients of determination (R² values)
Mean Squared Error (MSE)
Number of selected components (NComponent)
Total explained variance by selected components (Variance)
P-value of permutation test (p_val)  

