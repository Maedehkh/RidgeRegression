clc
clear all
close all

% Specify the data folder for analysis
Folder = '.\Data\MotorL'; % Change folder for different scores: MotorL, MotorR, Exec, or Speed
RR = fullfile(Folder, '*.mat'); % Get the path to all .mat files in the folder
RMat = dir(RR); % List all .mat files in the specified folder

% Loop through each .mat file in the folder
for R = 1:length(RMat)
    
    %%%  start %%%%%% Prediction model optimization, two-step training strategy
    %%%% Step 1
    % Load the input file
    infile = fullfile(RMat(R).folder, RMat(R).name);
    load(infile); % Load data from the .mat file
    Score = Score_zscore; % Functional scores (z-scores) from the data
    y = Score; % Assign scores to the target variable
    
    % Compute cumulative explained variance for components
    ExVar = cumsum(Explained1);
    Tmp =(find(ExVar >= 99)); % Find the number of components explaining 99% variance
    NComp = Tmp(2);
    % Define parameters for the analysis
    CS = 5; % Step size for component selection
    lambda = 0.01; % Ridge regression penalty parameter
    nperms = 10000; % Number of permutations for statistical testing
    
    % Select components explaining up to 99% variance
    NewExplained = Explained1(1:NComp);
    X = Score1(:, 1:NComp); % Component matrix for analysis
    
    % Compute correlations between components and target scores
    corrs = corr(X, y);
    [corrs_all, sorted_idx_all] = sort(abs(corrs), 'descend'); % Sort components by correlation strength
    
    % Rank components across training folds
    rankedPCs = rankPCsAcrossFolds(X, y);
    
    % Aggregate ranked components across folds
    SCMP = sortComponentsByFrequency(rankedPCs, NComp);
    
    % Compare sorted components with cross-fold rankings
    K = [sorted_idx_all SCMP]'; % Compare sorted and ranked components
    
     %%%% Step 2
   % Optimize the number of components based on cross-validation R²
    [bestR2, bestIdx, cvResults, cvResultsMean] = optimizePCs(X, y, SCMP, lambda, CS);
    
    % Determine the optimal number of components
    ND = 1:CS:length(SCMP);
    NComponent = length(1:ND(bestIdx)); % Optimal number of components
    Variance = sum(NewExplained(SCMP(1:ND(bestIdx)))); % Variance explained by selected components

    %%%  end %%%%%% Prediction model optimization, two-step training strategy
 
    %%% start %%%%%% Prediction model testing using optimal parameters
    % Initialize arrays for cross-validation results
    cvytrue = zeros(length(y), 1);
    cvypred = zeros(length(y), 1);
    
    % Perform leave-one-out cross-validation
    for Pi = 1:size(X, 1)
        APm = setdiff(1:size(X, 1), Pi); % Leave one patient out
        Xi = X(APm, :); % Training data
        Yi = y(APm); % Training scores
        
        % Select optimal components for the training set
        X_loo = Xi(:, SCMP(1:ND(bestIdx)));
        y_loo = Yi;
        
        % Ridge regression on the training set
        b = ridge(y_loo, X_loo, lambda, 0);
        
        % Predict the left-out patient’s score
        y_pred = b(1) + X(Pi, SCMP(1:ND(bestIdx))) * b(2:end);
        cvytrue(Pi) = y(Pi); % True score
        cvypred(Pi) = y_pred; % Predicted score
    end
    
    % Calculate R² and mean squared error (MSE)
    R2 = 1 - sum((cvytrue - cvypred).^2) / sum((cvytrue - mean(cvytrue)).^2);
    MSE = immse(cvytrue, cvypred);
    
    % Permutation test for statistical significance
    X_Selected = X(:, SCMP(1:ND(bestIdx))); % Selected components
    [p_val] = permutation_test(X_Selected, y, R2, nperms, lambda);
    
    % Save outputs to a specified folder
    [~, folderName] = fileparts(Folder);
    OutFolderPath = fullfile('.\Output', folderName); % Output folder path
    disp(['Saving the results for: ', folderName, '_', RMat(R).name]);
    
    % Create the output folder if it doesn't exist
    if ~exist(OutFolderPath, 'dir')
        mkdir(OutFolderPath);
    end
    
   %%% end %%%%%% Prediction model testing using optimal parameters

    % Save results to a .mat file
    FN = fullfile(OutFolderPath, ['Out_', RMat(R).name]);
    save(FN, 'R2', 'MSE', 'cvytrue', 'cvypred', 'NComponent', 'Variance', 'p_val');
end