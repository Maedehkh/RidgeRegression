function [bestR2, bestIdx, cvResults, cvResultsMean] = optimizePCs(X, y, SCMP, lambda, CS)
% optimizePCsNestedLOOCV: Identifies the optimal number of principal components (PCs)
% using nested Leave-One-Out Cross-Validation (LOOCV) on training data.
%
% Inputs:
%   - X: A matrix of features (subjects x components).
%   - y: A vector of target scores for each subject.
%   - SCMP: Sorted list of principal components based on their importance.
%   - lambda: Regularization parameter for Ridge Regression.
%   - CS: Step size for selecting the number of PCs to evaluate.
%
% Outputs:
%   - bestR2: The highest average R² value obtained across all folds.
%   - bestIdx: Index corresponding to the optimal number of PCs.
%   - cvResults: Cell array storing R² values for each fold and number of PCs.
%   - cvResultsMean: Mean R² values across all folds for each number of PCs.
%
% Example usage:
%   [bestR2, bestIdx, cvResults, cvResultsMean] = optimizePCsNestedLOOCV(X, y, SCMP, lambda, CS);

    % Initialize storage for R² results
    R2cur = []; 
    cvResults = cell(size(X, 1), ceil(length(SCMP) / CS)); % Cell array for R² values
    cvResultsMSE = cell(size(X, 1), ceil(length(SCMP) / CS)); % Cell array for MSE values

    % Outer loop: Leave-One-Out Cross-Validation (LOOCV) on test subjects
    for Pi = 1:size(X, 1)
        fprintf('Processing subject %d of %d...\n', Pi, size(X, 1));
        
        % Exclude the current test subject from the training set
        APm = setdiff(1:size(X, 1), Pi); 
        Xi = X(APm, :); % Training set features
        Yi = y(APm);    % Training set scores

        m = 0; % Counter for the number of PCs evaluated
        
        % Inner loop: Evaluate subsets of sorted PCs
        for ND = 1:CS:length(SCMP)
            m = m + 1;
            
            % Select the top `ND` PCs from the sorted list
            XiPC = Xi(:, SCMP(1:ND)); 

            % Nested LOOCV within the training set
            cv_y_true = zeros(length(Yi), 1); % True scores for validation
            cv_y_pred = zeros(length(Yi), 1); % Predicted scores for validation
            
            parfor j = 1:length(Yi) % Parallel loop for efficiency
                % Exclude one patient from the training set
                X_loo = XiPC;
                X_loo(j, :) = []; % Exclude test patient
                y_loo = Yi;
                y_loo(j) = [];    % Exclude test patient
                
                % Perform Ridge Regression on the remaining training set
                b = ridge(y_loo, X_loo, lambda, 0);

                % Predict the score for the left-out patient
                y_pred = b(1) + XiPC(j, :) * b(2:end);
                cv_y_true(j) = Yi(j); % True score
                cv_y_pred(j) = y_pred; % Predicted score
            end

            % Compute R² for the current fold
            cv_r2 = 1 - sum((cv_y_true - cv_y_pred).^2) / sum((cv_y_true - mean(cv_y_true)).^2);
            cvResults{Pi, m} = cv_r2; % Store R² for this fold and number of PCs
            cvResultsMSE{Pi, m} = immse(cv_y_true, cv_y_pred); % Store MSE
        end

        % Aggregate R² values for the current test subject
        R2cur = [R2cur; cell2mat(cvResults(Pi, :))];  
    end

    % Compute mean R² values across all folds for each number of PCs
    cvResultsMean = mean(R2cur, 1);
    
    % Identify the optimal number of PCs based on the maximum mean R²
    [bestR2, bestIdx] = max(cvResultsMean);

    % Display the results
    fprintf('Optimal number of PCs: %d (R² = %.4f)\n', bestIdx, bestR2);
end
