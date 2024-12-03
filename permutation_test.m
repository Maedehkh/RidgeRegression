function [p_val] = permutation_test(X,y,R2,nperms,lambda)
% PERMUTATION_TEST Performs a permutation test to compute p-value for R².
%
% Inputs:
%   - X: predictor variables.
%   - y: True response variable.
%   - lambda: Regularization parameter for ridge regression.
%   - R2: Observed R² value of the original model.
%   - nperms: Number of permutations to perform.
%
% Outputs:
%   - p_val: p-value indicating the significance of the observed R².


% Initialize permutation R² array
perm_R2 = zeros(nperms, 1);

% Permutation test loop
for i = 1:nperms
    % Permute the response variable
    y_perm = y(randperm(length(y)));
    
    % Perform ridge regression on permuted response
    b_perm = ridge(y_perm, X, lambda, 0);
    y_pred_perm = b_perm(1) + X * b_perm(2:end);
    
    % Compute R² for permuted response
    perm_R2(i) = 1 - sum((y_perm - y_pred_perm).^2) / sum((y_perm - mean(y_perm)).^2);
end

% Compute p-value for permutation test
p_val = sum(perm_R2 >= R2) / nperms;

end
