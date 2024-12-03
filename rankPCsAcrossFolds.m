function rankedPCs = rankPCsAcrossFolds(X, y)
% rankPCsAcrossFolds: Computes rankings of principal components (PCs) across training folds.
%
% Inputs:
%   - X: A matrix of features (subjects x components).
%   - y: A vector of scores corresponding to each subject.
%
% Output:
%   - rankedPCs: A matrix where each row contains the indices of PCs ranked
%                by their absolute correlation with y, for each leave-one-out fold.
%
% Example usage:
%   rankedPCs = rankPCsAcrossFolds(X, y);

    % Initialize matrix to store rankings of PCs across folds
    rankedPCs = [];

    % Loop through each subject for leave-one-out cross-validation
    for Pi = 1:size(X, 1)
        % Identify the indices of the training set (exclude the test subject)
        APm = setdiff(1:size(X, 1), Pi); 
        
        % Extract the training set features and scores
        Xi = X(APm, :); % Features for training set
        Yi = y(APm);    % Scores for training set
        
        % Compute correlations between each PC and the scores
        corrs = corr(Xi, Yi);
        
        % Sort PCs by the absolute value of correlations in descending order
        [~, sorted_idx] = sort(abs(corrs), 'descend');
        
        % Append the sorted indices to the results matrix
        rankedPCs = [rankedPCs; sorted_idx'];
    end
end