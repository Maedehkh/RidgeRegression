function SCMP = sortComponentsByFrequency(A, NComp)
% sortComponentsByFrequency: Identifies and sorts principal components (PCs)
% based on their frequency of high ranking across training folds.
%
% Inputs:
%   - A: A matrix of ranked PCs from `rankPCsAcrossFolds` function. 
%        Each row corresponds to a training fold, and each column contains the
%        ranking indices of the PCs for that fold.
%   - NComp: The total number of principal components.
%
% Output:
%   - SCMP: A vector of PC indices sorted by their frequency of high ranking.
%           PCs with higher frequency are ranked higher in the output.
%
% Example usage:
%   SCMP = sortComponentsByFrequency(A, NComp);

    % Initialize a matrix to store PC indices and their aggregate ranks
    AC = [];

    % Loop through each principal component
    for i = 1:NComp
        % Initialize a counter to sum the ranks for component `i`
        Ai = 0;

        % Loop through each fold (row in `A`)
        for j = 1:size(A, 1)
            % Find the rank of component `i` in fold `j`
            L = find(A(j, :) == i);

            % Aggregate ranks across all folds
            Ai = Ai + L; 
        end

        % Append the PC index and its total rank to the results matrix
        AC = [AC; i, Ai];
    end

    % Sort the components by their total rank (ascending)
    [~, sorted_indices] = sort(AC(:, 2));

    % Extract the sorted component indices
    SCMP = AC(sorted_indices, 1);
end