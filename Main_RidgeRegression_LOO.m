clc;
clear all
close all

Folder='.\Data\Exec';  % Change directory for different functional scores: MotorL or MotorR or Exec or Speed
RR=fullfile(Folder,'*.mat');
RMat=dir(RR);

for R=1:length(RMat)
    
    infile=strcat(RMat(R).folder,'\',RMat(R).name);
    load(infile)
    Score = Score_zscore; %Loading functional score (z-score)
    y = Score;
    ExVar = cumsum(Explained1);  %Explained variance for components
    a = find(ExVar >= 99);
    NComp = a(2);
    CS=5; %%% componenet step
    %lambda=logspace(-2,2,3);
    lambda = 0.01;
    nperms = 10000; % Number of permutations
    
    NewExplained = Explained1(1:NComp);
    X = Score1(:,1:NComp); % Use the components explaining 99% of the variance in data
    
    %%%%% compute xcorr between all components and all scores
    corrs = corr(X, y);
    [corrs_all, sorted_idx_all] = sort(abs(corrs), 'descend');
    
    %%%%% compute xcorr between all components for each training set and all scores (except for the test patients)
    rankedPCs = rankPCsAcrossFolds(X, y);
    
    %%% rank PCs across all training sets
    SCMP = sortComponentsByFrequency(rankedPCs, NComp);
    
    K=[sorted_idx_all SCMP]'; %%% compare the order of PCs, (sorted_idx_all) vs those ranked across training sets (SCMP)
    
    %%%% find the optimal number of sorted PCs across all training sets
    %%%% based on R2
    [bestR2, bestIdx, cvResults, cvResultsMean] = optimizePCs(X, y, SCMP, lambda, CS);
    
    ND=1:CS:length(SCMP);
    NComponent=length(1:ND(bestIdx)); %Number of the components
    Variance=sum(NewExplained(SCMP(1:ND(bestIdx)))); %Compute variance explained by selected components
    cvytrue = zeros(length(y), 1);
    cvypred= zeros(length(y), 1);
    %%%% perform leave-one-out cross validations across all patients using
    %%%% the optimal feature set, the same across all training sets
    for Pi=1:size(X,1)
        %         Pi
        APm = setdiff(1:size(X,1), Pi); %%% leave one patient out
        Xi=X(APm,:);
        Yi=y(APm);
        
        X_loo = Xi(:,SCMP(1:ND(bestIdx))); %%% select the optimal PC set
        y_loo = Yi;
        
        % ridge Regression on training set
        b = ridge(y_loo, X_loo, lambda,0);
        
        % Compute prediction on left-out patient
        y_pred=b(1)+ X(Pi,SCMP(1:ND(bestIdx))) * b(2:end);
        cvytrue(Pi) = y(Pi);
        cvypred(Pi) = y_pred;
    end
    
    R2 = 1 - sum((cvytrue - cvypred).^2) / sum((cvytrue - mean(cvytrue)).^2);
    MSE= immse(cvytrue,cvypred);
    % Permutation
    X_Selected = X(:, SCMP(1:ND(bestIdx))); % Select optimal components
    
    [p_val] = permutation_test(X_Selected,y,R2,nperms,lambda);
    %% Saving outputs
    [~, folderName] = fileparts(Folder);
    OutFolderPath = fullfile('.\Output', folderName);
    disp(['Saving the results for: ', folderName, '_', RMat(R).name]);
    % Check if the new folder exists
    if ~exist(OutFolderPath, 'dir')
        mkdir(OutFolderPath);
    end
    FN=strcat(OutFolderPath, '\Out_',RMat(R).name);
    save(FN,'R2','MSE','cvytrue','cvypred','NComponent','Variance','p_val');
     
end
