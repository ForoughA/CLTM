function [adjmatT,aug_covariates,aug_depCovariates,optionsEM,eachIterTree] = ...
    learnStructGaussian(...
    samples,...
    covariates,...
    covType,...
    depCovariates,...
    depCovType,...
    options)

%covariateType is a vector with the same length as the number of
%covariates. It is used to indicate whether a specific covariate is node
%specific (1), time varying (2) or both (3)

nObs = size(samples, 1);
nSamples = size(samples, 2);
dim = size(samples, 2);
nCov = size(covariates, 2);
ndepCov = size(depCovariates, 3);

if ~isfield(options,'nodeLabels')
    options.nodeLabels = cell(1,nObs);
    for i=1:nObs
        options.nodeLabels{i} = i;
    end
end

if ~isfield(options,'thrsh')
    options.thrsh = 0.9;
end

if ~isfield(options,'method')
    options.method = 'CLRG';
end

if ~isfield(options,'root')
    options.root = 1;
end

if ~isfield(options,'verbose')
    options.verbose = 0;
end
%%
verbose = 1;
maxHidden = inf;
options.startingTree = 'MST';

tic;
distance = computeCondCorr(samples,covariates+1);
t=toc;

switch options.method
    case 'reg_CLRG'
        [adjmatT, optionsEM.edge_distance] =...
            regCLRG_Gaussian(distance, nSamples, verbose, maxHidden, options.thrsh);
    case 'CLRG'
        [adjmatT, optionsEM.edge_distance,~,eachIterTree] = CLRG(distance, 1, nSamples,options.thrsh,options);
end
aug_covariates = [];
aug_depCovariates = [];
% drawLatentTree(adjmatT, nObs, options.root, options.nodeLabels);

% %% interpreting the variables
% nTotal = length(adjmatT);
% nHidden = nTotal - nObs;
% groups = cell(nHidden,1);
% adj = full(adjmatT);
% for i = nObs+1:nTotal
%     groups{i-nObs} = options.nodeLabels(adj(i,1:nObs));
% end
% %%
% %Augmenting the covariates for the hidden nodes
% optionsEM.root = 1;
% aug_covariates = [covariates;zeros(nHidden,nCov,nSamples)];
% aug_depCovariates = zeros(nTotal,nTotal,ndepCov,nSamples);
% aug_depCovariates(1:nObs,1:nObs,:,:) = depCovariates;
% 
% ind1_Cov = find(covType==1);
% ind1_depCov = find(depCovType==1);
% ind2_Cov = find(covType==2);
% ind2_depCov = find(depCovType==2);
% ind3_Cov = find(covType==3);
% ind3_depCov = find(depCovType==3);
% 
% 
% %Time varying, node constant covariates:
% aug_covariates(nObs+1:nTotal,ind2_Cov,:) = repmat(aug_covariates(nObs,ind2_Cov,:),[nHidden,1,1]);
% aug_depCovariates(nObs+1:nTotal,1:nTotal,ind2_depCov,:) = repmat(aug_depCovariates(nObs,:,ind2_depCov,:),[nHidden,1,1,1]);
% aug_depCovariates(1:nObs,nObs+1:nTotal,ind2_depCov,:) = repmat(aug_depCovariates(1:nObs,nObs,ind2_depCov,:),[1,nHidden,1,1]);
% 
% for i = nObs+1:nTotal
%     %finding the neighbors of node i:
%     neighbor_ind = find(adjmatT(i,1:i)==1);
%     
%     %node specific covariates
%     aug_covariates(i,ind1_Cov,:) = repmat(round(mean(aug_covariates(neighbor_ind,ind1_Cov,1),1)),[1,1,nSamples]);
%     
%     %time varying and node specific covariates
%     for n=1:nSamples
%         aug_covariates(i,ind3_Cov,n) = round(mean(aug_covariates(neighbor_ind,ind3_Cov,n),1));
%     end
%     
% end
% 
% %node specific covariates
% for i=1:length(ind1_depCov)
%     aug_depCovariates(:,:,ind1_depCov(i),:) = repmat(aug_covariates(:,ind1_Cov(i),1)*aug_covariates(:,ind1_Cov(i),1)',[1,1,1,nSamples]);
% end
% 
% %time varying and node specific covariates
% for n=1:nSamples
%     for i=1:length(ind3_depCov)
%         aug_depCovariates(:,:,ind3_depCov(i),n) = aug_covariates(:,ind3_Cov(i),n)*aug_covariates(:,ind3_Cov(i),n)';
%     end
% end
