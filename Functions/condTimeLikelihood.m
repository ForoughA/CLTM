%
clear
addpath(genpath('/home/forough/dvp/synthetic_exp/UGM/'))
addpath('/home/forough/dvp/synthetic_exp/toolbox/')

load('../Data/Samples.mat')
load('../learn_UGMboth/results2.mat')
% load('../learn_UGMboth/adjmatCL.mat')
ParamsEM = Params{1};
adjmatT = adjmatT{1};
Eclltot = Ecll{1};
dParams = dParams{1};
ll_approx = ll_approx{1};
% out_covariates = out_covariates{1];
out_covariates = out_covariates{1};
out_depCovariates = out_depCovariates{1};
load('../DNR/Params3.mat')
load('../DNR/LL3.mat')
load('../Data/covariates.mat')
load('../Data/depCovariates.mat')

% covariates(:,11,:) = [];
% Params(11) = [];

nTot = length(adjmatT);
% nTot = length(adjmatCL);
nObs = size(Samples,1);
nSamples = size(Samples,2);
% ndepCov = size(out_depCovariates,3);
ndepCov = size(out_depCovariates,3);
% nCov = size(out_covariates,2);
nCov = size(out_covariates,2);

edgeStruct = UGM_makeEdgeStruct(adjmatT,2);
edge_pairs = edgeStruct.edgeEnds;
nEdges = edgeStruct.nEdges;

P = nCov + ndepCov -1;
Pdnr = nCov - 1;

clampedSet = [];
% clampedSet = find(covariates(:,2,1)==1);
% clampedSet(randi(54,[1,30])) = [];
% clampedSet = [
%       1
%      4
%      7
%     12
%     13
%     15
%     16
%     17
%     18
%     28
%     29
%     30
%     37
%     39
%     41
%     43
%     44
%     45
%     48
%     50
%     58
%     59
%     60
%     63
%     66
%     67
%     69
%     70
%     74
%     95];
nSamples = size(Samples,2);
clampedCLRF = zeros(length(adjmatT),nSamples);
clampedCLRF(clampedSet,:) = Samples(clampedSet,:)+1;
clampedDNR = zeros(size(Samples));
clampedDNR(clampedSet,:) = Samples(clampedSet,:);

ll = zeros(nSamples,1);
Ecll = zeros(nSamples,1);
AIC = zeros(nSamples,1);
dnrLL = zeros(nSamples,1);
AICdnr = zeros(nSamples,1);
for t=1:nSamples
    
    %CLRF likelihood
    nodePot(:,2) = exp(out_covariates(:,:,t)*ParamsEM);
    nodePot(:,1) = ones(nTot,1);
    edgePot = ones(2,2,edgeStruct.nEdges);
    for e=1:nEdges
        par = edge_pairs(e,1);
        child = edge_pairs(e,2);
        edgePot(2,2,e) = exp(reshape(out_depCovariates(par,child,:,t),[1,ndepCov])*dParams);
    end
    
    %inference for computing the likelihood of the data under the
    %model. We could not use the conditional inference for this one.
    [node_marginal_LL,edge_marginal_LL,logZ_LL] = UGM_Infer_Conditional(nodePot,edgePot,edgeStruct,clampedCLRF(:,t),@UGM_Infer_Tree);
    for i=1:length(clampedSet)
       if  Samples(clampedSet(i),t) == 0
           nodePot(clampedSet(i),:) =  [1 0];
       else
           nodePot(clampedSet(i),:) =  [0 1];
       end
    end

    [node_marginal_LL2,edge_marginal_LL2,logZ_LL2] = UGM_Infer_Conditional(nodePot,edgePot,edgeStruct,clampedCLRF(:,t),@UGM_Infer_Tree);
    
    [ll(t),Ecll(t)] = computeCondLikelihood_UGM(nodePot,edgePot,node_marginal_LL,edge_marginal_LL,edgeStruct,Samples(:,t)+1,logZ_LL);
%     Ecll(t) = Ecll(t)/nTot;
%     ll(t) = ll(t)/nTot;
    
    AIC(t) = 2*P - 2*ll(t) + (2*P*(P+1))/(nSamples-P-1);
    
    %DNR likelihood
    p = logistic(covariates(:,:,t),Params);
    p(clampedSet) = Samples(clampedSet,t);
    dnrLL(t) = sum(Samples(:,t).*log(p+eps) + (1-Samples(:,t)).*log(1-p+eps));
%     dnrLL(t) = dnrLL(t);
%     dnrLL(t) = dnrLL(t);
    AICdnr(t) = 2*Pdnr - 2*dnrLL(t) + (2*Pdnr*(Pdnr+1))/(nSamples-Pdnr-1);
%     
%     dnrLL(t) = nll(covariates(1:nObs,:,t),Samples(:,t),Params);
%     dnrLL(t) = dnrLL(t) * nObs;
%     AICdnr(t) = 2*Pdnr + 2*dnrLL(t) + (2*Pdnr*(Pdnr+1))/(nSamples-Pdnr-1);
end

figure;
plot(AIC,'r');hold on;
plot(AICdnr,'b')

figure;
plot(ll,'r');hold on;
plot(Ecll,'m')
plot(dnrLL,'b')
% plot(ll,'m')