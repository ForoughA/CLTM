function [cov_prob_xij,cond_prob_bij] = computeBnCondStats4(samples,covariates)

%This function computes the conditional pairwise probability tables of the
%vertices given the covariates for the beach dataset. The covariates are
%contained in a 95x3x31 array and we have 3 covariates for each of the 95
%vertices in the network observed for 31 days with one missing day replaced
%with the mean of the other days. The first covariate indicates the
%reqularity of each node the second one indicates the temperature and the
%third covariate is the previous state of that vertex

Nc = size(covariates,1);% # of nodes
Kn = size(covariates,2);% # of covariates per user
Mc = size(covariates,3);% # of samples

N = size(samples,1);% # of nodes ==> N = Nc
M = size(samples,2);% # of samples

cov_prob_xij = sparse(4*N,4*N);
cond_prob_bij = zeros(2*Nc,2*Nc,2^(2*Kn));
regularity = covariates(:,1,1);%regularity of the users is fixed in time
covariateStates = zeros(Nc,Mc);%matrix containing the covariate states of each node sample
    %containing temperature and previous state
    
 for k=1:Kn
    [mn,mx] = range(covariates(:,k,:));
    for i=1:Nc
        for j=1:i-1
            
            
        end
    end
 end