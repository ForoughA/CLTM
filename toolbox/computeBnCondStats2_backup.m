function [cov_prob_Xk,cond_prob_bij] = computeBnCondStats2_backup(samples,covariates)

%covariates should be a 1/2 binary matrix and are fixed in time (not any
%more:D)

%This code works only for the synthetic data for which all the nodes in a
%time instance have the same set of covariates. for the real case we must
%come up with something else.

% Nc = size(covariates,1);% # of nodes
Kn = 2^(size(covariates,2) );% # of covariate states
% Mc = size(covariates,3);% # of samples

Nc = size(samples,1);% # of nodes ==> N = Nc
Mc = size(samples,2);% # of samples ==> M = Mc
    
% covariates = covariates - 1;%converting the 1/2 binary matrix of covariates to 0/1 binary
covariateStates = zeros(Nc);%matrix containing the covariate states of each node sample
for n = 1:Nc
%     tmp = covariates (1:end,m);%discarding the augmented all-one column
    covariateStates(n) = bin2dec(num2str(covariates(n,:)));
end


cov_prob_Xk = zeros(Kn,1);%Estimates covariate states probability
cond_prob_bij = zeros(2*Nc, 2*Nc, Kn);
% ind = cell(Kn,1);
for k = 1:Kn
    row = find(covariateStates==k-1);%We are only searching the first row since
        %in the synthetic data we have the same covariate states for all
        %nodes in one time instance
%     ind{k} = [row,clmn];
    cov_prob_Xk(k) = length(row);%# of samples whose covariates are in state k
    
    for i=1:Nc
        for j=1:i-1
            if ~isempty(find(row==i)) && ~isempty(find(row==j))
                sample_pairs = 2*samples(i,:) + samples(j,:);

                p00 = length(find(sample_pairs==3)); % P(bi=0,bj=0)
                p01 = length(find(sample_pairs==4)); % P(bi=0,bj=1)
                p10 = length(find(sample_pairs==5)); % P(bi=1,bj=0)
                p11 = length(find(sample_pairs==6)); % P(bi=1,bj=1)

                cond_prob_bij([2*(i-1)+1,2*(i-1)+2],[2*(j-1)+1,2*(j-1)+2],k) = [p00 p01; p10 p11];
            end
        end
    end
    cond_prob_bij(:,:,k) = cond_prob_bij(:,:,k) + cond_prob_bij(:,:,k)';
    
    for i=1:Nc
        if ~isempty(find(row==i))
            p00 = length(find(samples(row,:)==1)); % P(bi=0)
            p11 = length(find(samples(row,:)==2)); % P(bi=1)
        
        cond_prob_bij([2*(i-1)+1,2*(i-1)+2],[2*(i-1)+1,2*(i-1)+2],k) = [p00 0; 0 p11];
        end
    end
    cond_prob_bij(:,:,k) = cond_prob_bij(:,:,k) / Mc; 
end
cov_prob_Xk = cov_prob_Xk / length(covariates);

