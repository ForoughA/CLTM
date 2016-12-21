function [jointProb, covar, varnce] = computeCondCorrTest(samples, covariates)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the conditional correlation matrix and returns it
% in matrix dist which will be used by computeCondDist2D
%
% Inputs: samples -> is a (nVars x dim x nSamples) matrix (continuous
%         Gaussian)
%         covatiates -> is a (nVars x nCov x nSamples) matrix (binary 1/2)
%
% Outputs: covar -> a (2*nVar x 2*nVar x 4 x ncov) conditional covariance
%          matrix. each of the 4 plates of the covariance matrix indicates
%          the distribution conditioned on 00, 01, 10, and 11 states of 
%          the covariate pairs. The diagonal 2x2 blocks are the variance
%          matrices and are only conditioned on the 00 and 11 states as
%          expected
%          jointProb -> the (2*nVars x 2*nVars x nCov) joint probability
%          distribution of the covariate pairs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nVars = size(samples, 1);
nCov = size(covariates, 2);
nSamples = size(samples, 2);

covar = zeros(nVars*(nVars-1)/2, 4, nCov);%conditional covariance matrx
varnce = zeros(nVars, 2, nCov);
jointProb = zeros(2*nVars, 2*nVars, nCov);

% forming pairwise multiplications and self multiplications matrix for
% correlation computation
pairMul = zeros(nVars*(nVars-1)/2, nSamples);
ctr = 1;
for i = 1:nVars
    for j = 1:nVars
        pairMul(ctr, :) = samples(i,:) .* samples(j,:);
        ctr = ctr + 1;
    end
end
selfMul = zeros(nVars, nSamples);
for i=1:nVars
    selfMul(i,:) = samples(i,:) .* samples(i,:);
end

L = length(pairMul);
parfor k = 1:nCov
    j = 0;
    i = 1;
    for PMctr = 1:L
%     for i = 1:nVars
%         for j = 1:nVars

            j = j + 1;
            if j>=i
                i = i + 1;
                j = 1;
            end
            
            ind = cell(4,1);
            p = zeros(4,1);
            covariate_pairs = ...
                reshape(2*covariates(i,k,:) + covariates(j,k,:),....
                [nSamples,1]);
            for ctr = 1:4
                ind{ctr} = find(covariate_pairs==ctr+2);
                p(ctr) = length(ind{ctr});
            end
            
            jointProb(PMctr,k) = 2;
            
            
%             if mod(2,i)==1 && i~=floor(j/2)
%                 jointProb(i,j) = p(1);
%             elseif i==floor(i/2) && i~=floor(j/2)
%                 jointProb(i,j) = p(2);
%             end
            
%             jointProb([2*i-1,2*i],[2*j-1,2*j],k) = [p(1) p(2); p(3) p(4)];
            covar(PMctr,1,k) = sum(pairMul(PMctr,ind{1})) / p(1);
            
%             for ctr = 1:4
%                 covar(PMctr,ctr,k) = sum(pairMul(PMctr,ind{ctr})) / p(ctr);
%             end
%             PMctr = PMctr + 1;
%         end
%     end
    end
    jointProb(:,:,k) = jointProb(:,:,k) + jointProb(:,:,k)';
    for i = 1:nVars
        
        ind = cell(2,1);
        p = zeros(2,1);
        for ctr = 1:2
            ind{ctr} = find(covariates(i,k,:)==ctr);
            p(ctr) = length(ind{ctr}); % P(xi=0/1)
        end

%         jointProb([2*i-1,2*i],[2*i-1,2*i],k) = [p(1) 0; 0 p(2)];
%         
%         for ctr=1:2
%            varnce(i, ctr, k) = sum(selfMul(i, ind{ctr}))/p(ctr);
%         end
        
    end
end
jointProb = jointProb / nSamples;



% for k = 1:nCov
%     for i = 1:nVars
%         for j = 1:i-1
%             
%             ind = cell(4,1);
%             p = zeros(4,1);
%             
%             %conditioning on the covariates
%             covariate_pairs = 2*covariates(i,k,:) + covariates(j,k,:);
%             for ctr = 1:4
%                 ind{ctr} = find(covariate_pairs==ctr+2);
%                 p(ctr) = length(ind{ctr});
%             end
% 
%             %joint probability of the covariates
%             jointProb([2*(i-1)+1,2*(i-1)+2],[2*(j-1)+1,2*(j-1)+2],k) = ...
%                 [p(1) p(2); p(3) p(4)];
%             
%             for ctr = 1:4
%                 %elements of the covariance matrix conditioned on the
%                 %covariates:
%                 y11 = samples(i,1,ind{ctr}) .* samples(j,1,ind{ctr});
%                 y12 = samples(i,1,ind{ctr}) .* samples(j,2,ind{ctr});
%                 y21 = samples(i,2,ind{ctr}) .* samples(j,1,ind{ctr});
%                 y22 = samples(i,2,ind{ctr}) .* samples(j,2,ind{ctr});
%                 %unbiased sample covariance estimator:
%                 covar(2*(i-1)+1:2*(i-1)+2, 2*(j-1)+1:2*(j-1)+2, ctr, k)...
%                     = sum([y11, y12; y21, y22], 3) / p(ctr);
%             end
%         end
%     end
%     jointProb(:,:,k) = jointProb(:,:,k) + jointProb(:,:,k)';
%     for ctr = 1:4
%         covar(:,:,ctr,k) = covar(:,:,ctr,k) + covar(:,:,ctr,k)';
%     end
%     
%     for i=1:nVars
%         
%         ind = cell(2,1);
%         p = zeros(2,1);
%         for ctr = 1:2
%             ind{ctr} = find(covariates(i,k,:)==ctr);
%             p(ctr) = length(ind{ctr}); % P(xi=0/1)
%         end
% 
%         jointProb([2*(i-1)+1,2*(i-1)+2],[2*(i-1)+1,2*(i-1)+2],k) = ...
%             [p(1) 0; 0 p(2)];
%         
%         for ctr=1:2
%            y11 = samples(i,1,ind{ctr}) .* samples(i,1,ind{ctr});
%            y22 = samples(i,2,ind{ctr}) .* samples(i,2,ind{ctr});
%            covar(2*(i-1)+1:2*(i-1)+2, 2*(i-1)+1:2*(i-1)+2, ctr^2, k) = ...
%                 sum([y11, zeros(1,1,length(y11));...
%                 zeros(1,1,length(y11)), y22], 3) / ...
%                 (p(ctr)-1);
%         end
%     end
%     
% end
% jointProb = jointProb / nSamples;



