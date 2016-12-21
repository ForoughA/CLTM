function distance = computeCondCorr(samples, covariates)
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
% tic;
nVars = size(samples, 1);
nCov = size(covariates, 2);
nSamples = size(samples, 2);

% covar = zeros(nVars*(nVars-1)/2, 4, nCov);%conditional covariance matrx
varnce = zeros(nVars, 2);
% jointProb = zeros(2*nVars, 2*nVars, nCov);
distance = zeros(nVars, nVars, nCov);

% forming pairwise multiplications and self multiplications matrix for
% correlation computation
pairMul = zeros(nVars*(nVars-1)/2, nSamples);
ctr = 1;
tiny = 1e-10;
for i = 1:nVars
    for j = 1:i-1
        pairMul(ctr, :) = samples(i,:) .* samples(j,:);
        ctr = ctr + 1;
    end
end
selfMul = zeros(nVars, nSamples);
for i=1:nVars
    selfMul(i,:) = samples(i,:) .* samples(i,:);
end

for k = 1:nCov
    
    PMctr = 1;
    for i = 1:nVars
        for j = 1:i-1
            
            covariate_pairs = ...
                reshape(2*covariates(i,k,:) + covariates(j,k,:),....
                [nSamples,1]);
            
            ind00 = covariate_pairs==3;
            ind01 = covariate_pairs==4;
            ind10 = covariate_pairs==5;
            ind11 = covariate_pairs==6;
            
            p00 = sum(ind00);
            p01 = sum(ind01);
            p10 = sum(ind10);
            p11 = sum(ind11);

            joint = [p00 p01 p10 p11]/nSamples;
            
            %time averaging
            if p00 ~= 0
                covar1 = sum(pairMul(PMctr,ind00)) / p00;
                var11 = sum(selfMul(i,ind00)) / p00;
                var12 = sum(selfMul(j,ind00)) / p00;
            else
                covar1 = 0;
                var11 = 0;
                var12 = 0;
            end
            if p01 ~= 0
                covar2 = sum(pairMul(PMctr,ind01)) / p01;
                var21 = sum(selfMul(i,ind01)) / p01;
                var22 = sum(selfMul(j,ind01)) / p01;
            else
                covar2 = 0;
                var21 = 0;
                var22 = 0;
            end
            if p10 ~= 0
                covar3 = sum(pairMul(PMctr,ind10)) / p10;
                var31 = sum(selfMul(i,ind10)) / p10;
                var32 = sum(selfMul(j,ind10)) / p10;
            else
                covar3 = 0;
                var31 = 0;
                var32 = 0;
            end
            if p11 ~= 0
                covar4 = sum(pairMul(PMctr,ind11)) / p11;
                var41 = sum(selfMul(i,ind11)) / p11;
                var42 = sum(selfMul(j,ind11)) / p11;
            else
                covar4 = 0;
                var41 = 0;
                var42 = 0;
            end
            
            numer = [covar1;
                     covar2;
                     covar3;
                     covar4];
            denom = sqrt([var11*var12;
                          var21*var22;
                          var31*var32;
                          var41*var42] + tiny);
            % Average over the 4 states
%             if denom(1)~=0 && numer(1)~=0
                distance(i,j,k) = distance(i,j,k) +...
                    joint(1) * (-1*log(abs(numer(1)/denom(1))+tiny));
                if numer(1)/denom(1) > 1.01
                    error('correlation exceeds 1')
                end
%             end
%             if denom(2)~=0 && numer(2)~=0
                distance(i,j,k) = distance(i,j,k) +...
                    joint(2) * (-1*log(abs(numer(2)/denom(2))+tiny));
                if numer(2)/denom(2) > 1.01
                    error('correlation exceeds 1')
                end
%             end
%             if denom(3)~=0 && numer(3)~=0
                distance(i,j,k) = distance(i,j,k) +...
                    joint(3) * (-1*log(abs(numer(3)/denom(3))+tiny));
                if numer(3)/denom(3) > 1.01
                    error('correlation exceeds 1')
                end
%             end
%             if denom(4)~=0 && numer(4)~=0
                distance(i,j,k) = distance(i,j,k) +...
                    joint(4) * (-1*log(abs(numer(4)/denom(4))+tiny)); 
                if numer(4)/denom(4) > 1.01
                    error('correlation exceeds 1')
                end
                
            PMctr = PMctr + 1;
        end
    end
end
distance = sum(distance,3);
% mx = max(max(distance));
% % [indi, indj] = find(distance==0);
% % tmp = distance;
% % valid = indi>indj;
% % tmp(indi(valid),indj(valid)) = 10*mx;
% for i=1:nVars
%     for j=1:i-1
%         if distance(i,j) == 0
%             distance(i,j) = 10*mx;
%         end
%     end
% end
% % t = toc;
distance = distance + distance';