function [Params,LL] = paramsEstDNR(trainSamples,covariates,stepsize,epsilon,maxIter)
% Learning vertex states using logistic regression

nCov = size(covariates,2);  
nSamples = size(trainSamples,2);
nObs = size(trainSamples,1);

Params = randn(nCov,1)';
done = 0;
iter = 1;
if nargin<3
    stepsize = 1e-2;
    maxIter = 300;
    epsilon = 1e-5;
elseif nargin<4
    maxIter = 300;
    epsilon = 1e-5;
elseif nargin<5
    maxIter = 300;
end
err = zeros(maxIter,nSamples);
LL = zeros(maxIter,1);
while ~done
    step = stepsize;%/sqrt(iter);
    tP = 0;
    for t=1:nSamples%randperm(nSamples)
        tP = tP + 1;
       SampleHat(:,t) = round(logistic(covariates(:,:,t),Params)-eps);
       err(iter,t) = mean(SampleHat(:,t)~=trainSamples(:,t));
       for ctr=1:nObs
           sigX = logistic(covariates(ctr,:,t),Params);
           grad = -trainSamples(ctr,t)*(1-sigX)*covariates(ctr,:,t) + (1-trainSamples(ctr,t))*sigX*covariates(ctr,:,t);
           Params = Params - step*grad;
       end
    end
    LL(iter) = nll(covariates,trainSamples,Params);
    done = iter>maxIter || (iter>1 && abs(mean(LL(iter,:))-mean(LL(iter-1,:)))<epsilon) ;
    if done==1
       err = err(1:iter,:);
       LL = LL(1:iter,:);
    end
    fprintf('iteration number: %d, LL: %f \n',iter,LL(iter));
    iter = iter + 1;
end