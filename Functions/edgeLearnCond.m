% Learning edge states using logistic regression
clear

addpath(genpath('/home/forough/dvp/synthetic_exp/UGM'))
addpath('/home/forough/dvp/synthetic_exp/toolbox')

% load('/home/forough/dvp/Beach/DataSplit/splittedData2.mat')
% load('/home/forough/dvp/Beach/DataSplit/SplittedAdjData2.mat')

% Samples = trainSamples;
% eSamples = traineSamples;
% eCovariates = traineCovariates;

load('/home/forough/dvp/Twitter/DataWeek/adj_week.mat')
load('/home/forough/dvp/Twitter/DataWeek/vertices_week.mat')
load('/home/forough/dvp/Twitter/DataWeek/eCovariates_week.mat')

eSamples = adj;
Samples = vertices;

nCov = size(eCovariates,3)-1;
nObs = size(eSamples,1);
nSamples = size(eSamples,3);

eNum = 0;
for i=1:nObs;for j=1:i-1;eNum=eNum+1;end;end
totSamples = zeros(eNum,nSamples);
covariates = zeros(eNum,nCov+1,nSamples);
for t=1:nSamples
    ctr = 1;
    for i=1:nObs
        for j=1:i-1
            totSamples(ctr,t) = eSamples(i,j,t);
            covariates(ctr,:,t) = eCovariates(i,j,:,t);
            ctr = ctr + 1;
        end
    end
end
nEdges = size(totSamples,1);

currCov = [];
currSample = [];
neT = zeros(1,nSamples);%number of plausible edges at each time point
for t=1:nSamples
   subEdgeInd = find(Samples(:,t)==1);
   subAdj = eSamples(subEdgeInd,subEdgeInd,t);
   subEcov = eCovariates(subEdgeInd,subEdgeInd,:,t);
   currNode = length(subEdgeInd);
   ctr = 1;
   for i=1:currNode
       for j=1:i-1
           currSample(ctr,t) = subAdj(i,j);
           currCov(ctr,:,t) = reshape(subEcov(i,j,:),[1,nCov+1]);
           ctr = ctr + 1;
       end
   end
   neT(t) = ctr-1;
end

maximumIter = 30;
matlabpool local 12

parfor iteration = 1:maximumIter
SampleHat = zeros(size(currSample));
rng('shuffle')
Params = randn(nCov+1,1)';
done = 0;
iter = 1;
stepsize = 1e-3;
maxIter = 100;
epsilon = 0.5;
err = zeros(maxIter,nSamples);
LL = zeros(maxIter,1);
while ~done
    step = stepsize/(iter);
%     LL(iter) = nll(covariates,Samples,Params);
    grad = 0;
    tP = 0;
    for t=randperm(nSamples)
        tP = tP + 1;
%        LL(iter,tP) = nll(covariates,Samples,Params);
       SampleHat(1:neT(t),t) = round(logistic(currCov(1:neT(t),:,t),Params));
       err(iter,t) = mean(SampleHat(1:neT(t),t)~=currSample(1:neT(t),t));
       for ctr=1:neT(t)
           sigX = logistic(currCov(ctr,:,t),Params);
           grad = -currSample(ctr,t)*(1-sigX)*currCov(ctr,:,t) + (1-currSample(ctr,t))*sigX*currCov(ctr,:,t);
           Params = Params - step*grad;
       end
%        fprintf('iteration %d, sample %d',iter,t);
    end
    
    for t=1:nSamples
        LL(iter) = LL(iter) + nll(currCov(1:neT(t),:,t),currSample(1:neT(t),t),Params);
    end
%     LL(iter) = nll(covariates,totSamples,Params);
    done = iter>=maxIter || (iter>1 && abs(LL(iter,:)-LL(iter-1,:))<epsilon);
    if done==1
       err = err(1:iter,:);
       LL = LL(1:iter,:);
    end
    fprintf('iteration number: %d, LL: %f \n',iter,LL(iter));
    iter = iter + 1;
end

eParams = Params;
eLL = LL;

parsaveEdge(sprintf('edgeParamsFullTwitterTest/edgeOutCond%d.mat',iteration),eParams,eLL)


end
matlabpool close

% exit
