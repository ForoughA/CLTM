clear

%% CLRF
load('DNRvertexParamsFullTwitter/vertexOutCond1.mat')
% load('CLRFvertexParamsFullTwitter/Results4/eParamsIter7Run1_10.mat')
% load('CLRFvertexParamsFullTwitter/Results4/ehCovariates1.mat')
load('/home/forough/dvp/Twitter/DataWeek/eCovariates_week.mat')
for ctr = 1:30
load(sprintf('edgeParamsFullTwitter/edgeOutCond%d.mat',ctr))


clear testResults trainResults eLL

clampedSetTrain = [];
clampedSetTest = [];
dataset = 'FullTwitter';
isBox = 0;

Data = loadData(dataset);

trainSamples = Data.trainSamples-1;
testSamples = Data.testSamples-1;
traineSamples = Data.traineSamples;
testeSamples = Data.testeSamples;
trainCovariates = Data.trainCovariates;
testCovariates = Data.testCovariates;
traineCovariates = eCovariates;
testeCovariates = eCovariates;

Params = DNRout.params;

[nTrain,nSamples] = size(trainSamples);

% [nTrain,nSamples] = size(trainSamples);
clampedDNRtrain = -1*ones(nTrain,nSamples);
clampedDNRtrain(clampedSetTrain,:) = trainSamples(clampedSetTrain,:);

nTest = size(testSamples,1);
clampedDNRtest = -1*ones(nTest,nSamples);
clampedDNRtest(clampedSetTest,:) = testSamples(clampedSetTest,:);

    tmp = repmat(eye(nTrain),[1,1,nSamples]);
    trainCovariates = [tmp,trainCovariates(:,2:end,:)];
    testCovariates = trainCovariates;

[trainResults.VP,trainResults.CP,trainResults.CA,trainResults.EA,trainResults.EAcp,trainResults.EAca] = condPredBerAllEdge(Params,trainCovariates,trainSamples,clampedDNRtrain,traineSamples,traineCovariates,eParams);
% [testResults.VP,testResults.CP,testResults.CA,testResults.EA,testResults.EAcp,testResults.EAca] = condPredBerAllEdge(Params,testCovariates,testSamples,clampedDNRtest,testeSamples,testeCovariates,eParams);

% Co-occurence
[~,~,~,~,co_ocTrain] = DNRco_occur(Params,trainCovariates,trainSamples);
% [~,~,~,~,co_ocTest] = DNRco_occur(Params,testCovariates,testSamples);

nTrain = size(trainSamples,1);
trainResults.mean_CoOc = zeros(1,nSamples);
for t=1:nSamples
    for i=1:nTrain
        for j=1:i-1
            trainResults.mean_CoOc(t) = trainResults.mean_CoOc(t) + co_ocTrain(i,j,t);
        end
    end
end
trainResults.mean_CoOc = trainResults.mean_CoOc/(nTrain*(nTrain-1)/2);

% nTest = size(testSamples,1);
% testResults.mean_CoOc = zeros(1,nSamples);
% for t=1:nSamples
%     for i=1:nTest
%         for j=1:i-1
%             testResults.mean_CoOc(t) = testResults.mean_CoOc(t) + co_ocTest(i,j,t);
%         end
%     end
% end
% testResults.mean_CoOc = testResults.mean_CoOc/(nTest*(nTest-1)/2);

testResults = [];
DNRtestResults = testResults;
DNRtrainResults = trainResults;
clear testResults trainResults
save(sprintf('DNRvertexParamsFullTwitter/changedEdgeIter1_%d.mat',ctr+30),'DNRtestResults','DNRtrainResults')
%
end