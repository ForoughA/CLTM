clear
addpath(genpath('/home/forough/dvp/synthetic_exp/UGM'))
addpath('/home/forough/dvp/synthetic_exp/toolbox')

%% CLRF
load('CLRFvertexParamsFullTwitter/Results4/vertexOutCond5.mat')
% load('CLRFvertexParamsFullTwitter/Results4/eParamsIter7Run1_10.mat')
load('CLRFvertexParamsFullTwitter/Results4/ehCovariates5.mat')
% load('CLRFvertexParamsFullTwitter/Results6/edgeOutCond1.mat')
for ctr=11
load(sprintf('CLRFvertexParamsFullTwitter/Results4/edgeOutCond%d.mat',ctr))
% load('edgeParamsFullTwitterTest/edgeOutCond20.mat')
% load('/home/forough/dvp/Twitter/DataWeek/eCovariates_week.mat')
eCovariates = ehCovariates;

clear testResults trainResults eLL

clampedSetTrain = [];
clampedSetTest = [];
dataset = 'FullTwitter';
isBox = 1;

Data = loadData(dataset);

trainSamples = Data.trainSamples;
testSamples = Data.testSamples;
traineSamples = Data.traineSamples;
testeSamples = Data.testeSamples;
traineCovariates = eCovariates;
testeCovariates = eCovariates;

ParamsEM = EMout.paramsEM;
dParams = EMout.dParams;
adjmatTtest = augTestData.adjmatT;
aug_covariatesTest = augTestData.aug_covariates;
aug_depCovariatesTest = augTestData.aug_depCovariates;
adjmatT = augTrainData.adjmatT;
aug_covariates = augTrainData.aug_covariates;
aug_depCovariates = augTrainData.aug_depCovariates;

nSamples = size(trainSamples,2);
clampedCLRFtrain = zeros(length(adjmatT),nSamples);
clampedCLRFtrain(clampedSetTrain,:) = trainSamples(clampedSetTrain,:);

clampedCLRFtest = zeros(length(adjmatTtest),nSamples);
clampedCLRFtest(clampedSetTest,:) = testSamples(clampedSetTest,:);

[trainResults.VP,trainResults.CP,trainResults.CA,trainResults.EA,trainResults.EAcp,trainResults.EAca] = condPredAllEdge(ParamsEM,dParams,adjmatT,aug_covariates,aug_depCovariates,trainSamples,clampedCLRFtrain,traineSamples,traineCovariates,eParams,isBox);
% [testResults.VP,testResults.CP,testResults.CA,testResults.EA,testResults.EAcp,testResults.EAca] = condPredAllEdge(ParamsEM,dParams,adjmatTtest,aug_covariatesTest,aug_depCovariatesTest,testSamples,clampedCLRFtest,testeSamples,testeCovariates,eParams,isBox);

% Co-occurence
[~,~,~,~,co_ocTrain] = CLRFco_occur(ParamsEM,dParams,adjmatT,aug_covariates,aug_depCovariates,trainSamples);
% [~,~,~,~,co_ocTest] = CLRFco_occur(ParamsEM,dParams,adjmatTtest,aug_covariatesTest,aug_depCovariatesTest,testSamples);

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
testResultsc = testResults;
trainResultsc = trainResults;
clear testResults trainResults
save(sprintf('CLRFvertexParamsFullTwitter/Results4/changedEdgeIter%d_5Box.mat',ctr),'testResultsc','trainResultsc')
% 

end


