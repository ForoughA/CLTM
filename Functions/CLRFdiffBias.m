function [EMout,augTrainData,augTestData,trainResults,testResults] = CLRFdiffBias(Data,covType,depCovType,clampedSetTrain,clampedSetTest,eParams,options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inputs: Data is the structure containing the following
%         samples => binary training samples. The entries should be 1/2
%         covariates & depCovariates => node and edge covariates in the
%         vertex dependency model
%         covType and depCovType => array indicating whether the covariates
%         are node specific (1), time varying but node constant (2) or both
%         simultaneously (3).
%         clampedSet => array indicating the nodes that prediction is
%         conditioned on. If we want to condition on no nodes we can use an
%         empty matrix.
%         options => structure array giving the input options for structure
%         learning
%
%
% outputs:ParamsEM and sParams => node and edge parameters of the vertex
%         model (saved in EMout)
%         Ecll => expected complete data log likelihood (saved in EMout)
%         aug_covariates and aug_depCovariates are the augmented node and
%         edge covariates of the vertex model extended to include the
%         hidden variables
%         adjmatT => learned structure
%         testResults: structure containing the test results like vertex
%         and edge prediction accuracy as well as co-occurrence statistics.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load data :|
trainSamples = Data.trainSamples;
testSamples = Data.testSamples;
trainCovariates = Data.trainCovariates;
testCovariates = Data.testCovariates;
trainDepCovariates = Data.trainDepCovariates;
testDepCovariates = Data.testDepCovariates;
traineSamples = Data.traineSamples;
testeSamples = Data.testeSamples;
traineCovariates = Data.traineCovariates;
testeCovariates = Data.testeCovariates;

if nargin < 7
    options.method = 'regCLRG'; 
end

%% Structure Learning
[adjmatT,aug_covariates,aug_depCovariates,optionsEM] = learnStruct(trainSamples,trainCovariates,covType,trainDepCovariates,depCovType,options);
[adjmatTtest,aug_covariatesTest,aug_depCovariatesTest,~] = learnStruct(testSamples,testCovariates,covType,testDepCovariates,depCovType,options);

nSamples = size(trainSamples,2);
tmp = repmat(eye(length(adjmatT)),[1,1,nSamples]);
tmpTest = repmat(eye(length(adjmatTtest)),[1,1,nSamples]);
aug_covariates = [tmp,aug_covariates(:,2:end,:)];
aug_covariatesTest = [tmpTest,aug_covariatesTest(:,2:end,:)];

augTrainData.adjmatT = adjmatT;
augTrainData.aug_covariates = aug_covariates;
augTrainData.aug_depCovariates = aug_depCovariates;

augTestData.adjmatT = adjmatTtest;
augTestData.aug_covariates = aug_covariatesTest;
augTestData.aug_depCovariates = aug_depCovariatesTest;
%% Parameter Estimation
[ParamsEM, dParams, ~, Ecll] = paramsEst(trainSamples, adjmatT, aug_covariates, aug_depCovariates, optionsEM);

EMout.paramsEM = ParamsEM;
EMout.dParams = dParams;
EMout.Ecll = Ecll;
%% Testing
% VP/CP/CA/EA/EAca/EAcp
clampedCLRFtrain = zeros(length(adjmatT),nSamples);
clampedCLRFtrain(clampedSetTrain,:) = trainSamples(clampedSetTrain,:);

clampedCLRFtest = zeros(length(adjmatTtest),nSamples);
clampedCLRFtest(clampedSetTest,:) = testSamples(clampedSetTest,:);

[trainResults.VP,trainResults.CP,trainResults.CA,trainResults.EA,trainResults.EAcp,trainResults.EAca] = condPredAllEdge(ParamsEM,dParams,adjmatT,aug_covariates,aug_depCovariates,trainSamples,clampedCLRFtrain,traineSamples,traineCovariates,eParams);
[testResults.VP,testResults.CP,testResults.CA,testResults.EA,testResults.EAcp,testResults.EAca] = condPredAllEdge(ParamsEM,dParams,adjmatTtest,aug_covariatesTest,aug_depCovariatesTest,testSamples,clampedCLRFtest,testeSamples,testeCovariates,eParams);

% Co-occurence
[~,~,~,~,co_ocTrain] = CLRFco_occur(ParamsEM,dParams,adjmatT,aug_covariates,aug_depCovariates,trainSamples);
[~,~,~,~,co_ocTest] = CLRFco_occur(ParamsEM,dParams,adjmatTtest,aug_covariatesTest,aug_depCovariatesTest,testSamples);

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

nTest = size(testSamples,1);
testResults.mean_CoOc = zeros(1,nSamples);
for t=1:nSamples
    for i=1:nTest
        for j=1:i-1
            testResults.mean_CoOc(t) = testResults.mean_CoOc(t) + co_ocTest(i,j,t);
        end
    end
end
testResults.mean_CoOc = testResults.mean_CoOc/(nTest*(nTest-1)/2);
% 