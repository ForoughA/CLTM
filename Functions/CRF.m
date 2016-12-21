function[CRFout,augTrainData,augTestData,trainResults,testResults] = CRF(Data,covType,depCovType,clampedSetTrain,clampedSetTest,eParams,options)
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
    options.diffBias = 0;
end

if ~isfield(options,'diffBias')
    options.diffBias = 0;
end

if ~isfield(options,'isBox') 
    options.isBox = 0;
end

if ~isfield(options,'isTest') 
    options.isTest = 1;
end

options.method = 'ChowLiu';

%% Structure Learning

[adjmatT] = learnStruct(trainSamples,trainCovariates,covType,trainDepCovariates,depCovType,options);
if options.isTest
    [adjmatTtest] = learnStruct(testSamples,testCovariates,covType,testDepCovariates,depCovType,options);
else
    adjmatTtest = [];
    aug_covariatesTest = [];
    aug_depCovariatesTest = [];
end

if options.diffBias
    nSamples = size(trainSamples,2);
    nTot = size(adjmatT,1);
    nTotTest = size(adjmatTtest);
    tmp = repmat(eye(nTot),[1,1,nSamples]);
    tmpTest = repmat(eye(nTotTest),[1,1,nSamples]);
    aug_covariates = [tmp,aug_covariates(:,2:end,:)];
    aug_covariatesTest = [tmpTest,aug_covariatesTest(:,2:end,:)];
end

augTrainData.adjmatT = adjmatT;
augTrainData.aug_covariates = aug_covariates;
augTrainData.aug_depCovariates = aug_depCovariates;

if options.isTest
    augTestData.adjmatT = adjmatTtest;
    augTestData.aug_covariates = aug_covariatesTest;
    augTestData.aug_depCovariates = aug_depCovariatesTest;
else
    augTestData = [];
end
