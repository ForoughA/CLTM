function [DNRout,trainResults,testResults] = DNR(Data,clampedSetTrain,clampedSetTest,eParams,diffBias)
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
%         diffBias indicates whether we have node specific bias or not
%         (0/1)
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
traineSamples = Data.traineSamples;
testeSamples = Data.testeSamples;
traineCovariates = Data.traineCovariates;
testeCovariates = Data.testeCovariates;

if nargin<5
    diffBias = 0;
end

if diffBias
    [nTrain,nSamples] = size(trainSamples);
    tmp = repmat(eye(nTrain),[1,1,nSamples]);
    trainCovariates = [tmp,trainCovariates(:,2:end,:)];
    testCovariates = trainCovariates;
end

%% Parameter Estimation
[Params,LL] = paramsEstDNR(trainSamples,trainCovariates,5e-2,0.5,300);

DNRout.params = Params;
DNRout.LL = LL;
%% Testing
% VP/CP/CA/EA/EAca/EAcp
[nTrain,nSamples] = size(trainSamples);
clampedDNRtrain = -1*ones(nTrain,nSamples);
clampedDNRtrain(clampedSetTrain,:) = trainSamples(clampedSetTrain,:);

nTest = size(testSamples,1);
clampedDNRtest = -1*ones(nTest,nSamples);
clampedDNRtest(clampedSetTest,:) = testSamples(clampedSetTest,:);

[trainResults.VP,trainResults.CP,trainResults.CA,trainResults.EA,trainResults.EAcp,trainResults.EAca] = condPredBerAllEdge(Params,trainCovariates,trainSamples,clampedDNRtrain,traineSamples,traineCovariates,eParams);
[testResults.VP,testResults.CP,testResults.CA,testResults.EA,testResults.EAcp,testResults.EAca] = condPredBerAllEdge(Params,testCovariates,testSamples,clampedDNRtest,testeSamples,testeCovariates,eParams);

% Co-occurence
[~,~,~,~,co_ocTrain] = DNRco_occur(Params,trainCovariates,trainSamples);
[~,~,~,~,co_ocTest] = DNRco_occur(Params,testCovariates,testSamples);

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