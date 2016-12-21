clear

addpath(genpath('/home/forough/dvp/synthetic_exp/UGM'))
addpath('/home/forough/dvp/synthetic_exp/toolbox')

dataset = 'FullBeach';
Data = loadData(dataset);
trainSamples = Data.trainSamples - 1;
traineSamples = Data.traineSamples;
traineCovariates = Data.traineCovariates;
trainCovariates = Data.trainCovariates;
[nTrain,nSamples] = size(trainSamples);
tmp = repmat(eye(nTrain),[1,1,nSamples]);
trainCovariates = [tmp,trainCovariates(:,2:end,:)];

clampedSetTrain = [];
isBox = 1;

load('DNRvertexParamsFullBeach/vertexOutCond1.mat')
clear testResults trainResults
load('edgeParamsFullBeach/eParamsCond.mat')

clampedDNRtrain = -1*ones(nTrain,nSamples);
clampedDNRtrain(clampedSetTrain,:) = trainSamples(clampedSetTrain,:);

[DNRtrainResults.VP,DNRtrainResults.CP,DNRtrainResults.CA,DNRtrainResults.EA,DNRtrainResults.EAcp,DNRtrainResults.EAca] =...
    condPredBerAllEdge(...
    DNRout.params,...
    trainCovariates,...
    trainSamples,...
    clampedDNRtrain,...
    traineSamples,...
    traineCovariates,...
    eParams,...
    isBox);

    DNRtrainResults.CP(25,:) = NaN(1,100);
    DNRtrainResults.CA(25,:) = NaN(1,100);
    
    DNRtrainResults.EAcp(25,:) = NaN(1,100);
    DNRtrainResults.EAca(25,:) = NaN(1,100);

for i=10;

load(sprintf('CLRFvertexParamsFullBeach/Results7/changedEdgeIter1_%dBox.mat',i))
% clear augTestData testResults trainResults
% ParamsEM = EMout.paramsEM;
% dParams = EMout.dParams;
% adjmatT = augTrainData.adjmatT;
% aug_covariates = augTrainData.aug_covariates;
% aug_depCovariates = augTrainData.aug_depCovariates;
% 
% clampedCLRFtrain = zeros(length(adjmatT),nSamples);
% clampedCLRFtrain(clampedSetTrain,:) = trainSamples(clampedSetTrain,:);

    trainResultsc.CP(25,:) = NaN(1,100);
    trainResultsc.CA(25,:) = NaN(1,100);
    
    trainResultsc.EAcp(25,:) = NaN(1,100);
    trainResultsc.EAca(25,:) = NaN(1,100);

% [CLRFtrainResults.VP,CLRFtrainResults.CP,CLRFtrainResults.CA,CLRFtrainResults.EA,CLRFtrainResults.EAcp,CLRFtrainResults.EAca] = ...
%     condPredAllEdge(...
%     ParamsEM,...
%     dParams,...
%     adjmatT,...
%     aug_covariates,...
%     aug_depCovariates,...
%     trainSamples+1,...
%     clampedCLRFtrain,...
%     traineSamples,...
%     traineCovariates,...
%     eParams,...
%     isBox);

%     testResultsc = testResults;
%     trainResultsc = trainResults;
%     clear EMout testResults trainResults augTestData augTrainData  

    figure;
    boxplot(trainResultsc.CP','boxstyle','outline','colors','r');
    hold on;boxplot(DNRtrainResults.CP','boxstyle','filled','colors','k');title('Vertex conditional presence')
    xlabel('Time points');ylabel('Sccuracy');%legend('CLRF','DNR');
    hLegend = legend(findall(gca,'Tag','Box'), {'CLRF','DNR'});
    hChildren = findall(get(hLegend,'Children'), 'Type','Line');
    % Set the horizontal lines to the right colors
    set(hChildren(4),'Color','r')
    set(hChildren(2),'Color','k')

    figure;
    boxplot(trainResultsc.EAcp','boxstyle','outline','colors','r');
    hold on;boxplot(DNRtrainResults.EAcp','boxstyle','filled','colors','k');title('Edge conditional presence')
    xlabel('Time points');ylabel('Accuracy');%legend('CLRF','DNR');
    hLegend = legend(findall(gca,'Tag','Box'), {'CLRF','DNR'});
    hChildren = findall(get(hLegend,'Children'), 'Type','Line');
    % Set the horizontal lines to the right colors
    set(hChildren(4),'Color','r')
    set(hChildren(2),'Color','k')
end