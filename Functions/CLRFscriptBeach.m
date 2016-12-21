%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data : Beach
% Algorithm : CLRF
% This script performs training and testing using CLRF on 
% the beach dataset.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

% Parameters : %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxIter = 1;
options.method = 'CLRG4';
options.diffBias = 1;
% options.thrsh = 0;
dataset = 'FullBeach';
clampedSetTrain = [];
clampedSetTest = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(genpath('/home/forough/dvp/synthetic_exp/UGM'))
addpath('/home/forough/dvp/synthetic_exp/toolbox')

[Data,covType,depCovType] = loadData(dataset);

%loading the edge model
load('edgeParamsFullBeach/eParamsCond.mat')
% eParams = x;
% eLL = y;

% matlabpool local 12
% for iteration = 1:maxIter

    [EMout,augTrainData,augTestData,trainResults,testResults] = ...
        CLRF(...
        Data,...
        covType,...
        depCovType,...
        clampedSetTrain,...
        clampedSetTest,...
        eParams,...
        options);
    
%     CLRFparsave(sprintf('CLRFvertexParamsFullBeach/vertexOutCond%d.mat',iteration),EMout,augTrainData,augTestData,trainResults,testResults)
    
% end

% matlabpool close

% save('CLRFvertexParamsFullBeach/workspace.mat')

% exit

% figure;plot(testResults.EAcp);hold on;plot(trainResults.EAcp,'r')
%  figure;plot(testResults.VP,'r');hold on;plot(testResults.CP,'b');plot(testResults.CA,'k');plot(testResults.EA,'g');plot(testResults.EAcp,'m');plot(testResults.EAca,'y')
