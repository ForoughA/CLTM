%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data : Student
% Algorithm : CLRF
% This script performs training and testing using CLRF on 
% the beach dataset.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

% Parameters : %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxIter = 10;
options.method = 'CLRG1';
options.diffBias = 1;
options.thrsh = 1e-5;
options.isBox = 0;
options.isTest = 0;
dataset = 'StudentSmall';
clampedSetTrain = [];
clampedSetTest =[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(genpath('UGM'))
addpath('toolbox')
addpath('Functions')

rng('shuffle')

[Data,covType,depCovType] = loadData(dataset);
bCovType = Data.bCovType;

%loading the edge model
% load('../edgeParamsFullTwitter/eParamsCond.mat')
% eParams = x;
% eLL = y;

% matlabpool local 10
for iteration = 1:maxIter

    [EMout,augTrainData,augTestData,trainResults,testResults] = ...
        CLRFedu(...
        Data,...
        bCovType,...
        depCovType,...
        clampedSetTrain,...
        clampedSetTest,...
        options);
    
    CLRFparsave(sprintf('vertexOutCond%d.mat',iteration),EMout,augTrainData,augTestData,trainResults,testResults)
    
end
% matlabpool close

% save('workspace.mat')
% 
% exit
