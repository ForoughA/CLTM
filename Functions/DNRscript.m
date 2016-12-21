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
dataset = 'FullBeach';
diffBias = 1;% add node specific bias
clampedSetTrain = [];
clampedSetTest = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(genpath('/home/forough/dvp/synthetic_exp/UGM'))
addpath('/home/forough/dvp/synthetic_exp/toolbox')

Data = loadData(dataset);
Data.trainSamples = Data.trainSamples - 1;
Data.testSamples = Data.testSamples - 1;

%loading the edge model
load('edgeParamsFull/eParamsCond.mat')
% load('/home/forough/dvp/Beach/DNR/eParamsCond.mat')
% eParams = x;
% eLL = y;

% matlabpool local 12
for iteration = 2 %1:maxIter

    [DNRout,trainResults,testResults] = ...
        DNR(...
        Data,...
        clampedSetTrain,...
        clampedSetTest,...
        eParams,...
        diffBias);
       
    DNRparsave(sprintf('DNRvertexParamsFullBeach/vertexOutCond%d.mat',iteration),DNRout,trainResults,testResults);
    
end

% matlabpool close

% exit

% figure;plot(testResults.EAcp);hold on;plot(trainResults.EAcp,'r')
%  figure;plot(testResults.VP,'r');hold on;plot(testResults.CP,'b');plot(testResults.CA,'k');plot(testResults.EA,'g');plot(testResults.EAcp,'m');plot(testResults.EAca,'y')
