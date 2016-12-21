%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data : Educational
% Algorithm : CLRF
% This script performs training and testing using CLRF on 
% the educational dataset.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

% Parameters : %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxIter = 10;
options.diffBias = 0;
% options.thrsh = 0.9;
dataset = 'EducationalSmall';
% clampedSetTrain = [];
% clampedSetTest = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(genpath('/home/forough/dvp/synthetic_exp/UGM'))
addpath('/home/forough/dvp/synthetic_exp/toolbox')
addpath('/home/forough/dvp/CleanedUpCodes')
path = 'Educational/structOut2/';

rng('shuffle')

[Data,covType,depCovType] = loadData(dataset);
options.nodeLabels = Data.nodeLabels;
options.method = 'CLRG';
% options.thrsh = 0.3;
options.root = 1;

% eParams = x;
% eLL = y;

% matlabpool local 10
% iteration = 1;
% parfor iteration = 1:maxIter

%     [EMout,augTrainData,augTestData,trainResults,testResults] = ...
%         CLRF_Gaussian(...
%         Data,...
%         covType,...
%         depCovType,...
%         clampedSetTrain,...
%         clampedSetTest,...
%         eParams,...
%         options);

% for iter = 1:9
    
    options.thrsh = 9/10;

    [adjmatT,aug_covariates,aug_depCovariates,optionsEM] = ...
    learnStructGaussian(...
    Data.trainSamples,...
    Data.trainCovariates,...
    covType,...
    [],...
    depCovType,...
    options);
    distance = optionsEM.edge_distance;
    save('Educational/structOut2/Struct9.mat','adjmatT','optionsEM')
%%
    [r, c, d] = find(adjmatT);
    adjDump = [r, c, d];

    dlmwrite([path,'struct',num2str(iter),'.txt'],adjDump,'delimiter','\t');
% end

labels = Data.nodeLabels;
fileID = fopen('Educational/structOut2/labels2.txt','w');
formatSpec = '%d \t %s\n';
[nrows,ncols] = size(labels);
for row = 1:nrows
    fprintf(fileID,formatSpec,[row,labels{row,:}]);
end
fclose(fileID);

%     structParsave(sprintf('Educational/structOutCond%d.mat',iteration),adjmatT,aug_covariates,aug_depCovariates,optionsEM)
    
% end
% matlabpool close
% 
% save('workspace.mat')

% exit

% figure;plot(testResults.EAcp);hold on;plot(trainResults.EAcp,'r')
%  figure;plot(testResults.VP,'r');hold on;plot(testResults.CP,'b');plot(testResults.CA,'k');plot(testResults.EA,'g');plot(testResults.EAcp,'m');plot(testResults.EAca,'y')
