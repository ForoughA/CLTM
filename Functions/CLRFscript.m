%
clear

addpath(genpath('/home/forough/dvp/synthetic_exp/UGM'))
addpath('/home/forough/dvp/synthetic_exp/toolbox')

job = batch('CLRFscriptBeach','profile','local','matlabpool',11);

wait(job)

load(job)
% save('CLRFvertexParamsBeach/workspace.mat')

% exit
