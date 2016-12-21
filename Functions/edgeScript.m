clear

job = batch('edgeLearnCond','profile','local','matlabpool',11);

wait(job)

load(job)

% exit
