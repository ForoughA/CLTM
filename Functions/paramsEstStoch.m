function [Params,dParams,ll_approx, Ecll] = paramsEstStoch(trainSamples,adjmatT,aug_covariates,aug_depCovariates,optionsEM)

if ~isfield(optionsEM,'stepSize');
    optionsEM.stepSize = 1e-3;
end

optionsEM.maxIter = 10;
optionsEM.numStarting = 5;
Params_init = cell(optionsEM.numStarting,1);
dParams_int = cell(optionsEM.numStarting,1);
ecll_em_int = zeros(optionsEM.numStarting,1);
for i=1:optionsEM.numStarting
    [ Params_init{i}, dParams_int{i},ll_approx,Ecll] = EM_UGMbothStoch(adjmatT,trainSamples,aug_covariates,aug_depCovariates,optionsEM);
    ecll_em_int(i) = Ecll(end);
end
[foo,ind] = max(ecll_em_int);
optionsEM.initdParams = dParams_int{ind};
optionsEM.initParams = Params_init{ind};
clear dParams_int Params_init

optionsEM.maxIter = 300;
[Params, dParams, ll_approx,Ecll] = EM_UGMbothStoch(adjmatT,trainSamples,aug_covariates,aug_depCovariates,optionsEM);

