% 
clear
addpath('/home/forough/dvp/synthetic_exp/toolbox');
addpath(genpath('/home/forough/dvp/synthetic_exp/UGM'));

%Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataset = 'FullBeach';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ctr=1
    Data = loadData(dataset);
    load(sprintf('CLRFvertexParamsFullBeach/Results7/vertexOutCond%d',ctr))
    aug_covariates = augTrainData.aug_covariates;
    aug_depCovariates = augTrainData.aug_depCovariates;
    covariates = Data.trainCovariates;
    adjmatT = augTrainData.adjmatT;
    paramsEM = EMout.paramsEM;
    dParams = EMout.dParams;
    Samples = Data.trainSamples;

    [nObs,nSamples] = size(Samples);
    ndepCov = size(aug_depCovariates,3);
    nCov = size(covariates,2);
    nTot = length(adjmatT);
    nHidden = nTot - nObs;

    phStates = zeros(nObs, nHidden, nSamples);
    hCovariates = zeros(nObs, nCov+nHidden, nSamples);

    edgeStruct = UGM_makeEdgeStruct(adjmatT, 2);
    nEdges = edgeStruct.nEdges;
    edge_pairs = edgeStruct.edgeEnds;

    for t=1:nSamples

        nodePot(:,2) = exp(aug_covariates(:,:,t)*paramsEM);
        nodePot(:,1) = ones(nTot,1);
        edgePot = ones(2,2,edgeStruct.nEdges);
        for e=1:nEdges
            par = edge_pairs(e,1);
            child = edge_pairs(e,2);
            edgePot(2,2,e) = exp(reshape(aug_depCovariates(par,child,:,t),[1,ndepCov])*dParams);
        end 
        [nodeMar,edgeMar,logZ] = UGM_Infer_Tree(nodePot,edgePot,edgeStruct);

        for i=1:nObs
            parH = find(adjmatT(i,:)==1);
            parH = parH(parH>nObs);
            phStates(i,parH-nObs,t) = reshape(nodeMar(parH,2),[1,1,length(parH)]);
        end
        
    end

    hCovariates(:,1:nCov,:) = covariates;
    hCovariates(:,nCov+1:end,:) = phStates;

    save(sprintf('CLRFvertexParamsFullBeach/ResultsHcov/hCovariates%d',ctr),'hCovariates')
end
