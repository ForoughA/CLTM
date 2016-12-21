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
    adjmatT = augTrainData.adjmatT;
    paramsEM = EMout.paramsEM;
    dParams = EMout.dParams;
    Samples = Data.trainSamples;
    eCovariates = Data.traineCovariates;

    [nObs,nSamples] = size(Samples);
    ndepCov = size(aug_depCovariates,3);
    neCov = size(eCovariates,3);
    nTot = length(adjmatT);
    nHidden = nTot - nObs;

    hStates = zeros(nObs, nObs, 2, nSamples);
    ehCovariates = zeros(nObs, nObs, neCov+2, nSamples);

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
            for j=1:i-1
                parH = find(adjmatT(i,:)==1);
                parH = parH(parH>nObs);
                childH = find(adjmatT(j,:)==1);
                childH = childH(childH>nObs);
                hStates(i,j,1,t) = mean(nodeMar(parH,2));
                hStates(i,j,2,t) = mean(nodeMar(childH,2));
            end
        end
        hStates(:,:,1,t) = phStates(:,:,1,t) + phStates(:,:,1,t)';
        hStates(:,:,2,t) = phStates(:,:,2,t) + phStates(:,:,2,t)';
    end

    ehCovariates(:,:,1:neCov,:) = eCovariates;
    ehCovariates(:,:,end-1:end,:) = hStates;

    save(sprintf('CLRFvertexParamsFullBeach/Results7/ehCovariates%d_2',ctr),'ehCovariates')
end