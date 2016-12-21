% 
clear
addpath('/home/forough/dvp/synthetic_exp/toolbox');
addpath(genpath('/home/forough/dvp/synthetic_exp/UGM'));

%Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataset = 'FullTwitter';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ctr=5
    Data = loadData(dataset);
    load(sprintf('CLRFvertexParamsFullTwitter/Results4/vertexOutCond%d',ctr))
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

    phStates = zeros(nObs, nObs, nHidden, nSamples);
    chStates = zeros(nObs, nObs, nHidden, nSamples);
    ehCovariates = zeros(nObs, nObs, neCov+2*nHidden, nSamples);

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
                phStates(i,j,parH-nObs,t) = reshape(nodeMar(parH,2),[1,1,length(parH)]);
                chStates(i,j,childH-nObs,t) = reshape(nodeMar(childH,2),[1,1,length(childH)]);
            end
        end
        for h=1:nHidden
            phStates(:,:,h,t) = phStates(:,:,h,t) + phStates(:,:,h,t)';
            chStates(:,:,h,t) = chStates(:,:,h,t) + chStates(:,:,h,t)';
        end
    end

    ehCovariates(:,:,1:neCov,:) = eCovariates;
    ehCovariates(:,:,neCov+1:neCov+nHidden,:) = phStates;
    ehCovariates(:,:,neCov+nHidden+1:end,:) = chStates;

    save(sprintf('CLRFvertexParamsFullTwitter/Results4/ehCovariates%d',ctr),'ehCovariates')
end
