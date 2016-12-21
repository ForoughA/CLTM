function [Params,dParams,ll] = CRFlearn_UGMboth(adjmat,samples,covariates,dCovariates)

nSamples = size(samples,2);
nCov = size(covariates,2) - 1;
nObs = size(samples,1);

nStates = 2;
edgeStruct = UGM_makeEdgeStruct(adjmat,nStates);
nEdges = edgeStruct.nEdges;
edge_pairs = edgeStruct.edgeEnds;
ndepCov = size(dCovariates,3);

stepsize = 1e-3;
maxIter = 150;
ll = zeros(maxIter,1);

% load('/home/forough/dvp/synthetic_exp/learnFull20_UGMem_regCLRG/Data/edgePot')
% figure;
% subplot(2,1,1);plot(reshape(edgePot(2,2,:),[81,1]),'r');hold on
% clear edgePot

nodePot = zeros(nObs,nStates);
Params = randn(nCov+1,1);%should be initialized randomly
dParams = randn(ndepCov,1);
% load('/home/forough/dvp/synthetic_exp/learnFull20_UGMem_regCLRG/Data/Ck')
% Params = Ck;
% load('/home/forough/dvp/synthetic_exp/learnFull20_UGMem_regCLRG/Data/edgePot')
% subplot(2,1,2);plot(Ck,'r');hold on

for iter = 1:maxIter
    step = stepsize/iter;
    Gk = zeros(1,nCov+1);
    Ge = zeros(1,ndepCov);
%     LL(iter) = ;
    for t=1:nSamples
        %inference:
        nodePot(:,2) = exp((covariates(:,:,t)) * Params);%/exp(1);
        nodePot(:,1) = ones(nObs,1);
        edgePot = ones(2,2,edgeStruct.nEdges);
        for e=1:nEdges
            par = edge_pairs(e,1);
            child = edge_pairs(e,2);
            edgePot(2,2,e) = exp(reshape(dCovariates(par,child,:,t),[1,ndepCov])*dParams);
        end
        [node_marginals,edge_marginals,logZ] = UGM_Infer_Tree(nodePot,edgePot,edgeStruct);
        
        %gradient update:
        Nempirical = (samples(:,t)-1)' * (covariates(:,:,t));
        Nmu = (node_marginals*[0 1]')'*covariates(:,:,t);
        Eempirical = zeros(1,ndepCov);
        Emu = zeros(1,ndepCov);
        for e=1:nEdges
            par = edge_pairs(e,1);
            child = edge_pairs(e,2);
            Eempirical = Eempirical + (samples(par,t)-1)*(samples(child,t)-1)*reshape(dCovariates(par,child,:,t),[1,ndepCov]);
            Emu = Emu + edge_marginals(2,2,e) * reshape(dCovariates(par,child,:,t),[1,ndepCov]);
        end
        Gk = Gk + Nempirical - Nmu;
        Ge = Ge + Eempirical - Emu;
        
        %likelihood computation
        llt = computeCondLikelihood_UGM(nodePot,edgePot,node_marginals,edge_marginals,edgeStruct,samples(:,t),logZ);
        ll(iter) = ll(iter) + llt;
%         Ecll(iter) = Ecll(iter) + Ecllt;
    end
    
    %Parameter update:
    Params = Params + step * Gk';
    dParams = dParams + step * Ge';
    if iter>1 && abs(ll(iter)-ll(iter-1))<1
        ll = ll(1:iter);
%         Ecll = Ecll(iter);
        break
    end
%     subplot(2,1,2);plot(Params);hold on;
%     subplot(2,1,1);plot(reshape(edgePot(2,2,:),[81,1])); hold on
%     drawnow
    fprintf('iteration number: %d, ll: %f \n',iter,ll(iter));
end
