function [edgePot,Params,ll] = CRFlearn_UGM(adjmat,samples,covariates)

nSamples = size(samples,2);
nCov = size(covariates,2) - 1;
nObs = size(samples,1);

nStates = 2;
edgeStruct = UGM_makeEdgeStruct(adjmat,nStates);
nEdges = edgeStruct.nEdges;
edge_pairs = edgeStruct.edgeEnds;

stepsize = 1e-3;
maxIter = 150;
ll = zeros(maxIter,1);

% load('/home/forough/dvp/synthetic_exp/learnFull20_UGMem_regCLRG/Data/edgePot')
% figure;
% subplot(2,1,1);plot(reshape(edgePot(2,2,:),[81,1]),'r');hold on
% clear edgePot

nodePot = zeros(nObs,nStates);
edgePot = ones(nStates,nStates,nEdges);%should be initialized randomly
edgePot(2,2,:) = exprnd(5,[1,1,nEdges]);
Params = randn(nCov+1,1);%should be initialized randomly
% load('/home/forough/dvp/synthetic_exp/learnFull20_UGMem_regCLRG/Data/Ck')
% Params = Ck;
% load('/home/forough/dvp/synthetic_exp/learnFull20_UGMem_regCLRG/Data/edgePot')
% subplot(2,1,2);plot(Ck,'r');hold on

for iter = 1:maxIter
    step = stepsize/iter;
    Gk = 0;
    Ge = zeros(nEdges,1);
%     LL(iter) = ;
    for t=1:nSamples
        %inference:
        nodePot(:,2) = exp((covariates(:,:,t)) * Params);%/exp(1);
        nodePot(:,1) = ones(nObs,1);
        [node_marginals,edge_marginals,logZ] = UGM_Infer_Tree(nodePot,edgePot,edgeStruct);
        
        %gradient update:
        empirical = (samples(:,t)-1)' * (covariates(:,:,t));
        mu = (node_marginals*[0 1]')'*covariates(:,:,t);
        Gk = Gk + empirical - mu;
        for e=1:nEdges
            u = edge_pairs(e,1);
            v = edge_pairs(e,2);
            emar = edge_marginals(:,:,e);
            Ge(e) = Ge(e) + (samples(u,t)-1)*(samples(v,t)-1) - emar(2,2);
        end
        
        %likelihood computation
        llt = computeCondLikelihood_UGM(nodePot,edgePot,node_marginals,edge_marginals,edgeStruct,samples(:,t),logZ);
        ll(iter) = ll(iter) + llt;
%         Ecll(iter) = Ecll(iter) + Ecllt;
    end
    
    %Parameter update:
    Params = Params + step * Gk';
    log_ePot = reshape(log(edgePot(2,2,:)),[nEdges,1]);
    log_ePot = log_ePot + step * Ge;
    edgePot(2,2,:) = reshape(exp(log_ePot),[1,1,nEdges]);
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

