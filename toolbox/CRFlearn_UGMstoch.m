function [edgePot,Params,ll] = CRFlearn_UGMstoch(adjmat,samples,covariates)

nSamples = size(samples,2);
nCov = size(covariates,2) - 1;
nObs = size(samples,1);

nStates = 2;
edgeStruct = UGM_makeEdgeStruct(adjmat,nStates);
nEdges = edgeStruct.nEdges;
edge_pairs = edgeStruct.edgeEnds;

done = 0;
iter = 1;
stepsize = 1e-2;
maxIter = 300;
epsilon = 1e-5;
ll = zeros(maxIter,1);

nodePot = zeros(nTot,nStates);
edgePot = ones(nStates,nStates,nEdges);%should be initialized randomly
edgePot(2,2,:) = exprnd(1,[1,1,nEdges]);
Params = randn(nCov+1,1);%should be initialized randomly

while ~done
    step = stepsize/iter;
%     LL(iter) = ;
    for t=1:nSamples
        nodePot(:,2) = exp((covariates(:,:,t)) * Params);%/exp(1);
        nodePot(:,1) = ones(nObs,1);
        [node_marginals,edge_marginals,logZ] = UGM_Infer_Tree(nodePot,edgePot,edgeStruct);
        
        empirical = (samples(:,t)-1)' * (covariates(:,:,t));
        for ki = 1:nCov+1
            Params(ki) = Params(ki) + step*(empirical(ki) - (covariates(:,ki,t))' * (node_marginals*[0 1]'));
        end
        for e=1:nEdges
            u = edge_pairs(e,1);
            v = edge_pairs(e,2);
            emar = edge_marginals(:,:,e);
            log_ePot = log(edgePot(2,2,e));
            log_ePot = log_ePot + stp * ((samples(u,t)-1)*(samples(v,t)-1) - emar(2,2));
            edgePot(2,2,e) = exp(log_ePot);
        end
    end
    done = iter>maxIter ;%|| (iter>1 && abs(mean(LL(iter,:))-mean(LL(iter-1,:)))<epsilon) ;
    fprintf('iteration number: %d \n',iter);
    iter = iter + 1;
end