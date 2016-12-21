function [ll,Ecll] = computeCondLikelihood_UGM(node_potential,edge_potential,node_marginal,edge_marginal,edgeStruct,samples,logZ)
%   computing conditional likelihood
llnode = 0; llEdge = 0;
llhnode = 0; llhvEdge = 0; llhhEdge = 0;
nObs = size(samples,1);
edge_pairs = edgeStruct.edgeEnds;

for n = 1:length(edgeStruct.nStates)
    if n<=nObs
        llnode = llnode + log(node_potential(n,samples(n))+eps);
    else
        llhnode = llhnode + node_marginal(n,:)*log(node_potential(n,:)+eps)';
    end
 end

for e=1:edgeStruct.nEdges
    
    i = edge_pairs(e,1);
    j = edge_pairs(e,2);
    
    log_epot = log(edge_potential(:,:,e)+eps);
    emar = edge_marginal(:,:,e);
    
    if i<=nObs && j<=nObs
        llEdge = llEdge + log_epot(samples(i),samples(j));
    elseif i<=nObs && j>nObs
        %The only time the second term is non-zeros is when we choose
        %epot(2,2)
        llhvEdge = llhvEdge + (node_marginal(j,:)*log_epot(samples(i),:)');
    elseif i>nObs && j<=nObs
        llhvEdge = llhvEdge + (node_marginal(i,:)*log_epot(:,samples(j)));
    elseif i>nObs && j>nObs
        %E_{H|V,X}(\phi_{i,j}H_iH_j)
        llhhEdge = llhhEdge + emar(:)'*log_epot(:);
    end
end
ll = llnode + llEdge - logZ; 
Ecll = ll + llhnode + llhvEdge + llhhEdge;

