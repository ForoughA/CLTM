function [ll,Ecll] = computeCondLikelihood(node_potential,edge_potential,node_marginal,edge_marginal,tree_msg_order,samples,logZ)
%   computing conditional likelihood
llnode = 0; llEdge = 0;
llhnode = 0; llhvEdge = 0; llhhEdge = 0;
nObs = size(samples,1);
for n = 1:size(tree_msg_order,1)/2
    if n<=nObs
        llnode = llnode + (samples(n)-1)*log(node_potential(n,samples(n))+eps);
    else
        llhnode = llhnode + log(node_potential(n,:)+eps) * [0 1]';
    end
    
    i = tree_msg_order(n,1);
    j = tree_msg_order(n,2);
    
    epot = log(edge_potential(2*i-1:2*i,2*j-1:2*j)+eps);
    emar = edge_marginal(2*i-1:2*i,2*j-1:2*j);
    
    if i<=nObs && j<=nObs
        llEdge = llEdge + (samples(i)-1)*(samples(j)-1)*epot(samples(i),samples(j));
    elseif i<=nObs && j>nObs
        %The only time the second term is non-zeros is when we choose
        %epot(2,2)
        llhvEdge = llhvEdge + (samples(i)-1)*(node_marginal(j,2)*epot(2,2)*1);
    elseif i>nObs && j<=nObs
        llhvEdge = llhvEdge + (samples(j)-1)*(node_marginal(i,2)*epot(2,2)*1);
    elseif i>nObs && j>nObs
        %E_{H|V,X}(\phi_{i,j}H_iH_j)
        llhhEdge = llhhEdge + epot(2,2)*emar(2,2);
    end
end
ll = llnode + llEdge - logZ; 
Ecll = ll + llhnode + llhvEdge + llhhEdge;

