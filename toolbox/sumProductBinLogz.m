function [node_marginals,edge_marginals,logZ] = sumProductBinLogz(adjmat,node_potential,edge_potential,tree_msg_order)

% Sum-product on a tree with binary variables to compute node and edge marginals.
%
% PARAMETERS:
%       adjmat = adjacency matrix
%       node_potential(i,:) = node potential at node i (may include
%       evidence)
%       edge_potential(2*i-1:2*i,2*j-1:2*j) = edge potential at edge (i,j)
%       tree_msg_order = message order on a tree
%
% OUTPUTS:
%       node_marginals(i,:) = node marginal at node i
%       edge_marginals(2*i-1:2*i,2*j-1:2*j) = edge marginal at edge (i,j)\
%       logZ = log partition function of the tree
%
% modified from Myung Jin Choi, MIT, 2009 October

adjmat = logical(adjmat);
N = size(node_potential,1);
msg = ones(N,2*N);  % msg(i,2*j-1:2*j) = message from i to j
in_msg_prod = ones(N,2);  % in_msg_prod(i,:) = product of incoming messages except for the current target node
edge_marginals = sparse(2*N,2*N);

for n=1:size(tree_msg_order,1)
    i = tree_msg_order(n,1);
    j = tree_msg_order(n,2);
    
    neighbors = adjmat(i,:);
    neighbors(j) = 0;
    in_msg_prod(i,:) = prod(msg(neighbors,2*i-1:2*i),1);
    msg_ij = node_potential(i,:).*in_msg_prod(i,:);
    msg_ij = repmat(msg_ij',1,2).*edge_potential(2*i-1:2*i,2*j-1:2*j);
    %fprintf('%d, %d,%f\n',i,j,in_msg_prod(i,1))
    if(sum(msg_ij(:)) > 0)
        msg(i,2*j-1:2*j) = sum(msg_ij,1)/sum(msg_ij(:));
    end
    
    if (n>=size(tree_msg_order,1)/2) % downward pass
        emar = in_msg_prod(i,:)'*in_msg_prod(j,:);  % in_msg_prod(j) has product of messages from its children
        emar = emar.*edge_potential(2*i-1:2*i,2*j-1:2*j);
        emar = emar.*(node_potential(i,:)'*node_potential(j,:));
        emar = emar / sum(emar(:));
        edge_marginals(2*i-1:2*i,2*j-1:2*j) = emar;
        edge_marginals(2*j-1:2*j,2*i-1:2*i) = emar';
    end
end        

%degree = sum(adjmat,1);
node_marginals = full(sum(edge_marginals,2));
node_marginals = reshape(node_marginals',2,N)';
%in_msg_prod = prod(msg(adjmat),1);
%fnode_marginals = reshape(in_msg_prod,2,N).*node_potential;
node_marginals = node_marginals ./ repmat(sum(node_marginals,2),1,2);

%This part of the code is inspired by the UGM toolbox:
if nargout > 2
   % Compute Bethe free energy 
   % (Z could also be computed as normalizing constant for any node in the tree
   %    if unnormalized messages are used)
   Energy1 = 0; Energy2 = 0; Entropy1 = 0; Entropy2 = 0;
   node_marginals = node_marginals+eps;
   edge_marginals = edge_marginals+eps;
   for n=1:size(tree_msg_order,1)/2
       i = tree_msg_order(n,1);
       j = tree_msg_order(n,2);

       neighbors = adjmat(i,:);
       % Node Entropy (can get divide by zero if beliefs at 0)
       Entropy1 = Entropy1 + ((length(neighbors)-1))*sum(node_marginals(n,:).*log(node_marginals(n,:)+eps));
       
       % Node Energy
       Energy1 = Energy1 - sum(node_marginals(n,:).*log(node_potential(n,:)+eps));
       
       % Pairwise Entropy (can get divide by zero if beliefs at 0)
       eb = edge_marginals(2*i-1:2*i,2*j-1:2*j);
       Entropy2 = Entropy2 - sum(eb(:).*log(eb(:)+eps));

       % Pairwise Energy
       ep = edge_potential(2*i-1:2*i,2*j-1:2*j);
       Energy2 = Energy2 - sum(eb(:).*log(ep(:)+eps));
   end
   F = (Energy1+Energy2) - (Entropy1+Entropy2);
   logZ = -F;
end