function [node_potential,edge_potential]=marToPotBin(prob_bij,tree_msg_order)

% Changes marginal probabilities on a tree to node and edge potentials of a
% directed tree.
%
% PARAMETERS:
%       prob_bij(2*i-1:2*i,2*j-1:2*j) = P(bi,bj)
%       tree_msg_order = Edge ordering that goes from leaves to root and
%       back to the leaves
%
% OUTPUTS:
%       node_potential(i,:) = node potential of node i
%       edge_potential(2*i-1:2*i,2*j-1:2*j) = edge potential of edge (i,j)
%
% Myung Jin Choi, MIT, October 2009

N = size(prob_bij,1)/2;
node_potential = ones(N,2);
edge_potential = sparse(2*N,2*N);
root = tree_msg_order(N,1);
for node=1:N
    node_potential(node,:) = diag(prob_bij(2*node-1:2*node,2*node-1:2*node));
end

for n=size(tree_msg_order,1)/2+1:size(tree_msg_order,1)
    i = tree_msg_order(n,1); % parent
    j = tree_msg_order(n,2); % child
    p_i = diag(prob_bij(2*i-1:2*i,2*i-1:2*i));
    p_j = diag(prob_bij(2*j-1:2*j,2*j-1:2*j));
    p_i(p_i==0)=1;
    p_j(p_j==0)=1;
%     p_jli = prob_bij(2*i-1:2*i,2*j-1:2*j)./repmat(p_i,1,2); 
    p_jli = prob_bij(2*i-1:2*i,2*j-1:2*j)./[p_i,p_j];
    p_jli = p_jli / sum(p_jli(:));
    edge_potential(2*i-1:2*i,2*j-1:2*j) = p_jli;
end

edge_potential = edge_potential + edge_potential';