function [adjmatTree, edge_distance] = RG2(stats, useDistances, numSamples,covariates)

% Recursive grouping algorithm to learn latent trees
% PARAMETERS:
%       if useDistances==true:   
%           stats = information distance matrix of observed nodes
%       if useDistances==false:
%           stats = samples of binary observed variables
%
% OUTPUTS:
%       adjmatTree = an adjacency matrix of a tree including latent nodes
%       edge_distance = information distances on the edges of the tree
%
% Myung Jin Choi, Jan 2010, MIT

if useDistances
    distance = stats;
else
    samples = stats;
    numSamples = size(samples,2);
    [cov_prob_Xk,prob_bij] = computeBnCondStats2(samples,covariates);
    distance = computeCondDistance2(prob_bij,cov_prob_Xk);    
end


m = size(distance,1); % # observed nodes
edge_distance = sparse(m,m);
currDist = distance;
nodeSet = 1:m;
newNodeNum = m+1;

while(length(nodeSet) > 2)  % Continue grouping until one or two nodes remain
    [families, parents, avg_log_ratio] = queryFamiliesClustering(currDist,numSamples);
    num_next_nodes = length(families);
    nextNodeSet = zeros(num_next_nodes,1);
    adjNewOld = false(num_next_nodes, length(nodeSet));
    edge_distance_sum = zeros(num_next_nodes,1);
    
    % Expand the adjacency matrix to incorporate new nodes
    num_new_nodes = sum((parents == 0));    
    edge_distance = [edge_distance, sparse(size(edge_distance,1),num_new_nodes)];
    edge_distance = [edge_distance; sparse(num_new_nodes,size(edge_distance,2))];  
    
    for f=1:num_next_nodes
        fml = families{f};
        prt = parents(f);
        child_nodes = nodeSet(fml);
        if(prt > 0) % There exists a parent among families
            p_node = nodeSet(prt);
            adjNewOld(f,prt) = true;
            edge_distance(p_node,child_nodes) = currDist(prt,fml);            
        else % Introduce a hidden node
            p_node = newNodeNum;            
            newNodeNum = newNodeNum + 1;
            adjNewOld(f,fml) = true;
            [edge_distance(p_node,child_nodes), edge_distance_sum(f)]...
                    = computeHiddenChild(currDist(fml,fml),avg_log_ratio(fml(1),fml(2)));
        end        
        nextNodeSet(f) = p_node;
    end
    currDist = computeNewDistance(currDist, adjNewOld, edge_distance_sum);
    nodeSet = nextNodeSet;
end


if(length(nodeSet) == 2)
    edge_distance(nodeSet(1),nodeSet(2)) = currDist(1,2);
end

edge_distance = edge_distance + edge_distance';
edge_distance = edge_distance - diag(diag(edge_distance));
edge_distance = contractWeakEdges2(edge_distance,m);
adjmatTree = logical(edge_distance);


