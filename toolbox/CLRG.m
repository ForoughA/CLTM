function [adjmatTree, edge_distance,adjmatCL, eachUpdate] = CLRG(stats, useDistances, numSamples,thrsh,options)

% CL-recursive-grouping algorithm to learn latent trees
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
% Myung Jin Choi, August 2010, MIT

if nargin < 4
    thrsh = 0.9;
end

if nargin < 5
    verbose = 0;
else
    if ~isfield(options,'verbose')
        verbose = 0;
    else
        verbose = 1;
    end
end

if useDistances
    distance = stats;
else
    samples = stats;
    numSamples = size(samples,2);
    prob_bij = computeBnStats(samples);
    distance = computeDistance(prob_bij);    
end

adjmatCL = ChowLiu(-distance);
%mi = computeMutualInformationBin(prob_bij);
%adjmatCL = ChowLiu(mi);

edge_distance = distance.*adjmatCL;
m = size(distance,1);

degree = sum(adjmatCL,2);
internal_nodes = find(degree > 1);
[foo,ind] = sort(degree(internal_nodes),'descend');
internal_nodes = internal_nodes(ind);
num_nodes = size(adjmatCL,1);
surrogate_nodes = [];
dist2surrogate = [];

if verbose
    eachUpdate = cell(length(internal_nodes)+2,3);
    eachUpdate{1,1} = logical(edge_distance);
    visit = ones(1, num_nodes);
end

for j=1:length(internal_nodes)
    
    % Select neighbors of an internal node
     i = internal_nodes(j);
     i_family = union(i,find(edge_distance(i,:)));
        
    % Replace hidden neighbors with their surrogate nodes
    isHidden = (i_family > m);
    hidden_nodes = i_family(isHidden);
    i_family(isHidden) = surrogate_nodes(hidden_nodes-m);
    
    % Apply recursive grouping to i and its neighbors
    [adjmatS, edge_distanceS] = RG(distance(i_family,i_family), 1, numSamples);
    num_new_nodes = size(adjmatS,1) - length(i_family);
    edge_distance = [edge_distance, sparse(size(edge_distance,1),num_new_nodes)];
    edge_distance = [edge_distance; sparse(num_new_nodes,size(edge_distance,2))];
    
    % Replace surrogate nodes back to their hidden nodes
    i_family(isHidden) = hidden_nodes;
    hidden_index = find(isHidden);
    for h=1:length(hidden_index)
        h_ind = hidden_index(h);
        h_neigh = logical(edge_distanceS(h_ind,:));
        h_node = hidden_nodes(h);
        new_ed = max(edge_distanceS(h_ind,h_neigh) - dist2surrogate(h_node-m),-log(0.95));
        edge_distanceS(h_ind,h_neigh) = new_ed;
        edge_distanceS(h_neigh,h_ind) = new_ed;
    end
    
    % Update the edge distance matrix
    new_node_ind = num_nodes+1:num_nodes+num_new_nodes;    
    i_new_family = [i_family, new_node_ind];
    if verbose
        remove = zeros(size(edge_distance));
        remove = sparse(remove);
        remove(i_new_family,i_new_family) = ...
            edge_distance(i_new_family,i_new_family);
        eachUpdate{j+1,3} = logical(remove);
    end
    edge_distance(i_new_family,i_new_family) = edge_distanceS;    

    surhid = [i, new_node_ind];
    subtree_dist = treeDistance(edge_distance(surhid,surhid));
    dist2surrogate = [dist2surrogate; subtree_dist(2:end,1)];
    surrogate_nodes = [surrogate_nodes; i*ones(num_new_nodes,1)];      
    num_nodes = num_nodes+num_new_nodes;
    
    visit = [visit, zeros(1,num_new_nodes)];
    
    if verbose
        update = zeros(size(edge_distance));
        update = sparse(update);
        update(i_new_family,i_new_family) = edge_distanceS;
        eachUpdate{j+1,1} = logical(update);
        addNodes = [];
        for tmpCtr = 1:length(i_new_family)
            if visit(i_new_family(tmpCtr))==0
                addNodes = [addNodes, i_new_family(tmpCtr)];
                visit(i_new_family(tmpCtr)) = 1;
            end
        end
        eachUpdate{j+1,2} = addNodes;
    end
end

edge_distance = contractWeakEdges(edge_distance,m,thrsh);
adjmatTree = logical(edge_distance);
eachUpdate{j+2,1} = adjmatTree;
