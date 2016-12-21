function H = computeEntropy(nodePot,edgePot,adj)

nStates = size(nodePot,2);
edgeStruct = UGM_makeEdgeStruct(adj,nStates);
nNodes = size(nodePot,1);
nEdges = edgeStruct.nEdges;
[nodeMar,edgeMar] = UGM_Infer_Tree(nodePot,edgePot,edgeStruct);

entropy = 0;
for n=1:nNodes
   entropy = entropy - nodeMar(n,:) * log(nodeMar(n,:)');
end

energy = 0;
for e = 1:nEdges
    i = edgeStruct.edgeEnds(e,1);
    j = edgeStruct.edgeEnds(e,2);
    emar = edgeMar(:,:,e);
    prd = nodeMar(i,:)'*nodeMar(j,:);
    energy = energy + emar(:)' * log((emar(:)./prd(:)));
end

H = entropy - energy;

