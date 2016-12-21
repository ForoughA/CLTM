function [VP,CP,CA] = pred(nodePot,edgePot,adj,cov,samples,Ck)
Ck = Ck/max(Ck);
edgeStruct = UGM_makeEdgeStruct(adj,2);
nTot = length(adj);
nObs = size(cov,1);
nSamples = size(samples,2);

if size(edgePot,3) < 2
    edgePot = full(edgePot);
    testEdgePot = zeros(2,2,edgeStruct.nEdges);
    for e = 1:edgeStruct.nEdges
        child = edgeStruct.edgeEnds(e,1);
        parent = edgeStruct.edgeEnds(e,2);
        testEdgePot(:,:,e) = edgePot(2*parent-1:2*parent,2*child-1:2*child);
    end
else
    testEdgePot = edgePot;
end

VP = zeros(nSamples-1,1);
CP = zeros(nSamples-1,1);
CA = zeros(nSamples-1,1);
for t=2:nSamples
    testNodePot = zeros(nTot,2);
    testNodePot(1:nObs,2) = exp(cov(:,:,t)*Ck);
    testNodePot(1:nObs,1) = mean(testNodePot(1:nObs,2)) * ones(nObs,1);
    testNodePot(1:nObs,:) = testNodePot(1:nObs,:) ./ repmat(sum(testNodePot(1:nObs,:),2),[1,2]);
    testNodePot(nObs+1:nTot,:) = nodePot(nObs+1:nTot,:);

    Prediction = UGM_Decode_Tree(testNodePot, testEdgePot, edgeStruct);
    VP(t-1) = mean((Prediction(1:nObs)==samples(:,t)));
    ind0 = find(samples(:,t)==1);
    ind1 = find(samples(:,t)==2);
    if ~isempty(ind0)
        CA(t-1) = sum(Prediction(ind0)==1)/length(ind0);
    end
    if ~isempty(ind1)
        CP(t-1) = sum(Prediction(ind1)==2)/length(ind1);
    end
    cov(:,end,t+1) = Prediction(1:nObs)-1;
end
% error = mean(err);
