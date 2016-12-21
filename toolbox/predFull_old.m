function [VP,CP,CA] = predFull(Ck,edgePot,adj,cov,samples)
% function [VP,CP,CA,EA] = predFull(Ck,Dk,edgePot,adj,cov,eCov,samples,eSamples)
% Ck = abs(Ck);
% Ck = Ck/max(Ck);
edgeStruct = UGM_makeEdgeStruct(adj,2);
nTot = length(adj);
nObs = size(samples,1);
nSamples = size(samples,2);
% nEcov = length(Dk) - 1;

% eNum = 0;
% for i=1:nObs;for j=1:i-1;eNum=eNum+1;end;end
% edgeSamples = zeros(eNum,nSamples);
% edgeCov = zeros(eNum,neCov+1,nSamples);
% for t=1:nSamples
%     ctr = 1;
%     for i=1:nObs
%         for j=1:i-1
%             edgeSamples(ctr,t) = eSamples(i,j,t);
%             edgeCov(ctr,:,t) = eCov(i,j,:,t);
%             ctr = ctr + 1;
%         end
%     end
% end
% nEdges = size(edgeSamples,1);

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

%Vertex prediction
VP = zeros(nSamples-1,1);
CP = zeros(nSamples-1,1);
CA = zeros(nSamples-1,1);
Prediction = zeros(nTot,nSamples);
for t=2:nSamples
    testNodePot = zeros(nTot,2);
    testNodePot(:,2) = exp(cov(:,:,t)*Ck);%/exp(1);
    testNodePot(:,1) = ones(nTot,1);
%     testNodePot(:,1) = mean(testNodePot(:,2)) * ones(nTot,1);
    testNodePot = testNodePot ./ repmat(sum(testNodePot,2),[1,2]);
%     testNodePot(:,1) = 1 - testNodePot(:,2);

    Prediction(:,t) = UGM_Decode_Tree(testNodePot, testEdgePot, edgeStruct);
    VP(t-1) = mean((Prediction(1:nObs,t)==samples(:,t)));
    ind0 = find(samples(1:nObs,t)==1);
    ind1 = find(samples(1:nObs,t)==2);
    if ~isempty(ind0)
        CA(t-1) = sum(Prediction(ind0,t)==1)/length(ind0);
    end
    if ~isempty(ind1)
        CP(t-1) = sum(Prediction(ind1,t)==2)/length(ind1);
    end
    cov(:,end,t+1) = Prediction(:,t)-1;
%     edgeCov(:,5,t) = repmat(log(sum(Prediction(:,t)-1)+eps),[nEdges,1]);
%     eCov(:,:,5,t) = repmat(log(sum(Prediction(:,t)-1)+eps),[nObs,nObs]);
end

%Edge prediction
% EA = zeros(nSamples-1,1);
% % ePrediction = zeros(
% for t=2:nSamples
%    subEdgeInd = find(Prediction(1:nObs,t)==2);%vertices predicted as present
%    subEdgeIndTrue = find(samples(1:nObs,t)==2);%present vertices
%    ctr = 1;
%    ePrediction = 2*ones(nObs,nObs);%2 indicates edge inapplicability
%    eCov(:,:,6,t+1) = zeros(nObs,nObs);
%    for i=1:length(subEdgeInd)
%        for j=1:i-1
%            ePrediction(subEdgeInd(i),subEdgeInd(j)) = round(logistic(reshape(eCov(subEdgeInd(i),subEdgeInd(j),:,t),[1,nEcov+1]),Dk));
%            eCov(subEdgeInd(i),subEdgeInd(j),6,t+1) = round(logistic(reshape(eCov(subEdgeInd(i),subEdgeInd(j),:,t),[1,nEcov+1]),Dk));
%            ctr = ctr + 1;
%        end
%    end
%    eCov(:,:,6,t+1) = eCov(:,:,6,t+1) + eCov(:,:,6,t+1);
%    tmp = [];
%    ctr = 1;
%    for i=1:length(subEdgeIndTrue)
%        for j=1:i-1
%            %conditioning on the non-NA edges
%            tmp(ctr) = ePrediction(subEdgeIndTrue(i),subEdgeIndTrue(j))==eSamples(subEdgeIndTrue(i),subEdgeIndTrue(j),t);
%            ctr = ctr + 1;
%        end
%    end
%    EA(t-1) = sum(tmp)/(ctr-1);
% end

% error = mean(err);