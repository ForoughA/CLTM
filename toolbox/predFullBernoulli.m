%DNR pred with sampling
function [VP,CP,CA,EA,EAcp,EAca] = predFullBernoulli(Ck,Dk,edgePot,adj,cov,eCov,samples,eSamples)

nCov = length(Ck) - 1;
samNum = 100;

edgeStruct = UGM_makeEdgeStruct(adj,2);
nTot = length(adj);
nObs = size(samples,1);
nSamples = size(samples,2);
nEcov = length(Dk) - 1;

VP = zeros(nSamples-1,1);%vertex prediction accuracy
CP = zeros(nSamples-1,1);%conditional vertex presence
CA = zeros(nSamples-1,1);%conditional vertex absence
EA = zeros(nSamples-1,1);%edge prediction accuracy
EAca = zeros(nSamples-1,1);%conditional edge presence
EAcp = zeros(nSamples-1,1);%conditional edge absence
Prediction = zeros(nTot,nSamples);

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
for t=2:nSamples
    testNodePot = zeros(nTot,2);
    testNodePot(:,2) = exp(cov(:,:,t)*Ck);
    testNodePot(:,1) = ones(nTot,1);
    testNodePot = testNodePot ./ repmat(sum(testNodePot,2),[1,2]);

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
%     cov(:,end,t+1) = Prediction(:,t)-1;
%     edgeCov(:,5,t) = repmat(log(sum(Prediction(:,t)-1)+eps),[nEdges,1]);
%     eCov(:,:,5,t) = repmat(log(sum(Prediction(:,t)-1)+eps),[nObs,nObs]);
end

%Bernoulli Edge Prediction
for t = 2:nSamples  
    EAS = zeros(1,samNum);
    EASca = zeros(1,samNum);
    EAScp = zeros(1,samNum);
    subEdgeIndTrue = find(samples(1:nObs,t)==2);%present vertices
    subEdgeInd = find(Prediction(1:nObs,t)==2);%vertices predicted as present
%     for s = 1:samNum
       ePrediction = 2*ones(nObs,nObs,samNum);%2 indicates edge inapplicability
       for i=1:length(subEdgeInd)
            for j=1:i-1
                x = reshape(eCov(subEdgeInd(i),subEdgeInd(j),:,t),[nEcov+1,1])';
                p = logistic(x , Dk);
                for es = 1:samNum
                    ePrediction(subEdgeInd(i),subEdgeInd(j),es) = binornd(1,p);
                end
            end
       end
       %here you were supposed to make ePrediction symmetric but you
       %couldn't because of 2's

%        eCov(:,:,7,t+1) = zeros(nObs,nObs);
%        for i=1:length(subEdgeInd)
%            for j=1:i-1
%                eCov(subEdgeInd(i),subEdgeInd(j),7,t+1) = mean(ePrediction(subEdgeInd(i),subEdgeInd(j),:));
%            end
%        end
%        eCov(:,:,7,t+1) = eCov(:,:,7,t+1) + eCov(:,:,7,t+1)';
       
       tmp = [];
       tmpCA = [];
       tmpCP = [];
       ctr = 1;
       pctr = 0;
       actr = 0;
       for i=1:length(subEdgeIndTrue)
           for j=1:i-1
               etmp = 0;
               etmpCA = 0;
               etmpCP = 0;
               
               for es = 1:samNum
                   %conditioning on the non-NA edges
                   etmp = etmp + (ePrediction(subEdgeIndTrue(i),subEdgeIndTrue(j),es)==eSamples(subEdgeIndTrue(i),subEdgeIndTrue(j),t));
                   if eSamples(subEdgeIndTrue(i),subEdgeIndTrue(j),t)==0
                       etmpCA = etmpCA + (ePrediction(subEdgeIndTrue(i),subEdgeIndTrue(j),es)==eSamples(subEdgeIndTrue(i),subEdgeIndTrue(j),t));
                   elseif eSamples(subEdgeIndTrue(i),subEdgeIndTrue(j),t)==1
                       etmpCP = etmpCP + (ePrediction(subEdgeIndTrue(i),subEdgeIndTrue(j),es)==eSamples(subEdgeIndTrue(i),subEdgeIndTrue(j),t));
                   end
               end
               tmp(ctr) = etmp / samNum;
               if eSamples(subEdgeIndTrue(i),subEdgeIndTrue(j),t)==0
                   actr = actr + 1;
                   tmpCA(actr) = etmpCA / samNum;
               elseif eSamples(subEdgeIndTrue(i),subEdgeIndTrue(j),t)==1
                   pctr = pctr + 1;
                   tmpCP(pctr) = etmpCP / samNum;
               end
               ctr = ctr + 1;
           end
       end
       if isempty(tmpCP)
           tmpCP = 0;
       end
       if isempty(tmpCA)
           tmpCA = 0;
       end
       if isempty(tmp)
           tmp = 0;
       end
       
%        EAS(s) = mean(tmp);
%        EASca(s) = mean(tmpCA);
%        EAScp(s) = mean(tmpCP);
%     end
    EA(t-1) = mean(tmp);
    EAca(t-1) = mean(tmpCA);
    EAcp(t-1) = mean(tmpCP);
    fprintf('time sample: %d \n',t);
end



