function [VP,CP,CA] = condPredEdu(Params,dParams,adj,cov,dCov,Samples,clamped,isBox)

if nargin==10
    isBox = 0;
end

nDraws = 100;
nSamples = size(Samples,2);
nTot = size(adj,1);
nObs = size(Samples,1);
ndepCov = size(dCov,3);
edgeStruct = UGM_makeEdgeStruct(adj,2);
edgeStruct.maxIter = nDraws;
if isBox
    VP = zeros(nSamples,nDraws);
    CP = zeros(nSamples,nDraws);
    CA = zeros(nSamples,nDraws);
    EA = zeros(nSamples,nDraws);
    EAcp = zeros(nSamples,nDraws);
    EAca = zeros(nSamples,nDraws);
else
    VP = zeros(nSamples,1);
    CP = zeros(nSamples,1);
    CA = zeros(nSamples,1);
    EA = zeros(nSamples,1);
    EAcp = zeros(nSamples,1);
    EAca = zeros(nSamples,1);
end
for t=1:nSamples
    ObsInd = find(clamped(1:nObs,t)~=0);
    unObsInd = setdiff(1:nObs,ObsInd);
    ind0 = find(Samples(:,t)==1);
    ind0 = setdiff(ind0,ObsInd);
    ind1edge = find(Samples(:,t)==2);
    ind1 = setdiff(ind1edge,ObsInd);
    
%     ePrediction = tril(2*ones(nObs,nObs),-1);%2 indicates edge inapplicability
%     ePrediction = repmat(ePrediction,[1,1,nDraws]);
    ePrediction = zeros(nObs,nObs,nDraws);
    
    nodePot(:,2) = exp(cov(:,:,t) * Params);
    nodePot(:,1) = ones(nTot,1);
    edgePot = ones(2,2,edgeStruct.nEdges);
    for e=1:edgeStruct.nEdges
        par = edgeStruct.edgeEnds(e,1);
        child = edgeStruct.edgeEnds(e,2);
        edgePot(2,2,e) = exp(reshape(dCov(par,child,:,t),[1,ndepCov])*dParams);
    end
    clrfPredictions = UGM_Sample_Conditional(nodePot,edgePot,edgeStruct,clamped(:,t),@UGM_Sample_Tree);
    
    for n=1:nDraws
        
        if isBox
            VP(t,n) = mean(clrfPredictions(unObsInd,n)==Samples(unObsInd,t));
           if ~isempty(ind0)
               CA(t,n) = sum(clrfPredictions(ind0,n)==1)/length(ind0);
           end
           if ~isempty(ind1)
               CP(t,n) = sum(clrfPredictions(ind1,n)==2)/length(ind1);
           end
           
        else
        
           VP(t) = VP(t) + mean(clrfPredictions(unObsInd,n)==Samples(unObsInd,t));
           if ~isempty(ind0)
               CA(t) = CA(t) + sum(clrfPredictions(ind0,n)==1)/length(ind0);
           end
           if ~isempty(ind1)
               CP(t) = CP(t) + sum(clrfPredictions(ind1,n)==2)/length(ind1);
           end
       
        end
       
    end
    
    if ~isBox
        VP(t) = VP(t)/nDraws;
        CP(t) = CP(t)/nDraws;
        CA(t) = CA(t)/nDraws;

        EA(t) = EA(t)/nDraws;
        EAcp(t) = EAcp(t)/nDraws;
        EAca(t) = EAca(t)/nDraws;
    end
    
%     p = zeros(nObs,1);
%     p(ObsInd) = logistic(cov(ObsInd,:,t),Ck);
%     dnrPredictions
%     Predictions(:,t,:) = binornd(1,repmat(p,[1,1,nDraws]),[nObs,1,nDraws]);
%     dnrPredictions = 
end

