function [VP,CP,CA,EA,EAcp,EAca] = condPredBplot(Params,dParams,adj,cov,dCov,Samples,clamped,eSamples,eCov,Dk)

nDraws = 100;
nSamples = size(Samples,2);
nTot = size(adj,1);
nObs = size(Samples,1);
ndepCov = size(dCov,3);
edgeStruct = UGM_makeEdgeStruct(adj,2);
edgeStruct.maxIter = nDraws;
VP = zeros(nSamples,nDraws);
CP = zeros(nSamples,nDraws);
CA = zeros(nSamples,nDraws);
EA = zeros(nSamples,nDraws);
EAcp = zeros(nSamples,nDraws);
EAca = zeros(nSamples,nDraws);
for t=1:nSamples
    ObsInd = find(clamped(1:nObs,t)~=0);
    unObsInd = setdiff(1:nObs,ObsInd);
    ind0 = find(Samples(:,t)==1);
    ind0 = setdiff(ind0,ObsInd);
    ind1edge = find(Samples(:,t)==2);
    ind1 = setdiff(ind1edge,ObsInd);
    
    ePrediction = tril(2*ones(nObs,nObs),-1);%2 indicates edge inapplicability
    ePrediction = repmat(ePrediction,[1,1,nDraws]);
    
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
       VP(t,n) = mean(clrfPredictions(unObsInd,n)==Samples(unObsInd,t));
       if ~isempty(ind0)
           CA(t,n) = sum(clrfPredictions(ind0,n)==1)/length(ind0);
       end
       if ~isempty(ind1)
           CP(t,n) = sum(clrfPredictions(ind1,n)==2)/length(ind1);
       end
       
       %comment out this part upto end of n=1:nDraws to do only vertex
       %prediction
       %edge sample draw
       subEdgeInd = find(clrfPredictions(1:nObs,n)==2);
       
        ePrediction(subEdgeInd,subEdgeInd,n) = 0;
        mask = zeros(nObs);
        mask(subEdgeInd,subEdgeInd) = 1;
        mask = tril(mask,-1);
        
        p = logistic(eCov(:,:,:,t),Dk);
        p = p .* mask;
        
        s = binornd(1,p,[nObs,nObs]);
        ePrediction(:,:,n) = ePrediction(:,:,n) + s;
        
        ePrediction(:,:,n) = ePrediction(:,:,n) + ePrediction(:,:,n)';
       
       %edge prediction
       etmp = 0;
       etmpCA = 0;
       etmpCP = 0;
       ctr = 0;
       pctr = 0;
       actr = 0;
       for i=1:length(ind1edge)
           for j=1:i-1
                   %conditioning on the non-NA edges
                   etmp = etmp + (ePrediction(ind1edge(i),ind1edge(j),n)==eSamples(ind1edge(i),ind1edge(j),t));
                   if eSamples(ind1edge(i),ind1edge(j),t)==0
                       etmpCA = etmpCA + (ePrediction(ind1edge(i),ind1edge(j),n)==eSamples(ind1edge(i),ind1edge(j),t));
                       actr = actr + 1;
                   elseif eSamples(ind1edge(i),ind1edge(j),t)==1
                       etmpCP = etmpCP + (ePrediction(ind1edge(i),ind1edge(j),n)==eSamples(ind1edge(i),ind1edge(j),t));
                       pctr = pctr + 1;
                   end
               
               ctr = ctr + 1;
           end
       end
       if ctr~=0
            EA(t,n) = etmp/ctr;
       end
       if pctr~=0
            EAcp(t,n) = etmpCP/pctr;
       end
       if actr~=0
            EAca(t,n) = etmpCA/actr;
       end
       
    end
    
%     VP(t) = VP(t)/nDraws;
%     CP(t) = CP(t)/nDraws;
%     CA(t) = CA(t)/nDraws;
%     
%     EA(t) = EA(t)/nDraws;
%     EAcp(t) = EAcp(t)/nDraws;
%     EAca(t) = EAca(t)/nDraws;
     
end