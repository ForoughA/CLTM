function [VP,CP,CA,EA,EAcp,EAca] = PostPredTestBoth(Ck,Dk,dParams,adj,cov,dcov,eCov,samples,eSamples)
%form the edgepotential matrix
%form the node potential matrix
%draw from the vertex dist
%draw from the Bernoully model
%predict

nCov = length(Ck) - 1;
nDraws = 100;

edgeStruct = UGM_makeEdgeStruct(adj,2);
edgeStruct.maxIter = nDraws;
nTot = length(adj);
nObs = size(samples,1);
nSamples = size(samples,2);
nEcov = length(Dk) - 1;
ndepCov = size(dcov,3);

VP = zeros(nSamples,1);%vertex prediction accuracy
CP = zeros(nSamples,1);%conditional vertex presence
CPna = [];%non-applicable CA draws
CA = zeros(nSamples,1);%conditional vertex absence
CAna = [];%non-applicable CA draws
EA = zeros(nSamples,1);%edge prediction accuracy
EAca = zeros(nSamples,1);%conditional edge presence
EAcana = [];%non-applicable EAca draws
EAcp = zeros(nSamples,1);%conditional edge absence
EAcpna = [];%non-applicable EAcp draws


% if size(edgePot,3) < 2
%     edgePot = full(edgePot);
%     testEdgePot = zeros(2,2,edgeStruct.nEdges);
%     for e = 1:edgeStruct.nEdges
%         child = edgeStruct.edgeEnds(e,1);
%         parent = edgeStruct.edgeEnds(e,2);
%         testEdgePot(:,:,e) = edgePot(2*parent-1:2*parent,2*child-1:2*child);
%     end
% else
%     testEdgePot = edgePot;
% end


for t=1:nSamples
    ind0 = find(samples(1:nObs,t)==1);
    ind1 = find(samples(1:nObs,t)==2);
    
    ePrediction = tril(2*ones(nObs,nObs),-1);%2 indicates edge inapplicability
    ePrediction = repmat(ePrediction,[1,1,nDraws]);
    
    
%     ind0 = find(samples(1:nObs,t)==1);%this can be out of the loop but I will put it here for now
    
    if isempty(ind0)
        CAna = [CAna,t];
    elseif isempty(ind1)
        CPna = [CPna,t];
    end
    
    if isempty(find(eSamples(:,:,t)==0,1))
        EAcana = [EAcana,t];
    end
    if isempty(find(eSamples(:,:,t)==1,1))
        EAcpna = [EAcpna,t];
    end
    nodePot(:,2) = exp(cov(:,:,t) * Ck);
    nodePot(:,1) = ones(nTot,1);
    testEdgePot = ones(2,2,edgeStruct.nEdges);
    for e=1:edgeStruct.nEdges
        par = edgeStruct.edgeEnds(e,1);
        child = edgeStruct.edgeEnds(e,2);
        testEdgePot(2,2,e) = exp(reshape(dcov(par,child,:,t),[1,ndepCov])*dParams);
    end
    Predictions = UGM_Sample_Tree(nodePot,testEdgePot,edgeStruct);%drawing a vertex sample
   
    for n = 1:nDraws        
        %draw one node sample
        VP(t) = VP(t) + mean(Predictions(1:nObs,n)==samples(1:nObs,t));
       if ~isempty(ind0)
           CA(t) = CA(t) + sum(Predictions(ind0,n)==1)/length(ind0);
       end
       if ~isempty(ind1)
           CP(t) = CP(t) + sum(Predictions(ind1,n)==2)/length(ind1);
       end
        
        %draw one edge sample conditioned on the drawn node sample
        subEdgeInd = find(Predictions(1:nObs,n)==2);
        
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
       for i=1:length(ind1)
           for j=1:i-1
                   %conditioning on the non-NA edges
                   etmp = etmp + (ePrediction(ind1(i),ind1(j),n)==eSamples(ind1(i),ind1(j),t));
                   if eSamples(ind1(i),ind1(j),t)==0
                       etmpCA = etmpCA + (ePrediction(ind1(i),ind1(j),n)==eSamples(ind1(i),ind1(j),t));
                       actr = actr + 1;
                   elseif eSamples(ind1(i),ind1(j),t)==1
                       etmpCP = etmpCP + (ePrediction(ind1(i),ind1(j),n)==eSamples(ind1(i),ind1(j),t));
                       pctr = pctr + 1;
                   end
               
               ctr = ctr + 1;
           end
       end
       if ctr~=0
            EA(t) = EA(t) + etmp/ctr;
       end
       if pctr~=0
            EAcp(t) = EAcp(t) + etmpCP/pctr;
       end
       if actr~=0
            EAca(t) = EAca(t) + etmpCA/actr;
       end
    end
    
    VP(t) = VP(t)/nDraws;
    CA(t) = CA(t)/nDraws;
    CP(t) = CP(t)/nDraws;
    EA(t) = EA(t)/nDraws;
    EAcp(t) = EAcp(t)/nDraws;
    EAca(t) = EAca(t)/nDraws;
     
end

% samp = samples(1:nObs,:);
% ind0 = find(samp==0);
% ind1 = find(samp==1);
% 
% for n=1:nDraws
%     pred = Predictions(1:nObs,n);
%     VP(n) = sum(sum(samp==pred))/numel(samp);
%     CP(n) = sum(sum(pred(ind1)==1))/(length(ind1));
%     CA(n) = sum(sum(pred(ind0)==0))/(length(ind0));
%     
% end

%omitting the time points that had no present/absent vertices since this is
%a conditional measure of performance
if ~isempty(CAna)
    for i=length(CAna):-1:1
        CA(CAna(i)) = [];
    end
end
if ~isempty(CPna)
    for i=length(CPna):-1:1
        CP(CPna(i)) = [];
        EA(CPna(i)) = [];%if there are no present samples, then EA for
        %that time point is not defined
    end
end
if ~isempty(EAcana)
    for i=length(EAcana):-1:1
        EAca(EAcana(i)) = [];
    end
end
if ~isempty(EAcpna)
    for i=length(EAcpna):-1:1
        EAcp(EAcpna(i)) = [];
    end
end

end