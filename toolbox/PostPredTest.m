function [VP,CP,CA,EA,EAcp,EAca] = PostPredTest(Ck,Dk,edgePot,adj,cov,eCov,samples,eSamples)
%form the edgepotential matrix
%form the node potential matrix
%draw from the vertex dist
%draw from the Bernoully model
%predict

nCov = length(Ck) - 1;
nDraws = 100;

edgeStruct = UGM_makeEdgeStruct(adj,2);
edgeStruct.maxIter = 1;
nTot = length(adj);
nObs = size(samples,1);
nSamples = size(samples,2);
nEcov = length(Dk) - 1;

VP = zeros(nSamples-1,1);%vertex prediction accuracy
CP = zeros(nSamples-1,1);%conditional vertex presence
CPna = [];%non-applicable CA draws
CA = zeros(nSamples-1,1);%conditional vertex absence
CAna = [];%non-applicable CA draws
EA = zeros(nSamples-1,1);%edge prediction accuracy
EAca = zeros(nSamples-1,1);%conditional edge presence
EAcana = [];%non-applicable EAca draws
EAcp = zeros(nSamples-1,1);%conditional edge absence
EAcpna = [];%non-applicable EAcp draws


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

for t=2:nSamples
    Predictions = zeros(nTot,nDraws);
    ePrediction = tril(2*ones(nObs,nObs),-1);%2 indicates edge inapplicability
    ePrediction = repmat(ePrediction,[1,1,nDraws]);
    
    ind0 = find(samples(1:nObs,t)==1);%this can be out of the loop but I will put it here for now
    ind1 = find(samples(1:nObs,t)==2);
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
        
    for n = 1:nDraws        
        %draw one node sample
        nodePot(:,2) = exp(cov(:,:,t) * Ck);
        nodePot(:,1) = ones(nTot,1);
        Predictions(:,n) = UGM_Sample_Tree(nodePot,testEdgePot,edgeStruct);%drawing a vertex sample
        
        %draw one edge sample conditioned on the drawn node sample
        subEdgeInd = find(Predictions(1:nObs,n)==2);
       for i=1:length(subEdgeInd)
            for j=1:i-1
                x = reshape(eCov(subEdgeInd(i),subEdgeInd(j),:,t),[nEcov+1,1])';
                p = logistic(x , Dk);
                ePrediction(subEdgeInd(i),subEdgeInd(j),n) = binornd(1,p);
            end
       end
       ePrediction(:,:,n) = ePrediction(:,:,n) + ePrediction(:,:,n)';
       
       %compare the drawn edge and vertex samples to the true ones
       %vertex prediction
       VP(t-1) = VP(t-1) + mean(Predictions(1:nObs,n)==samples(:,t));
       if ~isempty(ind0)
           CA(t-1) = CA(t-1) + sum(Predictions(ind0,n)==1)/length(ind0);
       end
       if ~isempty(ind1)
           CP(t-1) = CP(t-1) + sum(Predictions(ind1,n)==2)/length(ind1);
       end
       
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
            EA(t-1) = EA(t-1) + etmp/ctr;
       end
       if pctr~=0
            EAcp(t-1) = EAcp(t-1) + etmpCP/pctr;
       end
       if actr~=0
            EAca(t-1) = EAca(t-1) + etmpCA/actr;
       end
    end
    
    VP(t-1) = VP(t-1)/nDraws;
    CA(t-1) = CA(t-1)/nDraws;
    CP(t-1) = CP(t-1)/nDraws;
    EA(t-1) = EA(t-1)/nDraws;
    EAcp(t-1) = EAcp(t-1)/nDraws;
    EAca(t-1) = EAca(t-1)/nDraws;
     
end

%omitting the time points that had no present/absent vertices since this is
%a conditional measure of performance
if ~isempty(CAna)
    for i=length(CAna):-1:1
        CA(CAna(i)-1) = [];
    end
end
if ~isempty(CPna)
    for i=length(CPna):-1:1
        CP(CPna(i)-1) = [];
        EA(CPna(i)-1) = [];%if there are no present samples, then EA for
        %that time point is not defined
    end
end
if ~isempty(EAcana)
    for i=length(EAcana):-1:1
        EAca(EAcana(i)-1) = [];
    end
end
if ~isempty(EAcpna)
    for i=length(EAcpna):-1:1
        EAcp(EAcpna(i)-1) = [];
    end
end

end