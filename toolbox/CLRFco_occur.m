function [VP,CP,CA,OC,co_oc] = CLRFco_occur(Ck,dParams,adj,cov,dcov,samples)
%form the edgepotential matrix
%form the node potential matrix
%draw from the vertex dist
%draw from the Bernoully model
%predict

nCov = length(Ck) - 1;
ndepCov = length(dParams);
nDraws = 100;

edgeStruct = UGM_makeEdgeStruct(adj,2);
edgeStruct.maxIter = nDraws;
nTot = length(adj);
nObs = size(samples,1);
nSamples = size(samples,2);
% nEcov = length(Dk) - 1;

VP = zeros(1,nSamples);%vertex prediction accuracy
CP = zeros(1,nSamples);%conditional vertex presence
CA = zeros(1,nSamples);%conditional vertex absence
OC = zeros(1,nSamples);
co_oc = zeros(nObs,nObs,nSamples);

   
% end

for t=1:nSamples
    
    ind0 = find(samples(:,t)==1);
    ind1 = find(samples(:,t)==2);
    
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
       
       OC(t) = OC(t) + isequal(samples(:,t),Predictions(1:nObs,n));
       
       for i=1:nObs
           for j=1:i-1
               co_oc(i,j,t) = co_oc(i,j,t) + (Predictions(i,n)==samples(i,t) && Predictions(j,n)==samples(j,t)); 
           end
       end
       
       
    end
    co_oc(:,:,t) = co_oc(:,:,t) + co_oc(:,:,t)';
end

VP = VP/nDraws;
CP = CP/nDraws;
CA = CA/nDraws;
co_oc = co_oc/nDraws;

    
    
%     co_oc = sum(co_oc,3)/nDraws;


end