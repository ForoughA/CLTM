function [VP,CP,CA,OC,co_oc] = DNRco_occur(Ck,cov,samples)
%form the edgepotential matrix
%form the node potential matrix
%draw from the vertex dist
%draw from the Bernoully model
%predict

nCov = length(Ck) - 1;
nDraws = 100;

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
    
    ind0 = find(samples(:,t)==0);
    ind1 = find(samples(:,t)==1);
    
    p = logistic(cov(1:nObs,:,t),Ck);
    Predictions = binornd(1,repmat(p,[1,nDraws]),[nObs,nDraws]);
    
    for n = 1:nDraws        
        %draw one node sample
        VP(t) = VP(t) + mean(Predictions(1:nObs,n)==samples(1:nObs,t));
       if ~isempty(ind0)
           CA(t) = CA(t) + sum(Predictions(ind0,n)==0)/length(ind0);
       end
       if ~isempty(ind1)
           CP(t) = CP(t) + sum(Predictions(ind1,n)==1)/length(ind1);
       end
       
       OC(t) = OC(t) + isequal(samples(:,t),Predictions(:,n));
       
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

end