function [VP,CP,CA] = condPredBerNoEdge(Ck,cov,Samples,clamped,isBox)

if nargin == 4 
    isBox = 0;
end

nDraws = 100;
nSamples = size(Samples,2);
nObs = size(Samples,1);
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
    ObsInd = find(clamped(1:nObs,t)~=-1);
    if isempty(ObsInd) && ~isempty(find(clamped+1,1))
        error('initialize clampedDNR with -1*ones');
    end
    unObsInd = setdiff(1:nObs,ObsInd);
    ind0 = find(Samples(:,t)==0);
    ind0 = setdiff(ind0,ObsInd);
    ind1edge = find(Samples(:,t)==1);
    ind1 = setdiff(ind1edge,ObsInd);
    
%     ePrediction = tril(2*ones(nObs,nObs),-1);%2 indicates edge inapplicability
%     ePrediction = repmat(ePrediction,[1,1,nDraws]);
    ePrediction = zeros(nObs,nObs,nDraws);
    
    p = zeros(nObs,1);
    p(ObsInd) = Samples(ObsInd,t);
    p(unObsInd) = logistic(cov(unObsInd,:,t),Ck);
    
    dnrPredictions = binornd(1,repmat(p,[1,nDraws]),[nObs,nDraws]);
    
    
    for n=1:nDraws
        if isBox
           VP(t,n) = mean(dnrPredictions(unObsInd,n)==Samples(unObsInd,t));
           if ~isempty(ind0)
               CA(t,n) = sum(dnrPredictions(ind0,n)==0)/length(ind0);
           end
           if ~isempty(ind1)
               CP(t,n) = sum(dnrPredictions(ind1,n)==1)/length(ind1);
           end 
        else
           VP(t) = VP(t) + mean(dnrPredictions(unObsInd,n)==Samples(unObsInd,t));
           if ~isempty(ind0)
               CA(t) = CA(t) + sum(dnrPredictions(ind0,n)==0)/length(ind0);
           end
           if ~isempty(ind1)
               CP(t) = CP(t) + sum(dnrPredictions(ind1,n)==1)/length(ind1);
           end
        end
       
       %edge sample draw and prediction
%        subEdgeInd = find(dnrPredictions(1:nObs,n)==1);
%         
%         ePrediction(subEdgeInd,subEdgeInd,n) = 0;
%         mask = zeros(nObs);
%         mask(subEdgeInd,subEdgeInd) = 1;
%         mask = tril(mask,-1);
%         
%         p = logistic(eCov(:,:,:,t),Dk);
%         p = p .* mask;
%         
%         s = binornd(1,p,[nObs,nObs]);
%         ePrediction(:,:,n) = ePrediction(:,:,n) + s;
%         
%         ePrediction(:,:,n) = ePrediction(:,:,n) + ePrediction(:,:,n)';
       
%        %edge prediction
%        etmp = 0;
%        etmpCA = 0;
%        etmpCP = 0;
%        ctr = 0;
%        pctr = 0;
%        actr = 0;
%        for i=1:nObs
%            for j=1:i-1
%                    %conditioning on the non-NA edges
%                    etmp = etmp + (ePrediction(i,j,n)==eSamples(i,j,t));
%                    if eSamples(i,j,t)==0
%                        etmpCA = etmpCA + (ePrediction(i,j,n)==eSamples(i,j,t));
%                        actr = actr + 1;
%                    elseif eSamples(i,j,t)==1
%                        etmpCP = etmpCP + (ePrediction(i,j,n)==eSamples(i,j,t));
%                        pctr = pctr + 1;
%                    end
%                
%                ctr = ctr + 1;
%            end
%        end
%        if isBox
%            if ctr~=0
%                 EA(t,n) = etmp/ctr;
%            end
%            if pctr~=0
%                 EAcp(t,n) = etmpCP/pctr;
%            end
%            if actr~=0
%                 EAca(t,n) = etmpCA/actr;
%            end
%        else
%            if ctr~=0
%                 EA(t) = EA(t) + etmp/ctr;
%            end
%            if pctr~=0
%                 EAcp(t) = EAcp(t) + etmpCP/pctr;
%            end
%            if actr~=0
%                 EAca(t) = EAca(t) + etmpCA/actr;
%            end
%        end
%        
    end
    
    if ~isBox
        VP(t) = VP(t)/nDraws;
        CP(t) = CP(t)/nDraws;
        CA(t) = CA(t)/nDraws;

        EA(t) = EA(t)/nDraws;
        EAcp(t) = EAcp(t)/nDraws;
        EAca(t) = EAca(t)/nDraws;
    end

end