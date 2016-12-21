%DNR pred with sampling
function [VP,CP,CA,EA,EAca,EAcp] = DNRpredBernoulliOrig(Ck,Dk,cov,eCov,samples,eSamples)

nObs = size(samples,1);
nSamples = size(samples,2);
nEcov = length(Dk) - 1;
nCov = length(Ck) - 1;
nDraws = 100;

%Vertex prediction
VP = zeros(nSamples-1,1);
CP = zeros(nSamples-1,1);
CPna = [];%non-applicable CA draws
CA = zeros(nSamples-1,1);
CAna = [];%non-applicable CP draws
EA = zeros(nSamples-1,1);
EAca = zeros(nSamples-1,1);
EAcana = [];%non-applicable EAca draws
EAcp = zeros(nSamples-1,1);
EAcpna = [];%non-applicable EAcp draws

for t=2:nSamples
    Predictions = zeros(nObs,nDraws);
    ePrediction = tril(2*ones(nObs,nObs),-1);%2 indicates edge inapplicability
    ePrediction = repmat(ePrediction,[1,1,nDraws]);

    ind0 = find(samples(1:nObs,t)==0);%this can be out of the loop but I will put it here for now
    ind1 = find(samples(1:nObs,t)==1);
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
        for i = 1:nObs
            p = logistic(cov(i,:,t),Ck);
            Predictions(i,n) = binornd(1,p);
        end
        %draw one edge sample conditioned on the drawn node sample
        subEdgeInd = find(Predictions(1:nObs,n)==1);
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
           CA(t-1) = CA(t-1) + sum(Predictions(ind0,n)==0)/length(ind0);
       end
       if ~isempty(ind1)
           CP(t-1) = CP(t-1) + sum(Predictions(ind1,n)==1)/length(ind1);
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