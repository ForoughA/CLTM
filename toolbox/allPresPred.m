%DNR pred with sampling
function [EA,EAcp,EAca] = allPresPred(Dk,eCov,eSamples,samples)

samNum = 100;

nObs = size(samples,1);
nSamples = size(samples,2);
nEcov = length(Dk) - 1;

EA = zeros(nSamples-1,1);%edge prediction accuracy
EAca = zeros(nSamples-1,1);%conditional edge presence
EAcp = zeros(nSamples-1,1);%conditional edge absence

subEdgeIndTrue = 1:nObs;%present vertices
subEdgeInd = 1:nObs;%vertices predicted as present

%Bernoulli Edge Prediction
for t = 2:nSamples  
%     for s = 1:samNum
%        subEdgeIndTrue = find(samples(1:nObs,t)==2);%present vertices
       ePrediction = zeros(nObs,nObs,samNum);%2 indicates edge inapplicability
       for i=1:length(subEdgeInd)
            for j=1:i-1
                x = reshape(eCov(subEdgeInd(i),subEdgeInd(j),:,t),[nEcov+1,1])';
                p = logistic(x , Dk);
                for es = 1:samNum
                    ePrediction(subEdgeInd(i),subEdgeInd(j),es) = binornd(1,p);
                end
            end
       end
       
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
%     end
    EA(t-1) = mean(tmp);
    EAca(t-1) = mean(tmpCA);
    EAcp(t-1) = mean(tmpCP);
    fprintf('time sample: %d \n',t);
end