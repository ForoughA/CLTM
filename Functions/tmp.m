clear
figure
for i=1:5
   load(sprintf('/home/forough/dvp/CleanedUpCodes/CLRFvertexParamsFullTwitter/Results6/edgeOutCond%d.mat',i))
    plot(eParams); hold on
    LL(i) = eLL(end);
end
