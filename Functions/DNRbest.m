clear
model = 1;
runs = 2;
trainMean = zeros(model,runs);
testMean = zeros(model,runs);
figure;
for i=1:model
    for j=1:runs
       load(sprintf('DNRvertexParamsFullTwitter/vertexOutCond%d',j));
%        figure;plot(testResults.VP);hold on;plot(trainResults.VP,'r')
       testMean(i,j) = mean(testResults.VP);
       trainMean(i,j) = mean(trainResults.VP);
       plot(DNRout.params);hold on
       ll(i,j) = DNRout.LL(end);
    end
end