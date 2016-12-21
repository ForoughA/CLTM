function Per = computePerplexity(Params,edgePot,adj)

nNodes = size(adj,1);

%all possible covariate configuations in order to compute the weighted
%average of conditional entropy.
cov = [1     0     1     0     0     0     0     0     0     0;
    1     1     1     0     0     0     0     0     0     0;
    1     0     1     0     0     0     0     0     0     1;
    1     1     1     0     0     0     0     0     0     1;
    1     0     0     1     0     0     0     0     0     0;
    1     1     0     1     0     0     0     0     0     0;
    1     0     0     1     0     0     0     0     0     1;
    1     1     0     1     0     0     0     0     0     1;
    1     0     0     0     1     0     0     0     0     0;
    1     1     0     0     1     0     0     0     0     0;
    1     0     0     0     1     0     0     0     0     1;
    1     1     0     0     1     0     0     0     0     1;
    1     0     0     0     0     1     0     0     0     0;
    1     1     0     0     0     1     0     0     0     0;
    1     0     0     0     0     1     0     0     0     1;
    1     1     0     0     0     1     0     0     0     1;
    1     0     0     0     0     0     1     0     0     0;
    1     1     0     0     0     0     1     0     0     0;
    1     0     0     0     0     0     1     0     0     1;
    1     1     0     0     0     0     1     0     0     1;
    1     0     0     0     0     0     0     1     0     0;
    1     1     0     0     0     0     0     1     0     0;
    1     0     0     0     0     0     0     1     0     1;
    1     1     0     0     0     0     0     1     0     1;
    1     0     0     0     0     0     0     0     1     0;
    1     1     0     0     0     0     0     0     1     0;
    1     0     0     0     0     0     0     0     1     1;
    1     1     0     0     0     0     0     0     1     1];

covStates = size(cov,1);
H = 0;
for i=1:covStates
   nodePot=ones(nNodes,2);
   nodePot(:,2) = exp(cov(i,:)*Params);
   H = H + computeEntropy(nodePot,edgePot,adj);
end

Per = H/covStates;%conditional entropy
Per = 2 ^(-Per);
    