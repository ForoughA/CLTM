
<!-- saved from url=(0072)http://www.di.ens.fr/~mschmidt/Software/UGM/updates/UGM_makeEdgeStruct.m -->
<html><head><meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1"></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">function [edgeStruct] = UGM_makeEdgeStruct(adj,nStates,useMex,maxIter)
% [edgeStruct] = UGM_makeEdgeStruct(adj,nStates,useMex,maxIter)
%
% adj - nNodes by nNodes adjacency matrix (0 along diagonal)
%

if nargin &lt; 3
    useMex = 1;
end
if nargin &lt; 4
    maxIter = 100;
end

nNodes = int32(length(adj));
[i j] = ind2sub([nNodes nNodes],find(adj));
nEdges = length(i)/2;
edgeEnds = zeros(nEdges,2,'int32');
eNum = 0;
for e = 1:nEdges*2
   if j(e) &lt; i(e)
       edgeEnds(eNum+1,:) = [j(e) i(e)];
       eNum = eNum+1;
   end
end

[V,E] = UGM_makeEdgeVE(edgeEnds,nNodes,useMex);


edgeStruct.edgeEnds = edgeEnds;
edgeStruct.V = V;
edgeStruct.E = E;
edgeStruct.nNodes = nNodes;
edgeStruct.nEdges = size(edgeEnds,1);

% Handle other arguments
if isscalar(nStates)
   nStates = repmat(nStates,[double(nNodes) 1]);
end
edgeStruct.nStates = int32(nStates(:));
edgeStruct.useMex = useMex;
edgeStruct.maxIter = int32(maxIter);


</pre></body></html>