
<!-- saved from url=(0066)http://www.di.ens.fr/~mschmidt/Software/UGM/updates/UGM_ChainFwd.m -->
<html><head><meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1"></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">function [alpha,kappa,mxState] = UGM_ChainFwd(nodePot,edgePot,nStates,maximize)
[nNodes,maxState] = size(nodePot);

mxState = zeros(nNodes,maxState);
alpha = zeros(nNodes,maxState);
alpha(1,1:nStates(1)) = nodePot(1,1:nStates(1));
kappa(1) = sum(alpha(1,1:nStates(1)));
alpha(1,1:nStates(1)) = alpha(1,1:nStates(1))/kappa(1);
for n = 2:nNodes
   tmp = repmat(alpha(n-1,1:nStates(n-1))',1,nStates(n)).*edgePot(1:nStates(n-1),1:nStates(n),n-1);
   if maximize
      alpha(n,1:nStates(n)) = nodePot(n,1:nStates(n)).*max(tmp);
      [mxPot mxState(n,1:nStates(n))] = max(tmp);
   else
       alpha(n,1:nStates(n)) = nodePot(n,1:nStates(n)).*sum(tmp);
   end
   
   % Normalize Message
   kappa(n) = sum(alpha(n,1:nStates(n)));
   alpha(n,1:nStates(n)) = alpha(n,1:nStates(n))/kappa(n);
end</pre></body></html>