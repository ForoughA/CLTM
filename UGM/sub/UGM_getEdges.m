
<!-- saved from url=(0066)http://www.di.ens.fr/~mschmidt/Software/UGM/updates/UGM_getEdges.m -->
<html><head><meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1"></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">function [edges] = UGM_getEdges(n,edgeStruct)
edges = edgeStruct.E(edgeStruct.V(n):edgeStruct.V(n+1)-1)';
</pre></body></html>