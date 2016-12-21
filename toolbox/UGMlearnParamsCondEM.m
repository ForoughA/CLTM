function [nodePot, edgePot, w, NLL] = UGMlearnParamsCondEM(samples, adjmat, covariates, options)

if ~isfield(options,'max_ite')
    options.max_ite = 20;
end

if ~isfield(options,'root')
    root = 1;
else
    root = options.root;
end

inputSamples = samples;
%sample type accepted by UGM
samples = int32(samples)';

Ntotal = size(adjmat,1);
Nobserved = size(inputSamples,1);
Nsamples = size(inputSamples,2);
hnodes = Nobserved+1:Ntotal;
NFeatures = size(covariates,2);

maxState = max(max(samples));
% maxState = max(nStates);
edgeStruct = UGM_makeEdgeStruct(adjmat,maxState);
edgeStruct.useMex = 0;
nEdges = edgeStruct.nEdges;

Xnode = zeros(Nsamples,NFeatures,Nobserved);
temp = zeros(NFeatures,Nsamples);
for ctr = 1:Nobserved
    temp(:,:) = covariates(ctr,:,:);
    Xnode(:,:,ctr) = temp';
end
Xnode = [ones(Nsamples,1,Nobserved) Xnode];
NnodeFeatures = size(Xnode,2);
Xedge = ones(Nsamples,1,nEdges);%no edge features

nodeMap = zeros(Nobserved,maxState,NnodeFeatures,'int32');
for f = 1:NnodeFeatures
    nodeMap(:,1,f) = f;
end
edgeMap = zeros(maxState,maxState,nEdges,'int32');
edgeMap(1,1,:) = NnodeFeatures+1;
edgeMap(2,1,:) = NnodeFeatures+2;
edgeMap(1,2,:) = NnodeFeatures+3;

nParams = max([nodeMap(:);edgeMap(:)]);
w = rand(nParams,1);

degree_hnodes = sum(adjmat(hnodes,hnodes),2);
tree_msg_order = treeMsgOrder(adjmat,root);
edge_pairs = tree_msg_order(size(tree_msg_order,1)/2+1:end,:);

nodePot = zeros(Ntotal,2);
edgePot = zeros(2*Ntotal,2*Ntotal);
if(isfield(options,'initNodePot') && isfield(options,'initEdgePot'))
    nodePot = options.initNodePot;
    edgePot = options.initEdgePot;
elseif(isfield(options,'edge_distance')) % Initialize parameters with edge distance
    prob_bij = initParamDist(options.edge_distance, edge_pairs, inputSamples);
    [nodePot, edgePot] = marToPotBin(prob_bij,tree_msg_order);
else
    nodePot(:,1) = rand(Ntotal,1);
    nodePot(:,2) = 1-nodePot(:,1);
    for e=1:size(edge_pairs,1)
        u = edge_pairs(e,1);
        v = edge_pairs(e,2);        
        r4 = rand(2);
        edgePot(2*u-1:2*u,2*v-1:2*v) = r4/sum(r4(:));
    end
    edgePot = edgePot + edgePot';
end

UGMedgePot = zeros(maxState,maxState,nEdges);
for e = 1:nEdges
    edgeEnds = edgeStruct.edgeEnds(e,:);
    UGMedgePot(:,:,e) = edgePot(2*edgeEnds(1)-1:2*edgeEnds(1),2*edgeEnds(2)-1:2*edgeEnds(2));
end
UGMnodePot = nodePot;

NLL = zeros(options.max_ite,1);
tiny = 1e-20;
for ite=1:5%options.max_ite
        entropy = 0;
    % E step: compute statistics of hidden variables for each sample using
    % the current parameter setting.
    new_prob_bij = zeros(2*Ntotal,2*Ntotal);  
    for n=1:Nsamples
        
        %Note: I dont know which of the following is correct for setting
        %the initial potentials. Check sumPorductBin if it gives the
        %marginal that you want for some observed node then maybe you
        %should use the second one that is also used by Jin
%         for node = 1:Nobserved
%             nodePot(node,2) = covariates(node,:,n)*w(1:NnodeFeatures);
%             if inputSamples(node,n) == 1
%                nodePot(node,1) = 10*nodePot(node,2);% if we have not observed
%                     %the vertex then the potential of non-occurance is higher
%             else
%                nodePot(node,1) = 1/10*nodePot(node,2); 
%             end
%             if nodePot(node,2)~=0 && nodePot(node,2) ~= 1
%                 nodePot(node,:) = nodePot(node,:)/sum(nodePot(node,:));
%             elseif nodePot(node,2)==1 && inputSamples(node,n)==1
%                 error('this cant happen')
%             elseif nodePot(node,2)==1 && inputSamples(node,n)==2
%                 nodePot(node,1) = 0;
%             elseif nodePot(node,2)==0
%                 nodePot(node,1) = 1;
%             end
        if edgeStruct.useMex
            [UGMnodePot(1:Nobserved,:),UGMedgePot] = UGM_CRF_makePotentialsC(w,Xnode,Xedge,nodeMap,edgeMap,nStates,edgeEnds,int32(n));
        else
            [UGMnodePot(1:Nobserved,:),UGMedgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,n);
        end
        
         [node_marginals,edge_marginals,logZ] = UGM_Infer_Tree(UGMnodePot,UGMedgePot,edgeStruct);          
            
    end

%         nodePot(1:Nobserved,:) = repmat([1 0],Nobserved,1);    
%         indSample2 = (inputSamples(:,n)==2);
%         nodePot(indSample2,1) = 0;
%         nodePot(indSample2,2) = 1;  
             
        

%         [node_marginals, edge_marginals] = sumProductBin(adjmat,nodePot,edgePot,tree_msg_order);
        
        
    
    % M step: Set the parameters using the complete statistics
    
    
   
    %Converting the potentials to those compatible with UGM for the M-Step
%     UGMedgePot = zeros(maxState,maxState,nEdges);
%     for e = 1:nEdges
%         edgeEnds = edgeStruct.edgeEnds(e,:);
%         UGMedgePot(:,:,e) = edgePot(2*edgeEnds(1)-1:2*edgeEnds(1),2*edgeEnds(2)-1:2*edgeEnds(2));
%     end
%     UGMnodePot = nodePot;
    
    
    
end



