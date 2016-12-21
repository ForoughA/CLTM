% function [nodePot,edgePot,ll] = learnParamsCondEM2(samples,covariates,adjmat,options)
% 
% obs = int32(samples+1);
% [nNodes,nInstances] = size(obs);
% nStates = max(max(obs));
% nTotal = size(adjmat,1);
% edgeStruct = UGM_makeEdgeStruct(adjmat,nStates);
% nFeatures = size(covariates,2);
% maxState = max(nStates);
% nEdges = edgestruct.nEdges;
% 
% nParams = nTotal*nFeatures + nEdges; 
% w = zeros(nParams,1);
% 
% nodePot = zeros(nTotal*nFeatures,1);
% edgePot = zeros(nEdges,1);
% ll = 
% 
% for iter = 1:options.max_ite
%     
% end

function [nodePot, edgePot, w, ll] = learnParamsCondEM2(samples, adjmat, covariates, options)

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
NnodeFeatures = size(covariates,2);

nStates = max(samples);
maxState = max(nStates);
edgeStruct = UGM_makeEdgeStruct(adj,maxState);
nEdges = edgeStruct.nEdges;

Xnode = zeros(Nsamples,NnodeFeatures,Nobserved);
temp = zeros(NnodeFeatures,Nsamples);
for ctr = 1:Nobserved
    temp(:,:) = covariates(ctr,:,:);
    Xnode(:,:,ctr) = temp';
end
Xedge = ones(Nsamples,1,nEdges);%no edge features

nodeMap = zeros(nNodes,maxState,nNodeFeatures,'int32');
for f = 1:nNodeFeatures
    nodeMap(:,1,f) = f;
end
edgeMap = zeros(maxState,maxState,nEdges,'int32');
edgeMap(1,1,:) = nNodeFeatures+1;
edgeMap(2,1,:) = nNodeFeatures+2;
edgeMap(1,2,:) = nNodeFeatures+3;

nParams = max([nodeMap(:);edgeMap(:)]);
w = zeros(nParams,1);

degree_hnodes = sum(adjmat(hnodes,hnodes),2);
tree_msg_order = treeMsgOrder(adjmat,root);
edge_pairs = tree_msg_order(size(tree_msg_order,1)/2+1:end,:);

nodePot = zeros(Ntotal,2);
edgePot = zeros(2*Ntotal,2*Ntotal);
if(isfield(options,'initNodePot') && isfield(options,'initEdgePot'))
    nodePot = options.initNodePot;
    edgePot = options.initEdgePot;
elseif(isfield(options,'edge_distance')) % Initialize parameters with edge distance
    prob_bij = initParamDist(options.edge_distance, edge_pairs, samples);
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

ll = zeros(options.max_ite,1);
tiny = 1e-20;
for ite=1:options.max_ite
    
    entropy = 0;
    % E step: compute statistics of hidden variables for each sample using
    % the current parameter setting.
    new_prob_bij = sparse(2*Ntotal,2*Ntotal);
    for n=1:Nsamples
        for node = 1:Nobserved
            nodePot(node,:) = [covariates*w' 1-covariates*w'];
        end
        nodePot(1:Nobserved,:) = repmat([1 0],Nobserved,1);  
        indSample2 = (samples(:,n)==2);
        nodePot(indSample2,1) = 0;
        nodePot(indSample2,2) = 1;    

        [node_marginals, edge_marginals] = sumProductBin(adjmat,nodePot,edgePot,tree_msg_order);
        
        new_prob_bij = new_prob_bij + diag(reshape(node_marginals',2*Ntotal,1));

        for e=1:size(edge_pairs,1)
            u = edge_pairs(e,1);
            v = edge_pairs(e,2);
            emar = edge_marginals(2*u-1:2*u,2*v-1:2*v);
            new_prob_bij(2*u-1:2*u,2*v-1:2*v) = new_prob_bij(2*u-1:2*u,2*v-1:2*v) + emar;
            
            if(all(ismember([u,v],hnodes))) % Compute the entropy
                entropy = entropy - emar(:)'*log(emar(:)+tiny);
                %fprintf('Edge entropy between %d and %d is %f\n',u,v,-emar(:)'*log(emar(:)+tiny));
            end            
        end   
        node_entropy = sum(node_marginals(hnodes,:).*log(node_marginals(hnodes,:)+tiny),2);
        entropy = entropy + sum((degree_hnodes-1).*node_entropy);
    end

    new_prob_bij = (new_prob_bij + new_prob_bij' - diag(diag(new_prob_bij)))/Nsamples;
    
    % M step: Set the parameters using the complete statistics
    [nodePot, edgePot] = marToPotBin(new_prob_bij,tree_msg_order);
    
    % Compute the log-likelihood.
    ll_term1 = computeAvgLLBin(nodePot(root,:),edgePot,new_prob_bij,edge_pairs);
    ll(ite) = Nsamples*ll_term1 + entropy;
    fprintf('Iteration %d, log-likelihood %f (first term %f) \n',ite,ll(ite),Nsamples*ll_term1);
    
    if(ite > 1 && ll(ite) - ll(ite-1) < 1)
        ll = ll(1:ite);
        break
    end
end

% Phi = mnrfit(covariates,nodePot);




