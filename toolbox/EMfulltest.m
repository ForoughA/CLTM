function [nodePot,edgePot,Params,ll] = EMfulltest(adjmat,samples,covariates,options)

% samples = samples - 1;
%covariate vector should be augmented
if ~isfield(options,'max_ite')
    options.max_ite = 20;
end

if ~isfield(options,'root')
    root = 1;
else
    root = options.root;
end

nObs = size(samples,1); %number of observed nodes
nSamples = size(samples,2);% number of samples
nTotal = length(adjmat);
nHidden = nTotal - nObs;% number of hidden nodes
nCov = size(covariates,2);%number of covariates
hnodes = nObs+1:nTotal;%hidden nodes

degree_hnodes = sum(adjmat(nObs+1:end,nObs+1:end),2);
tree_msg_order = treeMsgOrder(adjmat,root);
edge_pairs = tree_msg_order(size(tree_msg_order,1)/2+1:end,:);

% weighted_prob = 0; 
% LL = zeros(options.max_ite,1);
% initParams = zeros(nCov,1);
% initNodePot = ones(nTotal,2);
% initEdgePot = ones(2*nTotal,2*nTotal);

nodePot = zeros(nTotal,2);
edgePot = sparse(2*nTotal,2*nTotal);
if(isfield(options,'initNodePot') && isfield(options,'initEdgePot') && isfield(options,'initParams'))
    nodePot = options.initNodePot;
    edgePot = options.initEdgePot;
    Params = options.initParams;
elseif(isfield(options,'edge_distance')) % Initialize parameters with edge distance
    prob_bij = initParamDist(options.edge_distance, edge_pairs, samples);
    [nodePot, edgePot] = marToPotBin(prob_bij,tree_msg_order);
    Params = randn(nCov,1);
%     Params = zeros(nCov,1);
%     Params=[0.1146
%    -0.7224
%     0.1120
%     1.6613
%    -1.0504
%    -0.3509
%    -1.4191
%     0.7279
%    -1.2061
%    -0.8582];
%     Params = Params / (sum(Params([1:3,11:12]))+max(Params(4:10)));
else
    nodePot(:,1) = rand(nTotal,1);
    nodePot(:,2) = 1-nodePot(:,1);
    for e=1:size(edge_pairs,1)
        u = edge_pairs(e,1);
        v = edge_pairs(e,2);        
        r4 = rand(2);
        edgePot(2*u-1:2*u,2*v-1:2*v) = r4/sum(r4(:));
    end
    edgePot = edgePot + edgePot';
    Params = randn(nCov,1);
%     Params = zeros(nCov,1);
%      Params=[0.1146
%        -0.7224
%         0.1120
%         1.6613
%        -1.0504
%        -0.3509
%        -1.4191
%         0.7279
%        -1.2061
%        -0.8582];
%     Params = Params/(sum(Params([1:3,11:12]))+max(Params(4:10)));
end

% Params = initParams;
% nodePot = initNodePot;
% edgePot = initEdgePot;

ll = zeros(options.max_ite,1);
tiny = 1e-20;
epsilon = 1e-4;
for iter = 1:options.max_ite
    new_prob_bij = zeros(2*nTotal,2*nTotal);
    weighted_prob = zeros(nSamples,nCov);
    entropy = 0;
    %E-Step
    for t = 1:nSamples
        weighted_prob(t,:) = (samples(:,t)-1)' * (covariates(1:nObs,:,t));% samples(:,t) .* sum(covariates(:,:,t),2);
        nodePot(1:nObs,:) = repmat([1 0],nObs,1);    
        indSample2 = (samples(:,t)==2);
        nodePot(indSample2,1) = 0;
        nodePot(indSample2,2) = 1;
        nodePot(nObs+1:nTotal,2) = exp((covariates(nObs+1:nTotal,:,t)) * Params);%/exp(1);
%         nodePot(nObs+1:nTotal,1) = 1 - nodePot(nObs+1:nTotal,2);
%         nodePot(nObs+1:nTotal,1) = mean(nodePot(nObs+1:nTotal,2)) * ones(nHidden,1);
%        nodePot(1:nObs,1) = ones(nObs,1);
%         nodePot(nObs+1:nTotal,:) = nodePot(nObs+1:nTotal,:) ./ repmat(sum(nodePot(nObs+1:nTotal,:),2),1,2);
        nodePot(nObs+1:nTotal,1) = ones(nHidden,1);
        nodePot(nObs+1:nTotal,:) = nodePot(nObs+1:nTotal,:) ./ repmat(sum(nodePot(nObs+1:nTotal,:),2),1,2);
        
        [node_marginals, edge_marginals] = sumProductBin(adjmat,nodePot,edgePot,tree_msg_order);
        
        new_prob_bij = new_prob_bij + diag(reshape(node_marginals',2*nTotal,1));

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
    
    new_prob_bij = (new_prob_bij + new_prob_bij' - diag(diag(new_prob_bij)))/nSamples;
    hiddenNodeStats = reshape(diag(new_prob_bij),2,nTotal)';
    for t=1:nSamples
        weighted_prob(t,:) = weighted_prob(t,:) + (hiddenNodeStats(nObs+1:nTotal,:)*[0 1]')'*(covariates(nObs+1:nTotal,:,t));
    end
    
    %M-Step
    %maximizing edge potentials and hidden variable node potentialas
    [nodePot, edgePot] = marToPotBin(new_prob_bij,tree_msg_order);
    
    %gradient descent for observed variable potentials
    GDiter = 1;%gradient descent iteration
    stepsize = 1e-2;
    done = 0;
    maxSteps = 80;
    while ~done
        stepi = stepsize/GDiter;
        grad = 0;
        for t=1:nSamples
            tmp = compute_gradient(adjmat,edgePot,Params,samples(:,t),(covariates(:,:,t)),weighted_prob(t,:),tree_msg_order);
            grad = grad + tmp;
        end
        Params = Params + stepi * grad;%gradient ascent update
%         Params = Params / (sum(Params([1:3,11:12]))+max(Params(4:10)));
        GDiter = GDiter + 1;
        stp = (stepi*norm(grad,2));
        fprintf('EM iteration: %d, gradient iteration: %d, gradient update: %f \n',iter,GDiter,stp);
%         if isnan(stp)
%             error('for some reason you have a NaN :(')
%         end
        done = ( (stp) < epsilon) || (GDiter >= maxSteps) ;
    end
    
    % Compute the log-likelihood.
    ll_term1 = computeAvgLLBin(nodePot(root,:),edgePot,new_prob_bij,edge_pairs);
    ll(iter) = nSamples*ll_term1 + entropy;
    fprintf('Iteration %d, log-likelihood %f (first term %f) \n',iter,ll(iter),nSamples*ll_term1);
    
    if(iter > 1 && abs(ll(iter) - ll(iter-1)) < 1)
        ll = ll(1:iter);
        break
    end
end

function G = compute_gradient(adj,Phi_ij,params,sample,cov,first_term,msg_order)
        nParam = size(params,1);
        Nobs = size(sample,1);
%         Ntot = length(Phi_i);
%         Nhidd = Ntot - Nobs;
        Ncov = size(cov,2);
        Ntot = size(cov,1);
        G = zeros(nParam,1);
        Phi_i = zeros(Ntot,2);
        
        if Ncov~=nParam
            error('number of covariates and number of parameters should be the same');
        end
        
        Phi_i(:,2) = exp(cov * params);%/exp(1);%watch this
%         Phi_i(:,1) = mean(Phi_i(:,2)) *  ones(Ntot,1);
        Phi_i(:,1) = ones(Ntot,1);
        Phi_i = Phi_i ./ repmat(sum(Phi_i,2),1,2);
%         Phi_i(:,1) = 1 - Phi_i(:,2);
        
        [node_mar, edge_mar] = sumProductBin(adj,Phi_i,Phi_ij,msg_order);
        for ki = 1:nParam
            G(ki) = first_term(ki) - (cov(:,ki))' * (node_mar*[0 1]');
        end
end
        
end
