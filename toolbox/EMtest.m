function [nodePot,edgePot,Params,ll] = EMtest(adjmat,samples,covariates,options)

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
    Params = rand(nCov,1);
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
    Params = rand(nCov,1);
end

% Params = initParams;
% nodePot = initNodePot;
% edgePot = initEdgePot;

ll = zeros(options.max_ite,1);
tiny = 1e-20;
epsilon = 1e-2;
for iter = 1:options.max_ite
    new_prob_bij = zeros(2*nTotal,2*nTotal);
    weighted_prob = zeros(nSamples,nCov);
    entropy = 0;
    %E-Step
    for t = 1:nSamples
        weighted_prob(t,:) = (samples(:,t))' * (covariates(:,:,t)+1);% samples(:,t) .* sum(covariates(:,:,t),2);
        nodePot(1:nObs,2) = exp((covariates(:,:,t)+1) * Params);
        nodePot(1:nObs,1) = mean(nodePot(1:nObs,2)) * ones(nObs,1);
%        nodePot(1:nObs,1) = ones(nObs,1);
        nodePot(1:nObs,:) = nodePot(1:nObs,:) ./ repmat(sum(nodePot(1:nObs,:),2),1,2);
        
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
    
    %M-Step
    %maximizing edge potentials and hidden variable node potentialas
    [nodePot, edgePot] = marToPotBin(new_prob_bij,tree_msg_order);
    
    %gradient descent for observed variable potentials
    GDiter = 1;%gradient descent iteration
    stepsize = 1e-3;
    done = 0;
    maxSteps = 30;
    while ~done
        stepi = stepsize/GDiter;
        grad = 0;
        for t=1:nSamples
            tmp = compute_gradient(adjmat,nodePot,edgePot,Params,samples(:,t),(covariates(:,:,t)+1),weighted_prob(t,:),tree_msg_order);
            grad = grad + tmp;
        end
        Params = Params - stepi * grad;
        GDiter = GDiter + 1;
        stp = (stepi*norm(grad,2));
        fprintf('EM iteration: %d, gradient iteration: %d, gradient update: %f \n',iter,GDiter,stp);
%         if isnan(stp)
%             error('for some reason you have a NaN :(')
%         end
        done = ( stp < epsilon) || (GDiter >= maxSteps) ;
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

function G = compute_gradient(adj,Phi_i,Phi_ij,params,sample,cov,first_term,msg_order)
        nParam = size(params,1);
        Nobs = size(sample,1);
%         Ntot = length(Phi_i);
%         Nhidd = Ntot - Nobs;
        Ncov = size(cov,2);
        G = zeros(nParam,1);
        
        if Ncov~=nParam
            error('number of covariates and number of parameters should be the same');
        end
        
        Phi_i(1:Nobs,2) = exp(cov * params);%watch this
        Phi_i(1:Nobs,1) = mean(Phi_i(1:Nobs,2)) *  ones(Nobs,1);
        Phi_i(1:Nobs,:) = Phi_i(1:Nobs,:) ./ repmat(sum(Phi_i(1:Nobs,:),2),1,2);
        
        [node_mar, edge_mar] = sumProductBin(adj,Phi_i,Phi_ij,msg_order);
        for ki = 1:nParam
            G(ki) = first_term(ki) - cov(:,ki)' * (node_mar(1:Nobs,:)*[1 2]');
        end
end
        
end


    
        %partition function evaluation:
%         for k=1:nParam
%             pair_sum = 0;
%             for edge=1:size(edge_pairs,1)
%                 i = edge_pairs(edge,1); %parent
%                 j = edge_pairs(edge,2); %child
%                 pot = Phi_ij(2*i-1:2*i,2*j-1:2*j);
%                 pair_sum = pair_sum + pot .* [1 2;2 4];
%             end
%             hidd_sum = sum(Phi_i(Nobs+1:end,:).*repmat([1 2],Nhidd,1),1);
%             obs_sum = 0;
%             for ki=1:nParam
%                 obs_sum = obs_sum + params(ki)*sum(cov(:,ki).*repmat([1 2],Nobs,1));
%             end
% 
%             A = log(sum(exp(sum()+sum())));
%             G = zeros(nParam,1);
% 
%             G = first_term + 
%         end

    
