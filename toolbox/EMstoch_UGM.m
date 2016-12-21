function [Params,edgePot,llEM,EcllEM] = EMstoch_UGM(adjmat,samples,covariates,options)
% [nodePot,edgePot,Params,ll] = EMfull(adjmat,samples,covariates,options);
% [nodePot,edgePot,Params,ll,Ecll,llFinal,EcllFinal] = EMfullStoch(adjmat,samples,covariates,options);

%---------------------------------------------------------------------
%Note:
%nodePot and edgePot are the factor graph potentials and phi_i and phi_ij
%are the exponential family potentials for which we have
%nodePot(i,:)=[1,exp(phi_i] and edgePot(:,:,e)=[1,1;1,exp(phi_ij)]
%---------------------------------------------------------------------

if ~isfield(options,'maxIter')
    options.maxIter = 20;
end

if ~isfield(options,'gradIter')
    options.gradIter = 80;
end

if ~isfield(options,'stepSize')
    options.stepSize = 1e-2;
end

if ~isfield(options,'epsilon')
    options.epsilon = 0.5;
end


[nObs,nSamples] = size(samples);
nCov = size(covariates,2);
nTot = size(adjmat,1);
nHidden = nTot - nObs;
nStates = 2;
edgeStruct = UGM_makeEdgeStruct(adjmat,nStates);
nEdges = edgeStruct.nEdges;
edge_pairs = edgeStruct.edgeEnds;

%initializing the parameters
if (isfield(options,'initEdgePot') && isfield(options,'initParams'))
    edgePot = options.initEdgePot;
    Params = options.initParams;
else
    nodePot = zeros(nTot,nStates);
    edgePot = ones(nStates,nStates,nEdges);%should be initialized randomly
    edgePot(2,2,:) = exprnd(10,[1,1,nEdges]);
    Params = randn(nCov,1);%should be initialized randomly
end

if size(edgePot,3) < 2
    error('There might be some compatibility issue with UGM')
end

% Params = [-1.2424
%     0.4302
%    -0.0421
%     1.8006
%     0.5298
%     0.9855
%     0.4883
%    -0.4431
%     0.7171
%     0.7922];
% 
% edgePot(:,:,1) = [1,1;1,0.2621];
% 
% edgePot(:,:,2) = [1,1;1,1.1572];
% 
% edgePot(:,:,3) = [1,1;1,0.4794];

% load('Data/Ck')
% figure;plot(Ck,'r');hold on;
% plot(Params,'b');hold on

%Initializing with the optimal parameters
% load('Data/Ck');
% Params = Ck;
% load('Data/edgePot')

options.stepSize = 1e-2;
options.gradIter = 80;
Ecll = zeros(options.gradIter,options.maxIter);
ll = zeros(options.gradIter,options.maxIter);
EcllEM = zeros(options.maxIter,1);
llEM = zeros(options.maxIter,1);
options.epsilon = 1e-3;

for EMiter=1:options.maxIter
    node_marginal = cell(nSamples,1);
    edge_marginal = cell(nSamples,1);
    %E-step: inference for E[H_i] and E[H_i.H_j]
    for t=1:nSamples
        
%         nodePot(1:nObs,:) = repmat([1 0],nObs,1);    
%         indSample2 = (samples(:,t)==2);
%         nodePot(indSample2,1) = 0;
%         nodePot(indSample2,2) = 1;
%         nodePot(nObs+1:nTot,2) = exp(covariates(nObs+1:nTot,:,t) * Params);
%         nodePot(nObs+1:nTot,1) = ones(nHidden,1);
        
        nodePot(:,2) = exp(covariates(:,:,t)*Params);
        nodePot(:,1) = ones(nTot,1);
        clamped = zeros(nTot,1);
        clamped(1:nObs) = samples(:,t);
        %inference (it differs for each data point). This one is
        %conditioned on the observerd data to be used by the E-step
        [node_marginal{t},edge_marginal{t}] = UGM_Infer_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Infer_Tree);
        
        %inference for computing the likelihood of the data under the
        %model. We could not use the conditional inference for this one.
        [node_marginal_LL,edge_marginal_LL,logZ_LL] = UGM_Infer_Tree(nodePot,edgePot,edgeStruct);
        [llt,Ecllt] = computeCondLikelihood_UGM(nodePot,edgePot,node_marginal_LL,edge_marginal_LL,edgeStruct,samples(:,t),logZ_LL);
        llEM(EMiter) = llEM(EMiter) + llt;
        EcllEM(EMiter) = EcllEM(EMiter) + Ecllt;
        
    end
    fprintf('EM iteration: %d, Ecll: %f \n',EMiter,EcllEM(EMiter))
    if EMiter>1 && abs(EcllEM(EMiter)-EcllEM(EMiter-1))<options.epsilon
        llEM = llEM(1:EMiter);
        EcllEM = EcllEM(1:EMiter);
        break
    end

    %M-step: maximizing the expected complete data log likelihood using
    %gradient descent (change to SGD once u are sure of the convergence
    %properties and correctness of the EM implementation)
    for iter=1:options.gradIter
        Gk = zeros(nCov,1)';%graident update for the node potential
        Gvv = zeros(nEdges,1);%graident update for the vv edge potential
        Gvh = zeros(nEdges,1);%gradient update for the vh egde potential
        Ghh = zeros(nEdges,1);%gradient update for the hh edge potential
        stepi = options.stepSize/iter;%learning rate = 1/t
        Params_old = Params;
        edgePot_old = edgePot;
        for t=randperm(nSamples)

            nodePot_sgd = zeros(nTot,2);
            nodePot_sgd(:,2) = exp(covariates(:,:,t)*Params);
            nodePot_sgd(:,1) = ones(nTot,1);

            [n_mar,e_mar,lz] = UGM_Infer_Tree(nodePot_sgd,edgePot,edgeStruct);

            [lli,Eclli] = computeCondLikelihood_UGM(nodePot_sgd,edgePot,n_mar,e_mar,edgeStruct,samples(:,t),lz);
            ll(iter,EMiter) = ll(iter,EMiter) + lli;
            Ecll(iter,EMiter) = Ecll(iter,EMiter) + Eclli;

            node_grad1 = (samples(:,t)-1)'*covariates(1:nObs,:,t) + (node_marginal{t}(nObs+1:end,:)*[0 1]')'*covariates(nObs+1:end,:,t);
            node_grad2 = (n_mar*[0 1]')'*covariates(:,:,t);
            Gk = Gk + node_grad1 - node_grad2;
            for e=1:nEdges
                par = edge_pairs(e,1);
                child = edge_pairs(e,2);
                if par<=nObs && child<=nObs
                    Gvv(e) = Gvv(e) + (samples(par,t)-1)*(samples(child,t)-1) - e_mar(2,2,e);
                elseif par>nObs && child<=nObs
                    Gvh(e) = Gvh(e) + (samples(child,t)-1)*(node_marginal{t}(par,:)*[0 1]') - e_mar(2,2,e);
                elseif par<=nObs && child>nObs
                    Gvh(e) = Gvh(e) + (samples(par,t)-1)*(node_marginal{t}(child,:)*[0 1]') - e_mar(2,2,e);
                elseif par>nObs && child>nObs
                    Ghh(e) = Ghh(e) + edge_marginal{t}(2,2,e) - e_mar(2,2,e);
                end
            end
            Params = Params + stepi * Gk';
            phi_ij = reshape(log(edgePot(2,2,:)),[nEdges,1]);
            for e=1:nEdges
                par = edge_pairs(e,1);
                child = edge_pairs(e,2);
                if par<=nObs && child<=nObs
                    phi_ij(e) = phi_ij(e) + stepi * Gvv(e);
                elseif (par>nObs && child<=nObs) || (par<=nObs && child>nObs)
                    phi_ij(e) = phi_ij(e) + stepi * Gvh(e);
                elseif par>nObs && child>nObs
                    phi_ij(e) = phi_ij(e) + stepi * Ghh(e);
                end
            end
            edgePot(2,2,:) = reshape(exp(phi_ij),[1,1,nEdges]);
        end

        if norm(Params-Params_old)<1e-3 && norm(reshape(edgePot_old(2,2,:),[nEdges,1]) - reshape(edgePot(2,2,:),[nEdges,1]))<1e-3
            break
        end

        fprintf('GD iteration: %d, Ecll: %f \n',iter,Ecll(iter,EMiter))
    end
    plot(Params,'b');hold on;
    
   
end