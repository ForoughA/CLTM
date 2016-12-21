function [Params,dParams,llEM,EcllEM] = EM_UGMboth2(adjmat,samples,covariates,depCovariates,options)
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
ndepCov = size(depCovariates,3);
nTot = size(adjmat,1);
nHidden = nTot - nObs;
if ~isfield(options,'nStates')
    nStates = 2;
else
    nStates = options.nStates;
end
edgeStruct = UGM_makeEdgeStruct(adjmat,nStates);
nEdges = edgeStruct.nEdges;
edge_pairs = edgeStruct.edgeEnds;

%initializing the parameters
if (isfield(options,'initdParams') && isfield(options,'initParams'))
    dParams = options.initdParams;
    Params = options.initParams;
else
    nodePot = zeros(nTot,nStates);
    edgePot = ones(nStates,nStates,nEdges);%should be initialized randomly
%     edgePot(2,2,:) = exprnd(5,[1,1,nEdges]);
%     edgePot(2,2,:) = 3*ones([1,1,nEdges]);
    Params = randn(nCov,1);%should be initialized randomly
    dParams = exprnd(1,[ndepCov,1]);
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

Ecll = zeros(options.gradIter,options.maxIter);
ll = zeros(options.gradIter,options.maxIter);
EcllEM = zeros(options.maxIter,1);
llEM = zeros(options.maxIter,1);

for EMiter=1:options.maxIter
    node_marginal = cell(nSamples,1);
    edge_marginal = cell(nSamples,1);
    %E-step: inference for E[H_i] and E[H_i.H_j]
    tic;
    for t=1:nSamples
                
        nodePot(:,2) = exp(covariates(:,:,t)*Params);
        nodePot(:,1) = ones(nTot,1);
        edgePot = ones(2,2,edgeStruct.nEdges);
        for e=1:nEdges
            par = edge_pairs(e,1);
            child = edge_pairs(e,2);
            edgePot(2,2,e) = exp(reshape(depCovariates(par,child,:,t),[1,ndepCov])*dParams);
        end
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
    if EMiter>1 && abs(EcllEM(EMiter)-EcllEM(EMiter-1))<5
        llEM = llEM(1:EMiter);
        EcllEM = EcllEM(1:EMiter);
        break
    end
    tEstep = toc;
    fprintf('time taken during the E-step is: %f\n',tEstep);
    
    tic;
    %M-step: maximizing the expected complete data log likelihood using
    %gradient descent (change to SGD once u are sure of the convergence
    %properties and correctness of the EM implementation)
    for iter = 1:options.gradIter
        Gk = zeros(nCov,1)';%graident update for the node potential
        Gl = zeros(ndepCov,1);%graident update for the vv edge potential
        stepi = options.stepSize;%/iter;%learning rate = 1/t
        for t=1:nSamples
            
            nodePot_sgd = zeros(nTot,2);
            nodePot_sgd(:,2) = exp(covariates(:,:,t)*Params);
            nodePot_sgd(:,1) = ones(nTot,1);
            edgePot = ones(2,2,nEdges);
            for e=1:nEdges
                par = edge_pairs(e,1);
                child = edge_pairs(e,2);
                edgePot(2,2,e) = exp(reshape(depCovariates(par,child,:,t),[1,ndepCov])*dParams);
            end

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
                for ne = 1:ndepCov
                    if par<=nObs && child<=nObs
                        Gl(ne) = Gl(ne) + ((samples(par,t)-1)*(samples(child,t)-1)- e_mar(2,2,e))*depCovariates(par,child,ne,t);
                    elseif par>nObs && child<=nObs
                        Gl(ne) = Gl(ne) + ((samples(child,t)-1)*(node_marginal{t}(par,:)*[0 1]') - e_mar(2,2,e))*depCovariates(par,child,ne,t);
                    elseif par<=nObs && child>nObs
                        Gl(ne) = Gl(ne) + ((samples(par,t)-1)*(node_marginal{t}(child,:)*[0 1]') - e_mar(2,2,e))*depCovariates(par,child,ne,t);
                    elseif par>nObs && child>nObs
                        Gl(ne) = Gl(ne) + (edge_marginal{t}(2,2,e) - e_mar(2,2,e))*depCovariates(par,child,ne,t);
                    end
                end
            end
        end
        
        if iter>1 && abs(Ecll(iter,EMiter)-Ecll(iter-1,EMiter))<options.epsilon
            break
        end
        
        Params = Params + stepi * Gk';
        dParams = dParams + stepi * Gl;
    end
    tMstep = toc;
    fprintf('time taken during the M-step is: %f\n',tMstep);
    
end