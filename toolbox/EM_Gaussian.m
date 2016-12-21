function [Params,dParams,llEM,EcllEM] = EM_Gaussian(adjmat,samples,covariates,depCovariates,options)
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

%I doubt the initialization
if ~isfield(options,'GBPmaxround')
    options.maxround = 20;
end

if ~isfield(options,'GBPepsilon')
    options.GBPepsilon = 0.0001;
end

[nObs,nSamples] = size(samples);
nCov = size(covariates,2);
ndepCov = size(depCovariates,3);
nTot = size(adjmat,1);
nHidden = nTot - nObs;
nEdges = nTote - 1;
adjmat = adjmat + eye(nTot);%accounting for potential function v^2
[e1,e2,val] = find(adjmat);
ep = [e1,e2];
edgePairs = ep(ep(:,1)>=ep(:,2), :);
%depCovariates should be given in the form of edgePairs

%initializing the parameters
if (isfield(options,'initdParams') && isfield(options,'initParams'))
    dParams = options.initdParams;
    Params = options.initParams;
else
    nodePot = zeros(nTot,1);
    edgePot = ones(nEdges+nTot,1);%should be initialized randomly
%     edgePot(2,2,:) = exprnd(5,[1,1,nEdges]);
%     edgePot(2,2,:) = 3*ones([1,1,nEdges]);
    Params = randn(nCov,1);%should be initialized randomly
%     dParams = randn(nEdges+nTot);
    dParams = randn(ndepCov,1);
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
    %E-step: inference for E[H_i], E[H_i^2] and E[H_i.H_j]
    for t=1:nSamples
                
        edgePot = zeros(nTot);
        nodePot = (covariates(:,:,t)*Params);
        edgePotTmp = depCovariates(:,:,t)*dParams;
        %I was here
        for e = 1:nEdges
            e1 = edgePairs(e,1);
            e2 = edgePairs(e,2);
            edgePot(e1,e2) = edgePotTmp(e);
        end
        edgePot = edgePot + tril(edgePot,-1)';
        [node_marginal{t},edge_marginal{t},r,C] =...
            GBP(edgePot,nodePot,options.GBPmaxround,options.GBPepsilon);
        %inference (it differs for each data point). This one is
        %conditioned on the observerd data to be used by the E-step
        
        %inference for computing the likelihood of the data under the
        %model. We could not use the conditional inference for this one.
        
        %TODO: add the likelihood computation
%         [node_marginal_LL,edge_marginal_LL,logZ_LL] = UGM_Infer_Tree(nodePot,edgePot,edgeStruct);
%         [llt,Ecllt] = computeCondLikelihood_UGM(nodePot,edgePot,node_marginal_LL,edge_marginal_LL,edgeStruct,samples(:,t),logZ_LL);
%         llEM(EMiter) = llEM(EMiter) + llt;
%         EcllEM(EMiter) = EcllEM(EMiter) + Ecllt;
        
    end
    fprintf('EM iteration: %d, Ecll: %f \n',EMiter,EcllEM(EMiter))
    if EMiter>1 && abs(EcllEM(EMiter)-EcllEM(EMiter-1))<5
        llEM = llEM(1:EMiter);
        EcllEM = EcllEM(1:EMiter);
        break
    end

    %M-step: maximizing the expected complete data log likelihood using
    %gradient descent (change to SGD once u are sure of the convergence
    %properties and correctness of the EM implementation)
    for iter = 1:options.gradIter
        Gk = zeros(nCov,1)';%graident update for the node potential
        Gl = zeros(ndepCov,1);%graident update for the vv edge potential
        stepi = options.stepSize;%/iter;%learning rate = 1/t
        for t=1:nSamples
            
            nodePot_sgd = (covariates(:,:,t)*Params);
            edgePotTmp_sgd = depCovariates(:,:,t)*dParams;
            for e=1:nEdges
                par = edgePairs(e,1);
                child = edgePairs(e,2);
                edgePot_sgd(par,child) = edgePotTmp_sgd(e);
            end
            edgePot_sgd = edgePot_sgd + tril(edgePot_sgd,-1)';

            [n_mar,e_mar] = ...
                GBP(nodePot_sgd, edgePot_sgd, options.GBPmaxround, options.GBPepsilon);
            
%             [lli,Eclli] = computeCondLikelihood_UGM(nodePot_sgd,edgePot_sgd,n_mar,e_mar,edgeStruct,samples(:,t),lz);
%             ll(iter,EMiter) = ll(iter,EMiter) + lli;
%             Ecll(iter,EMiter) = Ecll(iter,EMiter) + Eclli;
            
            node_grad1 = (samples(:,t))'*covariates(1:nObs,:,t) +...
                (node_marginal{t}(nObs+1:end))'*covariates(nObs+1:end,:,t);
            node_grad2 = (n_mar)'*covariates(:,:,t);
            Gk = Gk + node_grad1 - node_grad2;
            
            for e=1:nEdges
                par = edgePairs(e,1);
                child = edgePairs(e,2);
                for ne = 1:ndepCov
                    if par<=nObs && child<=nObs
                        Gl(ne) = Gl(ne) + ...
                            (-0.5)*((samples(par,t))*(samples(child,t)) -...
                            e_mar(e))*depCovariates(e,ne,t);
                    elseif par>nObs && child<=nObs
                        Gl(ne) = Gl(ne) + ...
                            (-0.5)*((samples(child,t))*(node_marginal{t}(par)) -...
                            e_mar(e))*depCovariates(e,ne,t);
                    elseif par<=nObs && child>nObs
                        Gl(ne) = Gl(ne) + ...
                            (-0.5)*((samples(par,t))*(node_marginal{t}(child)) -...
                            e_mar(2,2,e))*depCovariates(e,ne,t);
                    elseif par>nObs && child>nObs
                        Gl(ne) = Gl(ne) + ...
                            (-0.5)*(edge_marginal{t}(e) - e_mar(e)) *...
                            depCovariates(e,ne,t);
                    end
                end
            end
        end
        
        %TODO: add likelihood function computation
        if iter>1 && abs(Ecll(iter,EMiter)-Ecll(iter-1,EMiter))<options.epsilon
            break
        end
        
        Params = Params + stepi * Gk';
        dParams = dParams + stepi * Gl;
    end
    
end