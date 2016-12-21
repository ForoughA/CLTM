function [sample_distance] = sampleFromTree(R, dist, n)% Assign parameters randomly to a tree and generate N samples.m = size(R,1);switch dist    case 'gaussian'                if(m*n > 1e8)            sample_Sigma = zeros(m,m);            ni = n/1e6;            for i=1:ni                x = R'*randn(m,1e6);                sample_Sigma = sample_Sigma + x*x';            end            sample_Sigma = sample_Sigma/n;        else                x = R'*randn(m,n);            sample_Sigma = x*x'/n;        end        %sample_Sigma = zeros(m,m);        %for ni = 1:n        %    x = R'*randn(m,1);        %    sample_Sigma = sample_Sigma + x*x';        %end        %sample_Sigma = sample_Sigma / n;        D = diag(1./sqrt(diag(sample_Sigma)));        sample_rho_matrix = D*sample_Sigma*D;        sample_distance = -log(abs(sample_rho_matrix));end