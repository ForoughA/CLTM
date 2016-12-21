function [cov_prob_xij,cond_prob_bij] = computeBnCondStatsN(samples,covariates)

Nc = size(covariates,1);% # of nodes
Kn = (size(covariates,2));% # of covariates per user
Mc = size(covariates,3);% # of samples

N = size(samples,1);% # of nodes ==> N = Nc
M = size(samples,2);% # of samples

covariateStates = zeros(Nc,Mc);%matrix containing the covariate states of each node sample
for n = 1:Nc
    for m=1:Mc
%     tmp = covariates (1:end,m);%discarding the augmented all-one column
        covariateStates(n,m) = bin2dec(num2str(covariates(n,:,m)));
    end
end

cov_prob_xij = zeros(2*Nc, 2*Nc, Kn);%probability distribution of covariate pairs
cond_prob_bij = zeros(4*N, 4*N, Kn);%conditional distances
 for k=1:Kn
    for i=1:Nc
        for j=1:i-1
            
            covariate_pairs = 2*covariates(i,:) + covariates(j,:);
            
            ind00 = find(covariate_pairs==3);
            ind01 = find(covariate_pairs==4);
            ind10 = find(covariate_pairs==5);
            ind11 = find(covariate_pairs==6);

            p00 = length(ind00); % P(xi=0,xj=0)
            p01 = length(ind01); % P(xi=0,xj=1)
            p10 = length(ind10); % P(xi=1,xj=0)
            p11 = length(ind11); % P(xi=1,xj=1)

            cov_prob_xij([2*(i-1)+1,2*(i-1)+2],[2*(j-1)+1,2*(j-1)+2],k) = [p00 p01; p10 p11];
            
            sample_pairs00 = 2*samples(i,ind00) + samples(j,ind00);%sample pairs conditioned on the kth covariate pair in state (0,0)
            sample_pairs01 = 2*samples(i,ind01) + samples(j,ind01);%sample pairs conditioned on the kth covariate pair in state (0,1)
            sample_pairs10 = 2*samples(i,ind10) + samples(j,ind10);%sample pairs conditioned on the kth covariate pair in state (1,0)
            sample_pairs11 = 2*samples(i,ind11) + samples(j,ind11);%sample pairs conditioned on the kth covariate pair in state (1,1)
            
            sP00_00 = length(find(sample_pairs00==3));%p[(bi=0,bj=0)|(xi=0,xj=0)]
            sP01_00 = length(find(sample_pairs00==4));%p[(bi=0,bj=1)|(xi=0,xj=0)]
            sP10_00 = length(find(sample_pairs00==5));%p[(bi=1,bj=0)|(xi=0,xj=0)]
            sP11_00 = length(find(sample_pairs00==6));%p[(bi=1,bj=1)|(xi=0,xj=0)]
            
            sP00_01 = length(find(sample_pairs01==3));%p[(bi=0,bj=0)|(xi=0,xj=1)]
            sP01_01 = length(find(sample_pairs01==4));%p[(bi=0,bj=1)|(xi=0,xj=1)]
            sP10_01 = length(find(sample_pairs01==5));%p[(bi=1,bj=0)|(xi=0,xj=1)]
            sP11_01 = length(find(sample_pairs01==6));%p[(bi=1,bj=1)|(xi=0,xj=1)]
            
            sP00_10 = length(find(sample_pairs10==3));%p[(bi=0,bj=0)|(xi=1,xj=0)]
            sP01_10 = length(find(sample_pairs10==4));%p[(bi=0,bj=1)|(xi=1,xj=0)]
            sP10_10 = length(find(sample_pairs10==5));%p[(bi=1,bj=0)|(xi=1,xj=0)]
            sP11_10 = length(find(sample_pairs10==6));%p[(bi=1,bj=1)|(xi=1,xj=0)]
            
            sP00_11 = length(find(sample_pairs11==3));%p[(bi=0,bj=0)|(xi=1,xj=1)]
            sP01_11 = length(find(sample_pairs11==4));%p[(bi=0,bj=1)|(xi=1,xj=1)]
            sP10_11 = length(find(sample_pairs11==5));%p[(bi=1,bj=0)|(xi=1,xj=1)]
            sP11_11 = length(find(sample_pairs11==6));%p[(bi=1,bj=1)|(xi=1,xj=1)]
            
            cond_prob_bij([4*(i-1)+1,4*(i-1)+2],[4*(j-1)+1,4*(j-1)+2],k) = [sP00_00 sP01_00; sP10_00 sP11_00] / p00;
            cond_prob_bij([4*(i-1)+1,4*(i-1)+2],[4*(j-1)+3,4*(j-1)+4],k) = [sP00_01 sP01_01; sP10_01 sP11_01] / p01;
            cond_prob_bij([4*(i-1)+3,4*(i-1)+4],[4*(j-1)+1,4*(j-1)+2],k) = [sP00_10 sP01_10; sP10_10 sP11_10] / p10;
            cond_prob_bij([4*(i-1)+3,4*(i-1)+4],[4*(j-1)+3,4*(j-1)+4],k) = [sP00_11 sP01_11; sP10_11 sP11_11] / p11;

            if((abs(p00+p01+p10+p11) - Mc) > 0)
                fprintf('%d %d\n', i,j);
            end
            if((abs(sP00_00+sP01_00+sP10_00+sP11_00) - p00) > 0)
                fprintf('%d %d\n', i,j);
            end
            if((abs(sP00_01+sP01_01+sP10_01+sP11_01) - p01) > 0)
                fprintf('%d %d\n', i,j);
            end
            if((abs(sP00_10+sP01_10+sP10_10+sP11_10) - p10) > 0)
                fprintf('%d %d\n', i,j);
            end
            if((abs(sP00_11+sP01_11+sP10_11+sP11_11) - p11) > 0)
                fprintf('%d %d\n', i,j);
            end
        end
    end


    cov_prob_xij(:,:,k) = cov_prob_xij(:,:,k) + cov_prob_xij(:,:,k)';
    cond_prob_bij(:,:,k) = cond_prob_bij(:,:,k) + cond_prob_bij(:,:,k)';

    for i=1:Nc

        ind0 = find(covariates(i,k+1,:)==1);
        ind1 = find(covariates(i,k+1,:)==2);
        
        p0 = length(ind0); % P(xi=0)
        p1 = length(ind1); % P(xi=1)

        cov_prob_xij([2*(i-1)+1,2*(i-1)+2],[2*(i-1)+1,2*(i-1)+2],k) = [p0 0; 0 p1];
        
        sP0_0 = length(find(samples(i,ind0)==1));% p(bi=0|xi=0)
        sP1_0 = length(find(samples(i,ind0)==2));% p(bi=1|xi=0)
        
        sP0_1 = length(find(samples(i,ind1)==1));% p(bi=0|xi=1)
        sP1_1 = length(find(samples(i,ind1)==2));% p(bi=1|xi=1)
        
        cond_prob_bij([4*(i-1)+1,4*(i-1)+2],[4*(i-1)+1,4*(i-1)+2],k) = [sP0_0 0; 0 sP1_0] / p0;
        cond_prob_bij([4*(i-1)+3,4*(i-1)+4],[4*(i-1)+3,4*(i-1)+4],k) = [sP0_1 0; 0 sP1_1] / p1;
        
    end
end
cov_prob_xij = cov_prob_xij / Mc;
