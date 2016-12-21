function [cov_prob_xij,cond_prob_bij,mar_prob_bii] = compute3aryCondStats(samples,covariates)

Nc = size(covariates,1);% # of nodes
Kn = (size(covariates,2));% # of covariates per user
Mc = size(covariates,3);% # of samples

N = size(samples,1);% # of nodes ==> N = Nc
% M = size(samples,2);% # of samples

cov_prob_xij = zeros(2*Nc, 2*Nc, Kn);%probability distribution of covariate pairs
cond_prob_bij = zeros(6*N, 6*N, Kn);%conditional distances
mar_prob_bii = zeros(6*N, 6*N, Kn);
 for k=1:Kn
    for i=1:Nc
        for j=1:i-1
            
            covariate_pairs = 2*covariates(i,k,:) + covariates(j,k,:);
            
            ind00 = find(covariate_pairs==3);
            ind01 = find(covariate_pairs==4);
            ind10 = find(covariate_pairs==5);
            ind11 = find(covariate_pairs==6);

            p00 = length(ind00); % P(xi=0,xj=0)
            p01 = length(ind01); % P(xi=0,xj=1)
            p10 = length(ind10); % P(xi=1,xj=0)
            p11 = length(ind11); % P(xi=1,xj=1)

            cov_prob_xij([2*(i-1)+1,2*(i-1)+2],[2*(j-1)+1,2*(j-1)+2],k) = [p00 p01; p10 p11];
            
            sample_pairs00 = 3*samples(i,ind00) + samples(j,ind00);%sample pairs conditioned on the kth covariate pair in state (0,0)
            sample_pairs01 = 3*samples(i,ind01) + samples(j,ind01);%sample pairs conditioned on the kth covariate pair in state (0,1)
            sample_pairs10 = 3*samples(i,ind10) + samples(j,ind10);%sample pairs conditioned on the kth covariate pair in state (1,0)
            sample_pairs11 = 3*samples(i,ind11) + samples(j,ind11);%sample pairs conditioned on the kth covariate pair in state (1,1)
            
            sP00_00 = sum(sample_pairs00==4);%p[(bi=0,bj=0)|(xi=0,xj=0)]
            sP01_00 = sum(sample_pairs00==5);%p[(bi=0,bj=1)|(xi=0,xj=0)]
            sP02_00 = sum(sample_pairs00==6);%p[(bi=0,bj=2)|(xi=0,xj=0)]
            sP10_00 = sum(sample_pairs00==7);%p[(bi=1,bj=0)|(xi=0,xj=0)]
            sP11_00 = sum(sample_pairs00==8);%p[(bi=1,bj=1)|(xi=0,xj=0)]
            sP12_00 = sum(sample_pairs00==9);%p[(bi=1,bj=2)|(xi=0,xj=0)]
            sP20_00 = sum(sample_pairs00==10);%p[(bi=2,bj=0)|(xi=0,xj=0)]
            sP21_00 = sum(sample_pairs00==11);%p[(bi=2,bj=1)|(xi=0,xj=0)]
            sP22_00 = sum(sample_pairs00==12);%p[(bi=2,bj=2)|(xi=0,xj=0)]
            
            si_0_00 = sum(samples(i,ind00)==1);
            si_1_00 = sum(samples(i,ind00)==2);
            si_2_00 = sum(samples(i,ind00)==3);
            sj_0_00 = sum(samples(j,ind00)==1);
            sj_1_00 = sum(samples(j,ind00)==2);
            sj_2_00 = sum(samples(j,ind00)==3);
            
            sP00_01 = sum(sample_pairs01==4);%p[(bi=0,bj=0)|(xi=0,xj=1)]
            sP01_01 = sum(sample_pairs01==5);%p[(bi=0,bj=1)|(xi=0,xj=1)]
            sP02_01 = sum(sample_pairs01==6);%p[(bi=0,bj=2)|(xi=0,xj=1)]
            sP10_01 = sum(sample_pairs01==7);%p[(bi=1,bj=0)|(xi=0,xj=1)]
            sP11_01 = sum(sample_pairs01==8);%p[(bi=1,bj=1)|(xi=0,xj=1)]
            sP12_01 = sum(sample_pairs01==9);%p[(bi=1,bj=2)|(xi=0,xj=1)]
            sP20_01 = sum(sample_pairs01==10);%p[(bi=2,bj=0)|(xi=0,xj=1)]
            sP21_01 = sum(sample_pairs01==11);%p[(bi=2,bj=1)|(xi=0,xj=1)]
            sP22_01 = sum(sample_pairs01==12);%p[(bi=2,bj=2)|(xi=0,xj=1)]
            
            si_0_01 = sum(samples(i,ind01)==1);
            si_1_01 = sum(samples(i,ind01)==2);
            si_2_01 = sum(samples(i,ind01)==3);
            sj_0_01 = sum(samples(j,ind01)==1);
            sj_1_01 = sum(samples(j,ind01)==2);
            sj_2_01 = sum(samples(j,ind01)==3);
            
            sP00_10 = sum(sample_pairs10==4);%p[(bi=0,bj=0)|(xi=1,xj=0)]
            sP01_10 = sum(sample_pairs10==5);%p[(bi=0,bj=1)|(xi=1,xj=0)]
            sP02_10 = sum(sample_pairs10==6);%p[(bi=0,bj=2)|(xi=1,xj=0)]
            sP10_10 = sum(sample_pairs10==7);%p[(bi=1,bj=0)|(xi=1,xj=0)]
            sP11_10 = sum(sample_pairs10==8);%p[(bi=1,bj=1)|(xi=1,xj=0)]
            sP12_10 = sum(sample_pairs10==9);%p[(bi=1,bj=2)|(xi=1,xj=0)]
            sP20_10 = sum(sample_pairs10==10);%p[(bi=2,bj=0)|(xi=1,xj=0)]
            sP21_10 = sum(sample_pairs10==11);%p[(bi=2,bj=1)|(xi=1,xj=0)]
            sP22_10 = sum(sample_pairs10==12);%p[(bi=2,bj=2)|(xi=1,xj=0)]
            
            si_0_10 = sum(samples(i,ind10)==1);
            si_1_10 = sum(samples(i,ind10)==2);
            si_2_10 = sum(samples(i,ind10)==3);
            sj_0_10 = sum(samples(j,ind10)==1);
            sj_1_10 = sum(samples(j,ind10)==2);
            sj_2_10 = sum(samples(j,ind10)==3);
            
            
            sP00_11 = sum(sample_pairs11==4);%p[(bi=0,bj=0)|(xi=1,xj=1)]
            sP01_11 = sum(sample_pairs11==5);%p[(bi=0,bj=1)|(xi=1,xj=1)]
            sP02_11 = sum(sample_pairs11==6);%p[(bi=0,bj=2)|(xi=1,xj=1)]
            sP10_11 = sum(sample_pairs11==7);%p[(bi=1,bj=0)|(xi=1,xj=1)]
            sP11_11 = sum(sample_pairs11==8);%p[(bi=1,bj=1)|(xi=1,xj=1)]
            sP12_11 = sum(sample_pairs11==9);%p[(bi=1,bj=2)|(xi=1,xj=1)]
            sP20_11 = sum(sample_pairs11==10);%p[(bi=2,bj=0)|(xi=1,xj=1)]
            sP21_11 = sum(sample_pairs11==11);%p[(bi=2,bj=1)|(xi=1,xj=1)]
            sP22_11 = sum(sample_pairs11==12);%p[(bi=2,bj=2)|(xi=1,xj=1)]
            
            si_0_11 = sum(samples(i,ind11)==1);
            si_1_11 = sum(samples(i,ind11)==2);
            si_2_11 = sum(samples(i,ind11)==3);
            sj_0_11 = sum(samples(j,ind11)==1);
            sj_1_11 = sum(samples(j,ind11)==2);
            sj_2_11 = sum(samples(j,ind11)==3);
            
            if p00~=0
                cond_prob_bij(6*(i-1)+1:6*(i-1)+3,6*(j-1)+1:6*(j-1)+3,k) = [sP00_00 sP01_00 sP02_00; sP10_00 sP11_00 sP12_00; sP20_00 sP21_00 sP22_00] / p00;
                mar_prob_bii(6*(i-1)+1:6*(i-1)+3,6*(j-1)+1:6*(j-1)+3,k)  = [si_0_00 si_1_00 si_2_00; sj_0_00 sj_1_00 sj_2_00; 0 0 0] / p00;
            end
            if p01~=0
                cond_prob_bij(6*(i-1)+1:6*(i-1)+3,6*(j-1)+4:6*(j-1)+6,k) = [sP00_01 sP01_01 sP02_01; sP10_01 sP11_01 sP12_01; sP20_01 sP21_01 sP22_01] / p01;
                mar_prob_bii(6*(i-1)+1:6*(i-1)+3,6*(j-1)+4:6*(j-1)+6,k)  = [si_0_01 si_1_01 si_2_01; sj_0_01 sj_1_01 sj_2_01; 0 0 0] / p01;
            end
            if p10~=0
                cond_prob_bij(6*(i-1)+4:6*(i-1)+6,6*(j-1)+1:6*(j-1)+3,k) = [sP00_10 sP01_10 sP02_10; sP10_10 sP11_10 sP12_10; sP20_10 sP21_10 sP22_10] / p10;
                mar_prob_bii(6*(i-1)+4:6*(i-1)+6,6*(j-1)+1:6*(j-1)+3,k)  = [si_0_10 si_1_10 si_2_10; sj_0_10 sj_1_10 sj_2_10; 0 0 0] / p10;
            end
            if p11~=0
                cond_prob_bij(6*(i-1)+4:6*(i-1)+6,6*(j-1)+4:6*(j-1)+6,k) = [sP00_11 sP01_11 sP02_11; sP10_11 sP11_11 sP12_11; sP20_11 sP21_11 sP22_11] / p11;
                mar_prob_bii(6*(i-1)+4:6*(i-1)+6,6*(j-1)+4:6*(j-1)+6,k)  = [si_0_11 si_1_11 si_2_11; sj_0_11 sj_1_11 sj_2_11; 0 0 0] / p11;
            end

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
    mar_prob_bii(:,:,k) = mar_prob_bii(:,:,k) + mar_prob_bii(:,:,k)';

end
cov_prob_xij = cov_prob_xij / Mc;