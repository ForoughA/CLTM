function [cov_prob_xij,cond_prob_bij,nodeSpecProb] = computeBnCondStats3(samples,covariates)

%This function computes the conditional pairwise probability tables of the
%vertices given the covariates for the beach dataset. The covariates are
%contained in a 95x3x31 array and we have 3 covariates for each of the 95
%vertices in the network observed for 31 days with one missing day replaced
%with the mean of the other days. The first covariate indicates the
%reqularity of each node the second one indicates the temperature and the
%third covariate is the previous state of that vertex

Nc = size(covariates,1);% # of nodes
Kn = size(covariates,2);% # of covariates per user
Mc = size(covariates,3);% # of samples

N = size(samples,1);% # of nodes ==> N = Nc
M = size(samples,2);% # of samples

regularity = covariates(:,1,1);%regularity of the users is fixed in time
covariateStates = zeros(Nc,Mc);%matrix containing the covariate states of each node sample
    %containing temperature and previous state
for n = 1:Nc
    for m=1:Mc
        covariateStates(n,m) = bin2dec(num2str(covariates(n,2:end,m)));
    end
end
numCovStates = max(max(covariateStates));
cov_prob_xij = zeros(4*N,4*N);
cond_prob_bij = zeros(2*Nc,2*Nc,numCovStates);
for i=1:Nc
    for j=1:i-1
        cov_pairs = covariateStates(i,:) + (numCovStates+1)*covariateStates(j,:);
        
        P_Cov = zeros(numCovStates+1);
        ctr = 0;
        for clmn = 1:numCovStates+1
            for row = 1:numCovStates+1
                ind = find(cov_pairs == ctr);
                P_Cov(row,clmn) = length(ind); %Pr(xi=row-1,xj=clmn-1)
                
                sample_pairs = 2*samples(i,ind) + samples(j,ind);
                p00 = length(find(sample_pairs==3)); % P(bi=0,bj=0|xi=row-1,xj=clmn-1)
                p01 = length(find(sample_pairs==4)); % P(bi=0,bj=1|xi=row-1,xj=clmn-1)
                p10 = length(find(sample_pairs==5)); % P(bi=1,bj=0|xi=row-1,xj=clmn-1)
                p11 = length(find(sample_pairs==6)); % P(bi=1,bj=1|xi=row-1,xj=clmn-1)

                if ~isempty(ind)
                    cond_prob_bij([2*(i-1)+1,2*(i-1)+2],[2*(j-1)+1,2*(j-1)+2],ctr+1) = [p00 p01; p10 p11]/length(ind);
                end
                ctr = ctr + 1;
            end
        end
        cov_prob_xij(4*(i-1)+1:4*(i-1)+4,4*(j-1)+1:4*(j-1)+4) = P_Cov;
    end
end
cov_prob_xij = cov_prob_xij + cov_prob_xij';
for ctr = 1:size(cond_prob_bij,3)
    cond_prob_bij(:,:,ctr) = cond_prob_bij(:,:,ctr) + cond_prob_bij(:,:,ctr)';
end

for i=1:Nc
    P_Cov = zeros(numCovStates+1);
    for k=1:numCovStates+1
        ind = find(covariateStates(i,:) == k-1);
        P_Cov(k,k) = length(ind);
        
        P0 = length(find(samples(i,ind))==1); %Pr(bi=0|xi=k-1)
        P1 = length(find(samples(i,ind))==2); %Pr(bi=1|xi=k-1)
        
        if ~isempty(ind)
            cond_prob_bij([2*(i-1)+1,2*(i-1)+2],[2*(i-1)+1,2*(i-1)+2],4*k) = [P0 0; 0 P1]/length(ind);
        end
    end
    cov_prob_xij(4*(i-1)+1:4*(i-1)+4,4*(i-1)+1:4*(i-1)+4) = P_Cov;
end
cov_prob_xij = cov_prob_xij / Mc;

nodeSpecProb = zeros(max(regularity)+1,1);%Probability table for node specific covariates
nodeSpecProb(1) = length(find(regularity==0)) / Nc;%Proportion of the non-regulars
nodeSpecProb(2) = length(find(regularity==1)) / Nc;%1 - nodeSpecProb(1);%Proportion of the regulars

        
%         ind00 = (find(cov_pairs==0));%Pr(xi=0,xj=0)
%         ind10 = (find(cov_pairs==1));%Pr(xi=1,xj=0)
%         ind20 = (find(cov_pairs==2));%Pr(xi=2,xj=0)
%         ind30 = (find(cov_pairs==3));%Pr(xi=3,xj=0)
%         ind01 = (find(cov_pairs==4));%Pr(xi=0,xj=1)
%         ind11 = (find(cov_pairs==5));%Pr(xi=1,xj=1)
%         ind21 = (find(cov_pairs==6));%Pr(xi=2,xj=1)
%         ind31 = (find(cov_pairs==7));%Pr(xi=3,xj=1)
%         ind02 = (find(cov_pairs==8));%Pr(xi=0,xj=2)
%         ind12 = (find(cov_pairs==9));%Pr(xi=1,xj=2)
%         ind22 = (find(cov_pairs==10));%Pr(xi=2,xj=2)
%         ind32 = (find(cov_pairs==11));%Pr(xi=3,xj=2)
%         ind03 = (find(cov_pairs==12));%Pr(xi=0,xj=3)
%         ind13 = (find(cov_pairs==13));%Pr(xi=1,xj=3)
%         ind23 = (find(cov_pairs==14));%Pr(xi=2,xj=3)
%         ind33 = (find(cov_pairs==15));%Pr(xi=3,xj=3)
        
%         p00 = length(ind00);%Pr(xi=0,xj=0)
%         p10 = length(ind10);%Pr(xi=1,xj=0)
%         p20 = length(ind20);%Pr(xi=2,xj=0)
%         p30 = length(ind30);%Pr(xi=3,xj=0)
%         p01 = length(ind01);%Pr(xi=0,xj=1)
%         p11 = length(ind11);%Pr(xi=1,xj=1)
%         p21 = length(ind21);%Pr(xi=2,xj=1)
%         p31 = length(ind31);%Pr(xi=3,xj=1)
%         p02 = length(ind02);%Pr(xi=0,xj=2)
%         p12 = length(ind12);%Pr(xi=1,xj=2)
%         p22 = length(ind22);%Pr(xi=2,xj=2)
%         p32 = length(ind32);%Pr(xi=3,xj=2)
%         p03 = length(ind03);%Pr(xi=0,xj=3)
%         p13 = length(ind13);%Pr(xi=1,xj=3)
%         p23 = length(ind23);%Pr(xi=2,xj=3)
%         p33 = length(ind33);%Pr(xi=3,xj=3)

%         cov_prob_xij([4*(i-1)+1,4*(i-1)+2,4*(i-1)+3,4*(i-1)+4],[4*(j-1)+1,4*(j-1)+2,4*(j-1)+3,4*(j-1)+4]) =...
%                     [p00,p01,p02,p03;
%                     p10,p11,p12,p13;
%                     p20,p21,p22,p23;
%                     p30,p31,p32,p33];
%         
%         switch reg
%             case 0 %irregular-irregular
%                 cov_prob_xij([],[])
%             case 1 %irregular-regular
%                 
%             case 2 %regular-irregular
%                 
%             case 3 %regular-regular
%                 
%         end

%         sample_pairs00 = 2*samples(i,ind00) + samples(j,ind00);
%         sample_pairs10 = 2*samples(i,ind10) + samples(j,ind10);
%         sample_pairs20 = 2*samples(i,ind20) + samples(j,ind20);
%         sample_pairs30 = 2*samples(i,ind30) + samples(j,ind30);
%         sample_pairs01 = 2*samples(i,ind01) + samples(j,ind01);
%         sample_pairs11 = 2*samples(i,ind11) + samples(j,ind11);
%         sample_pairs21 = 2*samples(i,ind21) + samples(j,ind21);
%         sample_pairs31 = 2*samples(i,ind31) + samples(j,ind31);
%         sample_pairs02 = 2*samples(i,ind02) + samples(j,ind02);
%         sample_pairs12 = 2*samples(i,ind12) + samples(j,ind12);
%         sample_pairs22 = 2*samples(i,ind22) + samples(j,ind22);
%         sample_pairs32 = 2*samples(i,ind32) + samples(j,ind32);
%         sample_pairs03 = 2*samples(i,ind03) + samples(j,ind03);
%         sample_pairs13 = 2*samples(i,ind13) + samples(j,ind13);
%         sample_pairs23 = 2*samples(i,ind23) + samples(j,ind23);
%         sample_pairs33 = 2*samples(i,ind33) + samples(j,ind33);

    

