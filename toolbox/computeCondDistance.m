function distance = ...
    computeCondDistance(joint_cond_prob_bij,cov_prob_Xij, mar_prob_bii)

M = size(joint_cond_prob_bij,1);
Kn = size(cov_prob_Xij,3);% # of covariate states
N = M/4;% # of nodes

% K = size(mar_prob,2);
K = 4;
distance = zeros(N);
tiny = 1e-10;
for i=1:N
    for j=i+1:N
        joint = zeros(2,2,4);
        mar = zeros(2,2,4);
        for k=1:Kn
            cov_prob = cov_prob_Xij(2*(i-1)+1:2*i,2*(j-1)+1:2*j,k)';
            cov_prob = cov_prob(:);
            joint_prob = joint_cond_prob_bij(K*(i-1)+1:K*i,K*(j-1)+1:K*j,k);
            mar_prob = mar_prob_bii(K*(i-1)+1:K*i,K*(j-1)+1:K*j,k);
            joint(:,:,1) = joint_prob([1,2],[1,2]);%P(bi,bj|Xi=0,Xj=0)
            mar(:,:,1) = mar_prob([1,2],[1,2]);
            joint(:,:,2) = joint_prob([1,2],[3,4]);%P(bi,bj|Xi=0,Xj=1)
            mar(:,:,2) = mar_prob([1,2],[3,4]);
            joint(:,:,3) = joint_prob([3,4],[1,2]);%P(bi,bj|Xi=1,Xj=0)
            mar(:,:,2) = mar_prob([3,4],[1,2]);
            joint(:,:,4) = joint_prob([3,4],[3,4]);%P(bi,bj|Xi=1,Xj=1)
            mar(:,:,4) = mar_prob([3,4],[3,4]);
            for ctr = 1:4
                Jij = det(joint(:,:,ctr));
                Mii = prod(mar(1,:,ctr));
                Mjj = prod(mar(2,:,ctr));
%                 if Jij~=0 && prod(Mii)~=0 && prod(Mjj)~=0
                distance(i,j) = distance(i,j) - ...
                    log(abs(Jij)/sqrt(Mii*Mjj+tiny)+tiny)*cov_prob(ctr);%\sqrt(Mii*Mjj+tiny)+tiny) * cov_prob(ctr);
%                 end
            end
            
            
        end
    end
end 
distance = distance + distance';