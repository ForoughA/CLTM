function [distance,cond_prob] = computeCondProbs(joint_cond_prob_bij,cov_prob_Xij)

M = size(joint_cond_prob_bij,1);
Kn = size(cov_prob_Xij,3);% # of covariate states
N = M/4;% # of nodes

mar_prob = zeros(N,4,Kn);

for k = 1:Kn
    for i = 1:N
        tmp = diag(joint_cond_prob_bij(:,:,k));
        mar_prob(:,:,k) = reshape(tmp,4,N)';
    end
end

% K = size(mar_prob,2);
K = 4;
distance = zeros(N);
cond_prob = zeros(2*N,2*N);
tiny = 1e-20;
for i=1:N
    for j=i+1:N
        joint = zeros(2,2,4);
        for k=1:Kn
            cov_prob = cov_prob_Xij(2*(i-1)+1:2*i,2*(j-1)+1:2*j,k)';
            cov_prob = cov_prob(:);
            joint_prob = joint_cond_prob_bij(K*(i-1)+1:K*i,K*(j-1)+1:K*j,k);
            joint(:,:,1) = joint_prob([1,2],[1,2]);%P(bi,bj|Xi=0,Xj=0)
            joint(:,:,2) = joint_prob([1,2],[3,4]);%P(bi,bj|Xi=0,Xj=1)
            joint(:,:,3) = joint_prob([3,4],[1,2]);%P(bi,bj|Xi=1,Xj=0)
            joint(:,:,4) = joint_prob([3,4],[3,4]);%P(bi,bj|Xi=1,Xj=1)
            ctr = 1;
            for n1 = 1:2
                for n2 = 1:2
                    Jij = det(joint(:,:,ctr));
                    Mii = prod(mar_prob(i,2*(n1-1)+1:2*n1,k));
                    Mjj = prod(mar_prob(j,2*(n2-1)+1:2*n2,k));
                    cond_prob(2*(i-1)+1:2*i,2*(j-1)+1:2*j) = cond_prob(2*(i-1)+1:2*i,2*(j-1)+1:2*j) + joint(:,:,ctr)*cov_prob(ctr);
                    if Jij~=0% && prod(Mii)~=0 && prod(Mjj)~=0
                        distance(i,j) = distance(i,j) - log(abs(Jij))* cov_prob(ctr);%/sqrt(Mii*Mjj+tiny)+tiny) * cov_prob(ctr);
                    end
                    ctr = ctr + 1;
                end
            end
            
            
%             for ctr = 1:4
%                 tmp = det(joint(:,:,ctr));
%                 if tmp~=0
%                     distance(i,j) = distance(i,j) - log(abs(tmp)/sqrt(prod(mar_prob(i,:,k))*prod(mar_prob(j,:,k))+tiny)+tiny) * cov_prob(ctr);
%                 end
%             end
            
            
%             if ~isequal(joint_prob,zeros(4))
%                 distance(i,j) = distance(i,j) - ...
%                     log(abs(det(joint_prob))) * cov_prob_Xij(k);%/sqrt(prod(mar_prob(i,:,k))*prod(mar_prob(j,:,k))+tiny)+tiny) * cov_prob_Xij(k);
%             end
        end
    end
end 
mx = max(max(distance));
for i=1:N
    for j=i+1:N
        if distance(i,j) == 0
            distance(i,j) = 10*mx;
        end
    end
end
distance = distance + distance';