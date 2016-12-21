function distance = computeCondDistance3(cond_prob_bij,cov_prob_Xk,nodeSpecProb)

M = size(cond_prob_bij,1);
Kn =  size(cond_prob_bij,3);% # of covariate states
N = M/2;% # of nodes

mar_prob = zeros(N,2,Kn);

for k = 1:Kn
    tmp = diag(cond_prob_bij(:,:,k));
    mar_prob(:,:,k) = reshape(tmp,2,N)';
end

K = size(mar_prob,2);
distance = zeros(N);
tiny = 1e-10;
for i=1:N
    for j=i+1:N
        Pcov = cov_prob_Xk(4*(i-1)+1:4*(i-1)+4,4*(j-1)+1:4*(j-1)+4);
        Pcov = Pcov(:);
        for k=1:Kn 
            joint_prob = cond_prob_bij(K*(i-1)+1:K*i,K*(j-1)+1:K*j,k);
            if ~isequal(joint_prob,[0,0;0,0])
%                 distance(i,j) = distance(i,j) - ...
%                     log(abs(det(joint_prob))/sqrt(prod(mar_prob(i,:,k))*prod(mar_prob(j,:,k))+tiny)+tiny) * Pcov(k);
                distance(i,j) = distance(i,j) - (log(abs(det(joint_prob))+tiny) * Pcov(k));
            end
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