function dist = computeCondDist2D(covar,jointProb)

nVars = size(covar, 1) / 2;
nCov = size(jointProb, 3);
dim = 2;
dist = zeros(nVars, nVars);%distance matrix (correlation)

for i = 1:nVars
    for j = 1:i-1
%         numerator = 0;
%         denom = 0;
        for k = 1:nCov
            numMat = covar([2*(i-1)+1,2*i], [2*(j-1)+1,2*j], :, k);
            denomiMat = covar([2*(i-1)+1,2*i], [2*(i-1)+1,2*i], :, k);
            denomjMat = covar([2*(j-1)+1,2*j], [2*(j-1)+1,2*j], :, k);
            joint = jointProb([2*(i-1)+1,2*i], [2*(j-1)+1,2*j], k);
            joint = joint';
            joint = joint(:);
            
            for ctr = 1:4
                tmpnum = svd(numMat(:,:,ctr));
                ind = (ceil(ctr/2))^2;
                denom = det(denomiMat(:,:,ind))*det(denomjMat(:,:,ind));
                if all(tmpnum)
                    dist(i,j) = dist(i,j) + ...
                        -1*log(abs(prod(tmpnum)/sqrt(denom))) * joint(ctr);
                end
            end
        end
    end
end
dist = dist + dist';
