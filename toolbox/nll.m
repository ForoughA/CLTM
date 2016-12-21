function L = nll(X,Y,Ck)
nSamples = size(Y,2);
nObs = size(Y,1);
L = 0;
ctr = 0;
tiny = 1e-10;
for t = 1:nSamples
    for i = 1:nObs
        x = X(i,:,t);
        y = Y(i,t);
        sigX = logistic(x,Ck);
        L = L - y*log(sigX + tiny) - (1-y)*log(1-sigX + tiny);
        if isnan(L)
            error('L is NaN');
        end
        ctr = ctr + 1;
    end
end
end
% L = L/ctr;
