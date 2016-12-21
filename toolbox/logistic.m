function Y = logistic(X,wght)
    if size(X,3)>1
        m=size(X,1);
        n=size(X,2);
        X = reshape(X,[m*n,size(X,3)]);
        Y = 1./(1+exp(-X*wght'));
        Y=reshape(Y,[m,n]);
        
    else
        f = X * wght';
        Y = 1./(1+exp(-f));
    end
end
