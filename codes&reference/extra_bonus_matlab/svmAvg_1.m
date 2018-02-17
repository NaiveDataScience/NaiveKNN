function [model] = svmAvg_1(X,y,lambda,maxIter)

% Add bias variable
[n,d] = size(X);
X = [ones(n,1) X];

% Matlab indexes by columns,
%  so if we are accessing rows it will be faster to use  the traspose
Xt = X';

% Initial values of regression parameters
w = zeros(d+1,1);
w_accumulate = zeros(d+1,1);


% Apply stochastic gradient method
for t = 1:maxIter
    w_mean = w_accumulate/t

    if mod(t-1,n) == 0
        % Plot our progress
        % (turn this off for speed)
        
        
        objValues(1+(t-1)/n) = (1/n)*sum(max(0,1-y.*(X*w_mean))) + (lambda/2)*(w_mean'*w_mean);
        semilogy([0:t/n],objValues);
        pause(.1);
    end
    
    % Pick a random training example
    i = ceil(rand*n);
    
    % Compute sub-gradient
    [f,sg] = hingeLossSubGrad(w_mean,Xt,y,lambda,i);
    
    % Set step size
    % alpha = 1/(lambda*t);
    alpha = 0.5
    % Take stochastic subgradient step
    w_accumulate += w_mean - alpha*(sg + lambda*w_mean);
end

model.w = w_mean;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);
Xhat = [ones(t,1) Xhat];
w = model.w;
yhat = sign(Xhat*w);
end

function [f,sg] = hingeLossSubGrad(w,Xt,y,lambda,i)

[d,n] = size(Xt);

% Function value
wtx = w'*Xt(:,i);
loss = max(0,1-y(i)*wtx);
f = loss;

% Subgradient
if loss > 0
    sg = -y(i)*Xt(:,i);
else
    sg = sparse(d,1);
end
end

