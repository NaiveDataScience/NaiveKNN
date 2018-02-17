function [f, df, y] = logistic_pen(weights, data, targets, hyperparameters)
% Calculate log likelihood and derivatives with respect to weights.
%
% Note: N is the number of examples and 
%       M is the number of features per example.
%
% Inputs:
% 	weights:    (M+1) x 1 vector of weights, where the last element
%               corresponds to bias (intercepts).
% 	data:       N x M data matrix where each row corresponds 
%               to one data point.
%   targets:    N x 1 vector of targets class probabilities.
%   hyperparameters: The hyperparameter structure
%
% Outputs:
%	f:             The scalar error value.
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
%   y:             N x 1 vector of probabilities. This is the output of the classifier.
%

%TODO: finish this function

[N,M] = size(data);

lambda = hyperparameters.weight_regularization

%% Calcuate y
y = logistic_predict(weights, data);

%% Calculate 
N = size(targets,1);
ce = -sum(targets.*log(y)+(1-targets).*log(1-y))/N;
lambda = hyperparameters.weight_regularization;
f = ce + 0.5*lambda*sum(weights.*weights);



%% Calculate df
df = zeros(M+1,1);

%%% partial theta_i
for j = 1:M
	df(j,1) = sum((y-targets).*data(:,j))/N+lambda*weights(j);
end

%%% partial b
df(M+1,1) = sum(y-targets)/N+lambda*weights(M+1);

end

