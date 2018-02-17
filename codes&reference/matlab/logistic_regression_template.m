%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train_small;
load mnist_train;

load mnist_valid;

%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate = 0.4
% Weight regularization parameter
hyperparameters.weight_regularization = 0.5
% Number of iterations
hyperparameters.num_iterations = 200
% Logistics regression weights
% TODO: Set random weights.
weights = rand(size(train_inputs,2)+1,1)/100;


%% Verify that your logistic function produces the right gradient, diff should be very close to 0
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on
% that data.
nexamples = 20;
ndimensions = 10;
diff = checkgrad('logistic', ...
	             randn((ndimensions + 1), 1), ...   % weights
                 0.001,...                          % perturbation
                 randn(nexamples, ndimensions), ... % data        
                 rand(nexamples, 1), ...            % targets
                 hyperparameters)                   % other hyperparameters

N = size(train_inputs,1);
%% Begin learning with gradient descent.


cross_entropy_trains = zeros(1, hyperparameters.num_iterations)
cross_entropy_valids = zeros(1, hyperparameters.num_iterations)

for t = 1:hyperparameters.num_iterations

    %% TODO: You will need to modify this loop to create plots etc.
    
    

    % Find the negative log likelihood and derivative w.r.t. weights.
    [f, df, predictions] = logistic(weights, ...
                                    train_inputs, ...
                                    train_targets, ...
                                    hyperparameters);

    [cross_entropy_train, frac_correct_train] = evaluate(train_targets, predictions)
    cross_entropy_trains(t) = cross_entropy_train

    % Find the fraction of correctly classified validation examples.
    [temp, temp2, frac_correct_valid] = logistic(weights, ...
                                                 valid_inputs, ...
                                                 valid_targets, ...
                                                 hyperparameters);

    if isnan(f) || isinf(f)
        error('nan/inf error');
    end

    %% Update parameters.
    
    weights = weights - hyperparameters.learning_rate .* df / N;

    predictions_valid = logistic_predict(weights, valid_inputs);
    [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);
    cross_entropy_valids(t) = cross_entropy_valid

    %% Print some stats.
    fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
            t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);

end
plot(1:hyperparameters.num_iterations, cross_entropy_valids, '-;valid;', 1:hyperparameters.num_iterations, cross_entropy_trains, '-;train;')



cross_entropy_trains = zeros(1, hyperparameters.num_iterations);
cross_entropy_valids = zeros(1, hyperparameters.num_iterations);

for t = 1:hyperparameters.num_iterations

    %% TODO: You will need to modify this loop to create plots etc.
    
    

    % Find the negative log likelihood and derivative w.r.t. weights.
    [f, df, predictions] = logistic(weights, ...
                                    train_inputs, ...
                                    train_targets, ...
                                    hyperparameters);

    [cross_entropy_train, frac_correct_train] = evaluate(train_targets, predictions)
    cross_entropy_trains(t) = cross_entropy_train

    % Find the fraction of correctly classified validation examples.
    [temp, temp2, frac_correct_valid] = logistic(weights, ...
                                                 valid_inputs, ...
                                                 valid_targets, ...
                                                 hyperparameters);

    if isnan(f) || isinf(f)
        error('nan/inf error');
    end

    %% Update parameters.
    
    weights = weights - hyperparameters.learning_rate .* df / N;

    predictions_valid = logistic_predict(weights, valid_inputs);
    [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);
    cross_entropy_valids(t) = cross_entropy_valid

    %% Print some stats.
    fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
            t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);

end


plot(1:hyperparameters.num_iterations, cross_entropy_valids, '-;valid;', 1:hyperparameters.num_iterations, cross_entropy_trains, '-;train;');


