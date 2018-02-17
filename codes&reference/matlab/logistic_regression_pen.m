%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train_small;
load mnist_train;

load mnist_valid;

%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate = 5
% Weight regularization parameter
hyperparameters.weight_regularization = 0.5
% Number of iterations
hyperparameters.num_iterations = 100;
% Logistics regression weights
% TODO: Set random weights.
weights = rand(size(train_inputs,2)+1,1)/100;


%% Verify that your logistic function produces the right gradient, diff should be very close to 0
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on
% that data.
nexamples = 20;
ndimensions = 10;
diff = checkgrad('logistic_pen', ...
	             randn((ndimensions + 1), 1), ...   % weights
                 0.001,...                          % perturbation
                 randn(nexamples, ndimensions), ... % data        
                 rand(nexamples, 1), ...            % targets
                 hyperparameters)                   % other hyperparameters

N = size(train_inputs,1);
%% Begin learning with gradient descent.


cross_entropy_trains = zeros(1, 11)
cross_entropy_valids = zeros(1, 11)
classification_error_train = zeros(1, 11)
classification_error_valid = zeros(1, 11)

cross_entropy_trains_avg = zeros(1, 4)
cross_entropy_valids_avg = zeros(1, 4)
classification_error_avg = zeros(1, 4)




count = 1
for lambda = linspace(0,1,11)
    hyperparameters.weight_regularization = lambda
    for t = 1:hyperparameters.num_iterations

        %% TODO: You will need to modify this loop to create plots etc.
        
        

        % Find the negative log likelihood and derivative w.r.t. weights.
        [f, df, predictions] = logistic_pen(weights, ...
                                        train_inputs, ...
                                        train_targets, ...
                                        hyperparameters);

        [cross_entropy_train, frac_correct_train] = evaluate(train_targets, predictions)
        
        % Find the fraction of correctly classified validation examples.
        [temp, temp2, frac_correct_valid] = logistic_pen(weights, ...
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
        
        %% Print some stats.
        %fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
        %        t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);
    end

    cross_entropy_valids(count) = cross_entropy_valid;
    cross_entropy_trains(count) = cross_entropy_train;
    classification_error_valid(count) = 1-frac_correct_valid;
    classification_error_train(count) = 1-frac_correct_train;
    count = count + 1
end

plot(linspace(0,1,11), cross_entropy_valids, '-;valid_entropy;', linspace(0,1,11), cross_entropy_trains, '-;train_entropy;')
hold on;
plot(linspace(0,1,11), classification_error_valid, '-;valid_error;', linspace(0,1,11), classification_error_train, '-;train_error;')




