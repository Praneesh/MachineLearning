function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    % Computing h_i(x) = theta' * X
    gdSummationVector = zeros(size(X,2), 1);  % since we have only two features
    for i=1:m
       x_i = X(i, :);% Picking the ith row
       x = x_i';
       h_i = theta' * x; 
       y_i = y(i, :);
       
       predictionDifference = h_i - y_i; % this is a scalar
       %fprintf('Prediction Difference %f = %f\n', iter, predictionDifference);
       featureVector = predictionDifference * x; % x is a vector here
       gdSummationVector = gdSummationVector + featureVector;
    end     
    delta = (1/m) * gdSummationVector;
    %fprintf('Delta in interation %f\n', delta);
    theta = theta - (alpha * delta);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
