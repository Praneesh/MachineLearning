function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

summation = 0.00;
  for i=1:m,
    x_i = X(i, :); % Picking the ith row
    x = x_i'; % Row is set of multiple features, we are transposing it to fit our equation
    h_i = theta' * x; 
    y_i = y(i, :);

    difference = h_i - y_i;
    mean_square = difference ** 2;
    summation = summation + mean_square;
  end
  multiplicationConstant = 1/(2*m);
  J = multiplicationConstant * summation;
end