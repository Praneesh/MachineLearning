function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Step 1: Calulate sigmoid function of all the values in data set
  % Calculating - theta' * X for all the values and involking sigmoid function
    htheta = zeros(m, 1);
    for row=1:m,
      feature_row = X(row,:); % Get each row in the X
      x = feature_row'; % Transpose the x
      z = theta' * x; 
      htheta(row) = sigmoid(z);
    end

    
% Step 2: Calucalte the Cost Function
  summation = 0.00000;
  for row=1:m,
     acceptedStatus = y(row); % Gives if it is 1 or 0 as per the dataset
     p_one = -(acceptedStatus) * log(htheta(row));
     p_zero = (1-acceptedStatus) * log(1-htheta(row));
     summation = summation + (p_one - p_zero);     
  end
  J = summation/m;
  
% Step 3: Calucalte the Gradient      
  for feature_col=1:length(grad),
    grad_summation = 0.00000;
    for row=1:m,
       pre_summation = (htheta(row) - y(row))* X(row, feature_col);
       grad_summation = grad_summation + pre_summation;
    end
    grad(feature_col) = grad_summation/m;
  end  
% =============================================================

end
