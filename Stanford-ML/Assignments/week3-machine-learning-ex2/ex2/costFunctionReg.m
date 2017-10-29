function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

[J, grad] = costFunction(theta, X, y);

% Calcluting Regularization Term
  theta_square_summation = 0.000;
  for col=2:length(theta),  % theta(1) should not be regularized as it would be set to 1 always
    theta_square_summation = theta_square_summation + (theta(col)) ** 2;    
  end   
  regularizationTerm = (lambda/(2*m))* theta_square_summation;
  
  J = J + regularizationTerm;

% Step 3: Calucalte the Gradient    
 
  % Regularized Gradient
  for row=2:length(grad),
      grad(row) = grad(row) + (lambda/m)*theta(row);
  end
  
% =============================================================

end









