function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% LINEAR REGRESSION: REGULARISED COST FUNCTION

% lamda : Regularisation parameter helps prevent overfitting by keeping all thetas small and smoothens the curve.
%         Else, some thetas can assume large value and some small values, and may lead to overfitting.
%         By using Regularisation, we are making sure all thetas are small and all features given equal importance,
%         since its hard to say which feature is importatnt.


cost_term = sum( ((X*theta) - y).^2 )
reg_term  = lambda * sum(theta(2:end).^2)

J = (1/(2*m)) * (cost_term + reg_term)



% LINEAR REGRESSION: REGULARISED GRADIENT FUNCTION



% =========================================================================

grad = grad(:);

end
