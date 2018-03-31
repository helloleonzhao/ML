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

% call costFunction without Regularized cost J and gradient descent
[J, grad] = costFunction(theta, X, y);

% calculate derivative J(grad) and r = lamda/m*theta
r = zeros(size(theta));
r = (lambda / m) * theta; 
% reset r(1) and theta(0)=1, doesn't need to be regularized
r(1) = 0;
theta(1) = 0;
% add r(j) to grad(j)
grad = grad + r;

% calculate the regularized cost J = J + lamda/(2m)*theta(j)^2
n = size(theta);
h = 0;
for j = 1:n
    h = h + (lambda / (2 * m)) * theta(j)^2;
end
J = J + h;

% =============================================================

end
