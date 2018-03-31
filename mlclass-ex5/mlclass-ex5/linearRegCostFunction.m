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

% reset derivative d to 0
% d = zeros(size(theta));

for i = 1:m
    % z = theta' * X(i,:)';
    % z = X(i,:) * theta;
    % s = sigmoid(z);
    
    % A better way to code, use for loop to cover dJ(theta1...thetaN)
    for j = 1:size(grad)
        % grad(j) = grad(j) + 1 / m * ( (s - y(i)) * X(i,j) );
        grad(j) = grad(j) + 1 / m * ( (theta' * X(i,:)' - y(i)) * X(i,j) );
    end
    
    % calculate the cost J = 1/2m*Sum( h(x(i))-y(i) )
    % Not J = J + (1 / m * (- y(i) * log(s) - (1 - y(i)) * log(1-s)));
    J = J + (1 / (2 * m) * (theta' * X(i,:)' - y(i))^2);
    
end

% [J, grad] = costFunction(theta, X, y);

% calculate the regularized cost J = J + lamda/(2m)*theta(j)^2
n = size(theta);
h = 0;

% Not regularize theta0(zero), which is theta(1)
for j = 2:n
    h = h + (lambda / (2 * m)) * theta(j)^2;
 end
J = J + h;

% calculate derivative J(grad) and r = lamda/m*theta
r = zeros(size(theta));
r = (lambda / m) * theta; 
% reset r(1) as theta(0)=1, doesn't need to be regularized
r(1) = 0;
% add r(j) to grad(j)
grad = grad + r;


% =========================================================================

grad = grad(:);

end
