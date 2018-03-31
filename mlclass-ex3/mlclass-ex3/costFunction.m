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

for i = 1:m
    % z = theta' * X(i,:)';
    z = X(i,:) * theta;
    s = sigmoid(z);
    
    % A better way to code, use for loop to cover dJ(theta1...thetaN)
    for j = 1:size(grad)
        grad(j) = grad(j) + 1 / m * ( (s - y(i)) * X(i,j) );
    end
    
    % calculate the cost J = ...
    % sum of (-1/m * y(i)*log(sigmoid(X*theta) + (1-y(i))*log(1-sigmoid(X*theta))
    J = J + (1 / m * (- y(i) * log(s) - (1 - y(i)) * log(1-s)));
    
end

% fprintf('J is %f\n', J);
% fprintf('grad is %f\n', grad);

% =============================================================

end
