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
% reset derivative d to 0
d = zeros(size(theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% reset derivative d to 0
% d1 = 0;
% d2 = 0;
% d3 = 0;

for i = 1:m
    z = theta' * X(i,:)';
    s = sigmoid(z);
    
    % calculate derative of J(theta) = sum of (1/m * (H(X*theta)-y)*X(i)
    % total 3 thetas, theta0, theta1, and theta2
    % d1 = d1 + 1 / m * ((s - y(i)) * X(i,1));
    % d2 = d2 + 1 / m * ((s - y(i)) * X(i,2));
    % d3 = d3 + 1 / m * ((s - y(i)) * X(i,3));
    % fprintf('d1: %f, d2: %f, d3: %f\n', d1, d2, d3);
    
    % A better way to code, use for loop to cover dJ(theta1...thetaN)
    for j = 1:size(d)
        d(j) = d(j) + 1 / m * ( (s - y(i)) * X(i,j) );
    end
    
    % error handling when log(0) happens
    % if isinf(log(s)) == 1
    %    logS = -1.0000e+100;
    % else logS = log(s);
    % end
    % if isinf(log(1-s)) == 1
    %    logNS = -1.0000e+100;
    % else logNS = log(1-s);
    % end
    
    % calculate the cost J = ...
    % sum of (-1/m * y(i)*log(sigmoid(X*theta) + (1-y(i))*log(1-sigmoid(X*theta))
    J = J + (1 / m * (- y(i) * log(s) - (1 - y(i)) * log(1-s)));
    
end

% return derative J to grad(Gradient Descent)
grad = d;

% =============================================================

end
