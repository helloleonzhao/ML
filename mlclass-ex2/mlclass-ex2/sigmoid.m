function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z = 1/(1+exp(-z))

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% this fuction will be called in costFunction for Logistic Regression
% [m, n] = size(g);
% for i = 1:m
%    for j = 1:n        
%        g(i,j) = 1 / (1 + exp(-z(i,j)));
%    end
% end

% another approach with .operation, much faster than for loops
g = 1 ./ (exp(-z) + 1);


% =============================================================

end
