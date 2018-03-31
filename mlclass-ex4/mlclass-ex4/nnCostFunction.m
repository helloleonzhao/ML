function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% add bias 1 to a1, which is X;
X = [ones(m, 1) X];

% initialize yVector(numberOfLabels by numberOfSamples)
yVector = zeros(num_labels, m);


% Feedforward and Cost Function with Regularization Code Here

for i = 1:m
    
    % change y [1;2;3...] to yVector [1;0;0...];[0;1;0...]
    yVector(y(i),i) = 1;
        
    % use Theta1(25by401) and X(i,;)(1by401) to calculate a2(25by1)
    z2 = Theta1 * X(i,:)';
    a2 = sigmoid(z2);
    
    % add bias 1 to a2(26by1);
    a2 = [1; a2];
    
    % use Theta2(10by26) and a2(26by1) to calculate a3(10by1)
    z3 = Theta2 * a2; 
    a3 = sigmoid(z3);
     
    for k = 1:num_labels  
        % calculate the cost J = ...
        % sum of (-1/m * y(i)*log(sigmoid(X*theta) + (1-y(i))*log(1-sigmoid(X*theta))
        J = J + (1 / m * (- yVector(k,i) * log(a3(k)) - (1 - yVector(k,i)) * log(1-a3(k))));
    
        % A better way to code, use for loop to cover dJ(theta1...thetaN)
        % for j = 1:size(grad)
        % grad(j) = grad(j) + 1 / m * ( (s - y(i)) * X(i,j) );
        % end
    
    end
end

% Regularized Cost Function
% calculate the regularized cost h = h + lamda/(2m)*(Theta1^2+Theta2^2)
h = 0;

for j = 1:hidden_layer_size
    % Theta(j,1) (a.k.a. the bias) is NOT regularized
    for k = 2:(input_layer_size+1)
        h = h + (lambda / (2 * m)) * Theta1(j,k)^2;
    end
end

for j = 1:num_labels
    % Theta(j,1) (a.k.a. the bias) is NOT regularized
    for k = 2:(hidden_layer_size+1)
        h = h + (lambda / (2 * m)) * Theta2(j,k)^2;
    end
end

% add regularized cost h to cost function J
J = J + h;

% Backpropagation with Regularized Gradient Code Here

% initialize yVector(numberOfLabels by numberOfSamples)
yVector = zeros(num_labels, m);

for k = 1:m
    
    % =========== Step 1: Calculate a1(400by1), a2(25by1), a3(10by1) =============
    % set input layer's value a(k) to X(k) 
    a1 = X(k,:); 
        
    % use Theta1(25by401) and X(i,;)(1by401) to calculate a2(25by1)
    z2 = Theta1 * X(k,:)';
    a2 = sigmoid(z2);
    
    % add bias 1 to a2(26by1);
    a2 = [1; a2];
    
    % use Theta2(10by26) and a2(26by1) to calculate a3(10by1)
    z3 = Theta2 * a2; 
    a3 = sigmoid(z3);
     
    % =========== Step 2: Calculate d3(10by1) =============
    % change y [1;2;3...] to yVector [1;0;0...];[0;1;0...]
    yVector(y(k),k) = 1;
    d3 = a3 - yVector(:,k);

    % =========== Step 3: Calculate d2(25by1) =============
    % size(Theta2) = [10,26]
    % size(d3) = [10,1]
    delta_2 = Theta2' * d3;
    % Remove d2 from (26by10) to (25by10) because of bias unit 
    delta_2 = delta_2(2:end);
    s = sigmoidGradient(z2);
    d2 = delta_2 .* s;

    % =========== Step 4: Accumulate each sample's gradients into Theta1 and Theta2 ===
    % size(d2) = [25,1]
    % size(a1) = [1,401]
    % size(Theta1_grad) = [25, 401]
    Theta1_grad = Theta1_grad + d2 * a1;
    
    % size(d3) = [10,1]
    % size(a2) = [26,1]
    % size(Theta2_grad) = [10,26]
    Theta2_grad = Theta2_grad + d3 * a2';
end

% =========== Step 5: Calculate Gradient Descent for Theta1 and Theta2 =============
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% Regularized Theta1_grad (25by401)
% size(Theta1) = [25,401]
r = zeros(size(Theta1));
r = (lambda / m) * Theta1; 
% reset r(1) as theta(0)=1 (a.k.a. j=0) doesn't need to be regularized
r(:,1) = zeros(hidden_layer_size,1);
% add r to Theta1_grad(25by401)
Theta1_grad = Theta1_grad + r;

% Regularized Theta2_grad (10by26)
% size(Theta2) = [10,26]
r = zeros(size(Theta2));
r = (lambda / m) * Theta2; 
% reset r(1) as theta(0)=1 (a.k.a. j=0) doesn't need to be regularized
r(:,1) = zeros(num_labels,1);
% add r to Theta2_grad(10by26)
Theta2_grad = Theta2_grad + r;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
