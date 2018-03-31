function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% add bias 1 to a1, which is X;
X = [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Total 5000 samples from X(5000by401, including bias), which is a1; 
% predict one sample at a time;
for i = 1:m
    
    % use Theta1(25by401) and X(i,;)(1by401) to calculate a2(25by1)
    z2 = Theta1 * X(i,:)';
    a2 = sigmoid(z2);
    
    % add bias 1 to a2(26by1);
    a2 = [1; a2];
    
    % use Theta2(10by26) and a2(26by1) to calculate a2(10by1)
    z3 = Theta2 * a2; 
    a3 = sigmoid(z3);
    
    % The highest match in 10 labels is the predict of this sample;
    for j = 1:num_labels
        if a3(j) == max(a3)
            p(i) = j;
        end
    end

end









% =========================================================================


end
