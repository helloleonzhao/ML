function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

for i = 1:K
    random_row = round(rand() * size(X,1));
    centroids(i,:) = X(random_row,:);
end

% Suggested by Programming Exercis
% Randomly reorder the indices of examples
% randomX= randperm(size(X,1));
% centroids = X(randomX(1:K),:);



% =============================================================

end

