function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% Go over every centroids
for k = 1:K
    % initialize number of X in one cluster(a.k.a. centroids)
    numberOfX = 0;
    % initialize sum of X in one cluster
    sumOfX = zeros(n, 1);
    for i = 1:m
        if ( idx(i) == k )
            numberOfX = numberOfX + 1;
            % calculate with column vector<n by 1>
            sumOfX = sumOfX + X(i, :)';
        end
    end
    % tranpose from column vector back to row vector and store the new cetroids in row k
    centroids(k, :) = (sumOfX / numberOfX)';
end



% =============================================================


end

