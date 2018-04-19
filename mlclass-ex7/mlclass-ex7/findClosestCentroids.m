function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

[m,n] = size(X);

for i = 1:m
    closest_distance = inf;
    for k = 1:K
        % should consider a vectorized implementation to save a loop K
        diff = X(i, :)' - centroids(k, :)';
        distance = diff' * diff;
        
        % an example of hard coded with feature n = 2
        % x1 = X(i,1) - centroids(k,1);
        % x2 = X(i,2) - centroids(k,2);
        % distance = sqrt(x1^2 + x2^2);
        if (distance < closest_distance)
            idx(i) = k;
            closest_distance = distance;
        end
    end
end
    





% =============================================================

end

