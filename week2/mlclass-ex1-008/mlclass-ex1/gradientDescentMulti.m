function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values


m = length(y); % number of training examples

J_history = zeros(num_iters, 1);

thetaSize = size(theta,1);

for iter = 1:num_iters
	Temp1 = (theta' * X')-y';
for i=1:thetaSize 
	theta(i,:) = theta(i,:) - ((alpha/m)*sum(Temp1*X(:,i))); 
	
end;


    J_history(iter) = computeCostMulti(X, y, theta);


end

end
