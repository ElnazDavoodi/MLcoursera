function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%disp(size(X));
%disp(size(y));
%disp(size(theta));
for iter = 1:num_iters

Temp1 = (theta' * X')-y';


theta(1,1) = theta(1,1) - ((alpha/m)*sum(Temp1*X(:,1))); 
theta(2,1) = theta(2,1) - ((alpha/m)*sum(Temp1*X(:,2))); 


J_history(iter) = computeCost(X, y, theta);
end


end
