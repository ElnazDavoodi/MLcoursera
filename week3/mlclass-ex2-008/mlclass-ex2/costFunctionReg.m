function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
z = theta' * X';
h = sigmoid(z);
term1 = (y .* log(h)');
h2 = 1-h;
term2 = ((1-y) .* (log(h2))');

s=0;
for i=2:size(theta)
	s = s+(theta(i) .^ 2);
end;


J = ((-1/m)*sum(term1+term2))+ ((lambda/(2*m))*s)

X1 = X(:,1);


grad(1) = ((h'-y)'* X1) * (1/m);
for i=2:size(theta)
	grad(i) = ((lambda/m)*theta(i) + ((h'-y)'*X(:,i) .* (1/m)));
disp('------')
disp(grad)
disp('------')




% =============================================================

end
