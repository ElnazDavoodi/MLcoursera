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

% ====================== YOUR CODE HERE ======================
z = theta' * X';
h = sigmoid(z);
disp(log(h))
%disp('-----')

term1 = (y .* log(h)');
disp(term1)

h2 = 1-h;
disp(log(h2))
term2 = ((1-y) .* (log(h2))');
disp(term2);


a = term1+term2;
disp(a);
A = sum(a);
J = (-1/m) * A;
disp(J);

B = (h'-y)'* X;
grad = B .* (1/m);
disp(grad);

end
