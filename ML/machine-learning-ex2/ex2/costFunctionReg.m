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
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);

J = sum(-1*((log(h)'*y + log(1-h)'*(1-y))))/m + (lambda*(theta(2:end)'*theta(2:end))/(2*m));

delta = h-y;

grad(2:end) = (X(:,2:end)'*delta)./m + lambda*(theta(2:end))/m;
grad(1) = (X(:,1)'*delta)/m;

%grad(1) = (X(:,1)'*delta)/m;
%for i=2:length(theta)
%  grad(i) = (X(:,i)'*delta)/m + lambda*theta(i)/m;
%end;

% =============================================================

end
