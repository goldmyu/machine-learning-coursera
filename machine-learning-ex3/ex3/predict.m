function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

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



% This is a vectorized calculation for a K class neural network

%we add the bias element to the first layer, AKA the input layer.
%hence we get 1,x1,x2...,xn for n feature we have a total of n+1 inputs into the hidden layer
 X = [ones(m,1) X];


% This is a vectorized calc of the activation layer 2, meaning a2 matrix, each row in the matrix represents n samples (meaning one imput)
%each row is like a single input of an n feature set into the neural netowrk, with additon of the bias,
% then we calc the sigmoid fun of X*Theta1'
a_2_matrix = [ones(m,1) sigmoid(X*Theta1')];


a_3_matrix = sigmoid(a_2_matrix*Theta2');


% each value in vector p will hold the index which has the highst probability for a given sample of n features x1,...,xn.
% each element in p represents our neural netowrk prediction for a given sample. when we have m samples compraised of n feature not including the "DC" feature\off-set\bias what ever you want to call it.
[prob p] = max(a_3_matrix, [], 2);


% =========================================================================


end
