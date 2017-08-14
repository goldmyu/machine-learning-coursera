function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%




  
%plot(X(:,1),X(:,2),(y==1),'k+',X(:,1),X(:,2), (y==0), 'ko');


y_pos = find(y==1);
y_neg = find(y==0);

plot(X(y_pos,1),X(y_pos,2),'k+');

hold on;

plot(X(y_neg,1),X(y_neg,2),'ko');




% =========================================================================



hold off;

end
