function [m,c,err]=linear_regression(x,y)
% 
% [m,c,err] = linear_regression(x,y)
% 
% fits a straight line to some data by minimizing the sum of squared
% y distances between the data and the fitted line.
% 
% Input:
%   x,y: data points to be fitted.
% Output:
%   m,c: slope and y-intercept of the fitted line y=m*x+c.
%   err: the y-distances between the data points and the fitted line.
% 
% copyright (c) Russell Fung 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  x = x(:);
  y = y(:);
  sum_x = sum(x);
  sum_xx = x'*x;
  sum_y = sum(y);
  sum_xy = x'*y;
  N = length(x);
  A = [sum_xx sum_x; sum_x N];
  b = [sum_xy; sum_y];
  soln = A\b;
  m = soln(1);
  c = soln(2);
  err = y-m*x-c;
%end
