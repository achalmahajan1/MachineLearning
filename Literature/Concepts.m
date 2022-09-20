% Basic ML and AI to understand the concepts
clear all;clc;close all;
mu = [0 0];
Sigma = [0.25 0.3; 0.3 1];
% Create a grid of evenly spaced points in two-dimensional space.

x1 = -3:0.2:3;
x2 = -3:0.2:3;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
% Evaluate the pdf of the normal distribution at the grid points.

y = mvnpdf(X,mu,Sigma);
y = reshape(y,length(x2),length(x1));

%% Surrogate optimization
clear all;clc;close all;
Read this part: https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec
