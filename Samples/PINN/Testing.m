%% Testing some of the inbuilt functions 
clear all;clc;close all;
X = dlarray(randn(7,7,32),'SSCB');
weights = dlarray(ones(10,7,7,32)); bias = dlarray(ones(10,1));
% Compute fullyconnect
Z = fullyconnect(X,weights,bias);% Y = WEIGHTS*X + BIAS
%% Testing dlgradient
clear all;clc;close all;
x0 = dlarray([-1,2]);
[fval,gradval] = dlfeval(@rosenbrock,x0);
%% Testing how dlgradient works using my example
% start with a simple function y = sin(x)
clear all;clc;close all;
x = linspace(0,2*pi,50);
x = dlarray(x);
[y,dydx] = dlfeval(@trigsin,x);
plot(x,dydx,'-k',x,cos(x),'*r');
legend('autodiff solution','analytical solution')
%% Testing gradients and enforcing burgers equation
[gradval] = dlfeval(@num,U,X,T);
%% function for my test case
function [y,dydx] = trigsin(x) 
    y = sin(x); 
    dydx = dlgradient(sum(y,"all"),x,EnableHigherDerivatives=true);
end
%%
function [dUdX] = num(U,X,T) 
 dUdX = dlgradient(sum(U,"all"),{X,T},EnableHigherDerivatives=true);
end
%%
function [y,dydx] = rosenbrock(x)

y = 100*(x(2) - x(1).^2).^2 + (1 - x(1)).^2;
dydx = dlgradient(y1,x);
end
