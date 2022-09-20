function [loss,gradients] = modelLoss1(parameters,kel,t,kel0,t0,T0,C0,D0)
% Takes as input as model paramters, the network inputs, and the initial
% and boundary conditions and returns the loss and gradients of the loss
% with respect to the learnable parameters.
% Make predictions with the initial conditions.
% Let's define the parameters of the ODE as well
% kel = 0.5230;
kon = 0.0485;
km = 0.0458;
koff = 0.0138;
kdeg = 0.0934;
ksyn = 0.934;
%T = model(parameters,t);% Calculate XT*weights + bias at each layer, doing the forward propagation
%C = model(parameters,t);% Calculate XT*weights + bias at each layer, doing the forward propagation
%D = model(parameters,t);% Calculate XT*weights + bias at each layer, doing the forward propagation
Pred = model1(parameters,kel,t);% Calculate XT*weights + bias at each layer, doing the forward propagation
T = Pred(1,:);
C = Pred(2,:);
D = Pred(3,:);

% Calculate derivatives with respect to X and T.
gradientsT = dlgradient(sum(T,"all"),t,EnableHigherDerivatives=true); % calculate derivatives of T with respect to t
gradientsC = dlgradient(sum(C,"all"),t,EnableHigherDerivatives=true); % calculate derivatives of C with respect to t
gradientsD = dlgradient(sum(D,"all"),t,EnableHigherDerivatives=true); % calculate derivatives of D with respect to t

Tt = gradientsT;
Dt = gradientsD;
Ct = gradientsC;

% Calculate second-order derivatives with respect to X.
% Uxx = dlgradient(sum(Ux,"all"),X,EnableHigherDerivatives=true); Can be
% used for a PDE

% Calculate lossF. Enforce Burger's equation.
%f = Ut + U.*Ux - (0.01./pi).*Uxx;
f1 = Tt + kon.*T.*D - koff.*C - ksyn + kdeg.*T;
f2 = Ct - kon.*T.*D + koff.*C + km.*C;
f3 = Dt + kel.*D + kon.*T.*D - koff.*C;
        
zeroTarget = zeros(size(f1), "like", f1);
lossF1 = mse(f1, zeroTarget);
zeroTarget = zeros(size(f2), "like", f2);
lossF2 = mse(f2, zeroTarget);
zeroTarget = zeros(size(f3), "like", f3);
lossF3 = mse(f3, zeroTarget);

% Calculate lossU. Enforce initial and boundary conditions.

ICPred = model1(parameters,kel0,t0);
T0Pred = ICPred(1,:);
C0Pred = ICPred(2,:);
D0Pred = ICPred(3,:);

lossT = mse(T0Pred, T0);
lossD = mse(D0Pred, D0);
lossC = mse(C0Pred, C0);
% lossF4 = mse(T0Pred+D0Pred+C0Pred, T0+D0+C0);
% Somehow if you combine the boundary conditions like lossF4, most likely the
% boundary conditions will not be satisfied. So, I think each loss term
% should be satisfied independently.

% Combine losses.
 loss = lossF1 + lossF2 + lossF3 + lossT + lossD + lossC;
% loss = lossF1 + lossF2 + lossF3 + lossF4;
% loss = lossF1 + lossT + lossD + lossC;
% loss = lossF1 + lossF4;
% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end