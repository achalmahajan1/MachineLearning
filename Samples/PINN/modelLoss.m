function [loss,gradients] = modelLoss(parameters,X,T,X0,T0,U0)
% Takes as input as model paramters, the network inputs, and the initial
% and boundary conditions and returns the loss and gradients of the loss
% with respect to the learnable parameters.
% Make predictions with the initial conditions.
U = model(parameters,X,T);% Calculate XT*weights + bias at each layer, doing the forward propagation

% Calculate derivatives with respect to X and T.
gradientsU = dlgradient(sum(U,"all"),{X,T},EnableHigherDerivatives=true); % calculate derivatives with respect to X and T

%My version
%for i = 1:length(U)
%    gradientsU(i,:) = dlgradient(U(i),{X(i),T(i)},EnableHigherDerivatives=true);
%end
Ux = gradientsU{1};
Ut = gradientsU{2};

% Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(Ux,"all"),X,EnableHigherDerivatives=true);

% Calculate lossF. Enforce Burger's equation.
f = Ut + U.*Ux - (0.01./pi).*Uxx;
zeroTarget = zeros(size(f), "like", f);
lossF = mse(f, zeroTarget);

% Calculate lossU. Enforce initial and boundary conditions.
U0Pred = model(parameters,X0,T0);
lossU = mse(U0Pred, U0);
% Combine losses.
loss = lossF + lossU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end