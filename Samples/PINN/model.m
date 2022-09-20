function U = model(parameters,X,T)
% The function model takes an input the model parameters and the network
% inputs, and returns the model output
XT = [X;T];
numLayers = numel(fieldnames(parameters));

% First fully connect operation.
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
U = fullyconnect(XT,weights,bias); % Performs U = weights*XT + bias
% The size of U will depend on the number of hidden units (20 in this
% case) and the data size (1000 in this case, for mini batch size)
% tanh (activation function) and fully connect operations for remaining
% layers. Now the  number of parameters are unique to each layer too which
% are the weights (20*2 for layer 1, 20*20 for the n-1) and bias (20*1 for
% all layers).
% The first layer has two input channels (x,t), and output has one u(x,t).
% In between hidden layers has 20 inputs and 20 ouputs
for i=2:numLayers
    name = "fc" + i;

    U = tanh(U);

    weights = parameters.(name).Weights;
    bias = parameters.(name).Bias;
    U = fullyconnect(U, weights, bias);
end

end