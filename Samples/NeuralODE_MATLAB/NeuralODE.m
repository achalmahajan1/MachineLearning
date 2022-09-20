% Dynamical Systems modeling using neural ODE. The problem is a first order
% ODE with the form x' = Ax with x0 as it's initial condition. The neural
% network takes initial condition as the input and computes the ODE
% solution through the learned neural ODE model.
% Here is how the workflow looks like
% ----------- Input ----> Fully connect ----> tanh ----> Fully connect
% ----> Output.
% We will use the xTrain data as ground truth data for learning an
% approximated dynamics with a neural ODE model.
clear all;clc;close all;
x0 = [2; 0];
A = [-0.1 -1; 1 -0.1];
trueModel = @(t,y) A*y;% Actual model

numTimeSteps = 2000; % number of time steps in the time series
T = 15; % End time for the time series. Simulation is performed in the interval [0 15];
odeOptions = odeset(RelTol=1.e-7);
t = linspace(0, T, numTimeSteps);
[~, xTrain] = ode45(trueModel, t, x0, odeOptions);% Solving the ODE and then using it as a training data.
xTrain = xTrain';
% Visualize the training data in a plot.
%%
figure
plot(xTrain(1,:),xTrain(2,:))
title("Ground Truth Dynamics") 
xlabel("x(1)") 
ylabel("x(2)")
grid on
%%
%Define and Initialize Model Parameters
%The model function consists of a single call to dlode45 to solve the ODE defined by the approximated dynamics f(t,y,θ) for 40 time steps.
neuralOdeTimesteps = 40;
dt = t(2);
timesteps = (0:neuralOdeTimesteps)*dt;
neuralOdeParameters = struct;
% Initialize the parameters for the fully connected operations in the ODE model. The first fully connected operation takes as input a vector of size stateSize and increases its length to hiddenSize. Conversely, the second fully connected operation takes as input a vector of length hiddenSize and decreases its length to stateSize.
neuralOdeParameters.fc1 = struct;
stateSize = size(xTrain,1);
hiddenSize = 20;
% Number of layers in the neural networks depends on the inputs or
% variables of the ODE system.
sz = [hiddenSize stateSize];%{Number of variables Number of neurons or size of the hidden layer)
neuralOdeParameters.fc1.Weights = initializeGlorot(sz, hiddenSize, stateSize);
neuralOdeParameters.fc1.Bias = initializeZeros([hiddenSize 1]);

neuralOdeParameters.fc2 = struct;
sz = [stateSize hiddenSize];
neuralOdeParameters.fc2.Weights = initializeGlorot(sz, stateSize, hiddenSize);
neuralOdeParameters.fc2.Bias = initializeZeros([stateSize 1]);
%%
gradDecay = 0.9;
sqGradDecay = 0.999;
learnRate = 0.002;
% Train for 1200 iterations with a mini-batch-size of 200.

numIter = 1200;
miniBatchSize = 200;% Create a batch size of miniBatchSize as an input to the network.
% Every 50 iterations, solve the learned dynamics and display them against the ground truth in a phase diagram to show the training path.

plotFrequency = 50;
%% Setting up the plotting
f = figure;
f.Position(3) = 2*f.Position(3);

subplot(1,2,1)
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on
%%  Train the network using a custom training loop
averageGrad = [];
averageSqGrad = [];

numTrainingTimesteps = numTimeSteps;
trainingTimesteps = 1:numTrainingTimesteps;
plottingTimesteps = 2:numTimeSteps;

start = tic;

for iter = 1:1%numIter
    
    % Create batch 
    [X, targets] = createMiniBatch(numTrainingTimesteps, neuralOdeTimesteps, miniBatchSize, xTrain);

    % Evaluate network and compute loss and gradients
    [loss,gradients] = dlfeval(@modelLoss,timesteps,X,neuralOdeParameters,targets);
    
    % Update network 
    [neuralOdeParameters,averageGrad,averageSqGrad] = adamupdate(neuralOdeParameters,gradients,averageGrad,averageSqGrad,iter,...
        learnRate,gradDecay,sqGradDecay);
    
    % Plot loss
    subplot(1,2,1)
    currentLoss = double(loss);
    addpoints(lineLossTrain,iter,currentLoss);
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    title("Elapsed: " + string(D))
    drawnow
    
    % Plot predicted vs. real dynamics
    if mod(iter,plotFrequency) == 0  || iter == 1
        subplot(1,2,2)

        % Use ode45 to compute the solution 
        y = dlode45(@odeModel,t,dlarray(x0),neuralOdeParameters,DataFormat="CB");
        
        plot(xTrain(1,plottingTimesteps),xTrain(2,plottingTimesteps),"r--")
        
        hold on
        plot(y(1,:),y(2,:),"b-")
        hold off

        xlabel("x(1)")
        ylabel("x(2)")
        title("Predicted vs. Real Dynamics")
        legend("Training Ground Truth", "Predicted")

        drawnow
    end
end
%%
% openExample('nnet/TrainNeuralODENetworkWithRungeKuttaODESolverExample')