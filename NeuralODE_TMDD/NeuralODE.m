% Dynamical Systems modeling using neural ODE. The problem is a first order
% ODE with the form x' = Ax with x0 as it's initial condition. The neural
% network takes initial condition as the input and computes the ODE
% solution through the learned neural ODE model.
% Here is how the workflow looks like
% ----------- Input ----> Fully connect ----> tanh ----> Fully connect
% ----> Output.
% We will use the xTrain data as ground truth data for learning an
% approximated dynamics with a neural ODE model.

%Update: Since the time series is not constant it might be hard to
clear all;clc;close all;
format short;
% Need to setup the system. Ideally we should describe the model and then train the data using the
% model but that would be the next step.
results = load('Data.txt');
t1 = load('Time.txt')';
x0 = [10; 0; 5];
numTimeSteps = 5000; % number of time steps in the time series
T = 80; % End time for the time series. Simulation is performed in the interval [0 15];
t = linspace(0, T, numTimeSteps);
T = results(:,1);% Target concentration with time
C = results(:,2);% Complex concentration with time
D = results(:,3);% Drug concentration with time
[~,xTrain] = ode45(@odefcn,t,x0);
% plot(t,xTrain(:,1),'-k',t,xTrain(:,2),'-r',t,xTrain(:,3),'-g');
% hold on
% plot(t1(1:5:end),T(1:5:end),'*k',t1(1:5:end),C(1:5:end),'*r',t1(1:5:end),D(1:5:end),'*g');
xTrain = xTrain';
%%
%Define and Initialize Model Parameters
%The model function consists of a single call to dlode45 to solve the ODE defined by the approximated dynamics f(t,y,θ) for 40 time steps.
neuralOdeTimesteps = 100;
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
sz = [hiddenSize hiddenSize];
neuralOdeParameters.fc2.Weights = initializeGlorot(sz, hiddenSize, hiddenSize);
neuralOdeParameters.fc2.Bias = initializeZeros([hiddenSize 1]);

neuralOdeParameters.fc3 = struct;
neuralOdeParameters.fc3.Weights = initializeGlorot(sz, hiddenSize, hiddenSize);
neuralOdeParameters.fc3.Bias = initializeZeros([hiddenSize 1]);

neuralOdeParameters.fc4 = struct;
neuralOdeParameters.fc4.Weights = initializeGlorot(sz, hiddenSize, hiddenSize);
neuralOdeParameters.fc4.Bias = initializeZeros([hiddenSize 1]);

%neuralOdeParameters.fc5 = struct;
%neuralOdeParameters.fc5.Weights = initializeGlorot(sz, hiddenSize, hiddenSize);
%neuralOdeParameters.fc5.Bias = initializeZeros([hiddenSize 1]);

%neuralOdeParameters.fc6 = struct;
%neuralOdeParameters.fc6.Weights = initializeGlorot(sz, hiddenSize, hiddenSize);
%neuralOdeParameters.fc6.Bias = initializeZeros([hiddenSize 1]);

neuralOdeParameters.fc5 = struct;
sz = [stateSize hiddenSize];
neuralOdeParameters.fc5.Weights = initializeGlorot(sz, stateSize, hiddenSize);
neuralOdeParameters.fc5.Bias = initializeZeros([stateSize 1]);
%%
gradDecay = 0.9;
sqGradDecay = 0.999;
learnRate = 0.002;
% Train for 1200 iterations with a mini-batch-size of 200.

numIter = 3000;
miniBatchSize = 500;% Create a batch size of miniBatchSize as an input to the network.
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

for iter = 1:numIter
    
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
%% Now try to check if it can reproduce the results with a different initial condition
tPred = t;
% 10;0;5
x0Pred1 = [15;0;3];
x0Pred2 = [50;10;4];
x0Pred3 = [11;1;6];
x0Pred4 = [4;20;25];
% Numerically solve for the 4 different conditions
[~, xTrue1] = ode45(@odefcn, tPred, x0Pred1);
[~, xTrue2] = ode45(@odefcn, tPred, x0Pred2);
[~, xTrue3] = ode45(@odefcn, tPred, x0Pred3);
[~, xTrue4] = ode45(@odefcn, tPred, x0Pred4);
% Numerically solve the ODE with the learned neural ODE dynamics.
xPred1 = dlode45(@odeModel,tPred,dlarray(x0Pred1),neuralOdeParameters,DataFormat="CB");
xPred2 = dlode45(@odeModel,tPred,dlarray(x0Pred2),neuralOdeParameters,DataFormat="CB");
xPred3 = dlode45(@odeModel,tPred,dlarray(x0Pred3),neuralOdeParameters,DataFormat="CB");
xPred4 = dlode45(@odeModel,tPred,dlarray(x0Pred4),neuralOdeParameters,DataFormat="CB");
%%

figure
subplot(2,2,1)
plotTrueAndPredictedSolutions(t, xTrue1, xPred1);
subplot(2,2,2)
plotTrueAndPredictedSolutions(t, xTrue2, xPred2);
subplot(2,2,3)
plotTrueAndPredictedSolutions(t, xTrue3, xPred3);
subplot(2,2,4)
plotTrueAndPredictedSolutions(t, xTrue4, xPred4);