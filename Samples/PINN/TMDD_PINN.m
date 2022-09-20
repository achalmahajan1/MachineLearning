% Physics informed Neural Networks
clear all;clc;close all;
% Here we will start with the example of basic target mediated drug
% disposition model and using physics informed neural network to generate
% the underlying neural network model which accounts for the underlying

numInitialConditionPoints  = 50;

t0IC = zeros(1,numInitialConditionPoints);
T0IC = 10;
C0IC = 0;
D0IC = 0;
% Grouping together data for initial and boundary conditions

% Select 10,000 points to enforce the output of the network to fulfill the
% Burger's equation
numInternalCollocationPoints = 10000;

pointSet = sobolset(1); % This just create sequence of random numbers (first dimension is X and second dimension is t).
points = net(pointSet,numInternalCollocationPoints);

dataT = points.*10;% Ranges from 0 to 1

% Creating an array datastore containing the training set
ds = arrayDatastore(dataT);

%% Let's start to define the Deep Learning Model
% We define a multilayer perception architecture with 9 fully connect
% operations and 20 hidden neurons. The first fully connect operation has
% two input channels corresponding to the inputs x and t. The last fully
% connect operation has one output u(x,t).
numLayers = 9;
numNeurons = 20;
% Initialize the parameters for the first fully connect operation. The first fully connect operation has two input channels.
parameters = struct;
sz = [numNeurons 2];
parameters.fc1.Weights = initializeHe(sz,2);% This function generates weights from normal distribution for 20 neurons in each layer
parameters.fc1.Bias = initializeZeros([numNeurons 1]);
% Initialize the parameters for each of the remaining intermediate fully
% connect operations.
for layerNumber=2:numLayers-1% looping over number of layers except the first and last layer.
    name = "fc"+layerNumber;
    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe(sz,numIn);
    parameters.(name).Bias = initializeZeros([numNeurons 1]);
end
% Initialize the parameters for the final fully connect operation. The final fully connect operation has one output channel.
sz = [1 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers).Weights = initializeHe(sz,numIn);
parameters.("fc" + numLayers).Bias = initializeZeros([1 1]);
%% Let's now define Model and Model loss Functions
% We will train the model for 3000 epochs with a mini-batch size of 1000.
numEpochs = 1;
miniBatchSize = 1000;
% Specify ADAM optimization options
initialLearnRate = 0.01;
decayRate = 0.005;
%% We can train the network now
mbq = minibatchqueue(ds, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFormat="BC");
% Convert the intial and boundary conditions to dlarray.
t0 = dlarray(t0IC,"CB");
T0 = dlarray(T0IC);
C0 = dlarray(C0IC);
D0 = dlarray(D0IC);
% Initialize the parameters for the Adam solver.
averageGrad = [];
averageSqGrad = [];
% Accelerate the model loss function using the dlaccelerate function
accfun = dlaccelerate(@modelLoss);
% Initialize the training progress plot.
figure
C = colororder;
lineLoss = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on
%%
% Train the network. For each iteration
% 1. Read a mini-batch of data from the mini-batch queue.
% 2. Evaluate the model loss and gradients using the accelerated model loss
% and dlfeval functions
% 3. Update the learning rate. Is this part of Adams optimization?
% 4. Update the learnable parameters using the adamupdate function
start = tic;

iteration = 0;

for epoch = 1:numEpochs
    reset(mbq);

    while hasdata(mbq)
        iteration = iteration + 1;

        XT = next(mbq);% Out of 10000 total data set, you can use next function to obtain mini-batches from mbq. 
        % The actual data is saved in dataX and dataT. You can compare the
        % first 1000 enteries of dataX and dataT and XT for confirmation.
        X = XT(1,:);
        T = XT(2,:);

        % Evaluate the model loss and gradients using dlfeval and the
        % modelLoss function.
        [loss,gradients] = dlfeval(accfun,parameters,X,T,X0,T0,U0);

        % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);

        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end

    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration, loss);

    D = duration(0,0,toc(start),Format="hh:mm:ss");
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
end