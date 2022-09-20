% Physics informed Neural Networks (TMDD model)
clear all;clc;close all;
% Here we will start with the example of basic target mediated drug
% disposition model and using physics informed neural network to generate
% the underlying neural network model and compare the simulation results
% New update 5/30/22: Adding doses into the PINN model. The idea behind is
% that, we can divide the time series into ranges where the doses are
% added, and train them 1 by 1 such that initial BC for the next time
% series is the final output of the previous time series.
numInitialConditionPoints = 1;
tend = 10;% Simulation end time
t0IC = zeros(1,numInitialConditionPoints);
T0IC = 10;
C0IC = 0;
D0IC = 5;
% Dose information (Number of time intervals need to formed based on the
% dosing information): Could be obtained from sim object
Amount = 60;
Interval = 1;
RepeatCount = 4;
Starttime = 0;
% Select 10,000 points to enforce the output of the network to fulfill the
% TMDD coupled equations
numInternalCollocationPoints = 10000;

pointSet = sobolset(1); % This just create sequence of random numbers (first dimension is X and second dimension is t).
points = net(pointSet,numInternalCollocationPoints);
numpoints = numInternalCollocationPoints*RepeatCount/tend;% Number of points must be distributed according to the repeatCount and interval
% This loops divide the time series into ranges where doses are added. For
% example for RepeatInterval of 4 and interval of 1, this creates 4 time
% ranges 0-1,1-2,2-3,3-4 where doses are applied and then another one 4-10
% where no doses are applied.
for i = 1:RepeatCount
    dataT_1((i-1)*numpoints/RepeatCount+1:i*numpoints/RepeatCount) = Interval*(i-1) + points((i-1)*numpoints/RepeatCount+1:i*numpoints/RepeatCount).*Interval;% Ranges from 0 to 1 (multiplied by the time range)
end
% After we are done dividing the time series where the doses are added.
% Rest of the time series is separately added where there is no dose.
i = i+1;
dataT_1((i-1)*numpoints/RepeatCount+1:numInternalCollocationPoints) = Interval*(i-1) + (tend - Interval*(i-1))*points((i-1)*numpoints/RepeatCount+1:numInternalCollocationPoints);
% Creating an array datastore containing the training set
%% Let's start to define the Deep Learning Model
% We define a multilayer perception architecture with 9 fully connect
% operations and 20 hidden neurons. The first fully connect operation has
% two input channels corresponding to the inputs x and t. The last fully
% connect operation has one output u(x,t).
numLayers = 9;
numNeurons = 20;
% Initialize the parameters for the first fully connect operation. The first fully connect operation has two input channels.
parameters = struct;
sz = [numNeurons 1];
parameters.fc1.Weights = initializeHe(sz,1);% This function generates weights from normal distribution for 20 neurons in each layer
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
% Initialize the parameters for the final fully connect operation. The final fully connect operation has three output channel.
sz = [3 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers).Weights = initializeHe(sz,numIn);
parameters.("fc" + numLayers).Bias = initializeZeros([3 1]);
%% Let's now define Model and Model loss Functions
% We will train the model for 3000 epochs with a mini-batch size of 1000.
numEpochs = 500;%
numPredictions = 1001;
% Specify ADAM optimization options
initialLearnRate = 0.01;
decayRate = 0.005;
TPred(1:RepeatCount+1,1:numPredictions) = T0IC;
CPred(1:RepeatCount+1,1:numPredictions) = C0IC;
DPred(1:RepeatCount+1,1:numPredictions) = D0IC;
T0 = dlarray(TPred(1,end));
C0 = dlarray(CPred(1,end));
D0 = dlarray(DPred(1,end)+Amount);
%%
for i = 1:RepeatCount+1
  % We can train the network now
    if i == RepeatCount+1
        miniBatchSize = length(dataT_1((i-1)*numpoints/RepeatCount+1:numInternalCollocationPoints))/120;% 
        ds_1 = arrayDatastore(dataT_1((i-1)*numpoints/RepeatCount+1:numInternalCollocationPoints)');        
        mbq = minibatchqueue(ds_1, ...
            MiniBatchSize=miniBatchSize, ...
            MiniBatchFormat="BC");
        TTest1(i,:) = linspace(Interval*(i-1),tend,numPredictions);
    else
        miniBatchSize = length(dataT_1((i-1)*numpoints/RepeatCount+1:i*numpoints/RepeatCount))/20;% 
        ds_1 = arrayDatastore(dataT_1((i-1)*numpoints/RepeatCount+1:i*numpoints/RepeatCount)');
        mbq = minibatchqueue(ds_1, ...
            MiniBatchSize=miniBatchSize, ...
            MiniBatchFormat="BC");
        TTest1(i,:) = linspace(Interval*(i-1),i*Interval,numPredictions);
    end
    t0 = dlarray(t0IC+Interval*(i-1),"CB");
    % Convert the intial and boundary conditions to dlarray.
    t0
    T0
    C0
    D0
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
    
            T1 = next(mbq);% Out of 10000 total data set, you can use next function to obtain mini-batches from mbq. 
            % The actual data is saved in dataX and dataT. You can compare the
            % first 1000 enteries of dataX and dataT and XT for confirmation.
            % Evaluate the model loss and gradients using dlfeval and the
            % modelLoss function.
            [loss,gradients] = dlfeval(accfun,parameters,T1,t0,T0,C0,D0);

            % Update learning rate.
            learningRate = initialLearnRate / (1+decayRate*iteration);
    
            % Update the network parameters using the adamupdate function.
            [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                averageSqGrad,iteration,learningRate);
        end
        % loss
        % Plot training progress.
        loss = double(gather(extractdata(loss)));
        addpoints(lineLoss,iteration, loss);
    
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
        drawnow
    end
% Prediction from the first will be used as an input to the next time
% interval
    TTest = dlarray(TTest1(i,:),"CB");
    Pred = model(parameters,TTest);
    TPred(i,:) = Pred(1,:);
    CPred(i,:) = Pred(2,:);
    DPred(i,:) = Pred(3,:);

    T0 = dlarray(TPred(i,end));
    C0 = dlarray(CPred(i,end));
    D0 = dlarray(DPred(i,end)+Amount);% Here we add the amount of drug at the end of one time series.
end
% Let now train the data for the
%% Let's compare with the actual solution
figure
% Concatenate data from the Predictions
Tf = [TPred(1,:) TPred(2,:) TPred(3,:) TPred(4,:) TPred(5,:)];
Cf = [CPred(1,:) CPred(2,:) CPred(3,:) CPred(4,:) CPred(5,:)];
Df = [DPred(1,:) DPred(2,:) DPred(3,:) DPred(4,:) DPred(5,:)];
tf = [TTest1(1,:) TTest1(2,:) TTest1(3,:) TTest1(4,:) TTest1(5,:)];
% Import it from Data saved using SimBio
load('Results_TMDD_dose.mat');
% % Calculate error.
% errT = norm(extractdata(TPred) - results(:,1)) / norm(results(:,1));
% errC = norm(extractdata(CPred) - results(:,2)) / norm(results(:,2));
% errD = norm(extractdata(DPred) - results(:,3)) / norm(results(:,3));
% Plot predictions.
tiledlayout(2,2)
nexttile
plot(results.Time(1:10:end),results.Data(1:10:end,1),'*r',tf,Tf,'-k')
legend("True: Target","Predicted: Target")
xlabel('time');
ylabel('Conc')
nexttile
plot(results.Time(1:10:end),results.Data(1:10:end,2),'*r',tf,Cf,'-k')
legend("True: Complex","Predicted: Complex", 'Location','Southwest')
xlabel('time');
ylabel('Conc')
nexttile([1 2])
plot(results.Time(1:10:end),results.Data(1:10:end,3),'*r',tf,Df,'-k')
legend("True: Drug","Predicted: Drug")
xlabel('time');
ylabel('Conc')
print('Figure_Dose','-djpeg')