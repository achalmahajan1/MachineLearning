function X = model(tspan,X0,neuralOdeParameters)

X = dlode45(@odeModel,tspan,X0,neuralOdeParameters,DataFormat="CB");

end