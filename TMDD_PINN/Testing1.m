%% Testing sbiosampleerror
% clear all;clc;close all;
sbioloadproject radiodecay;
%Simulate the model.

[t,sd,names] = sbiosimulate(m1);
%Plot the simulation results.

plot(t,sd);
legend(names,'AutoUpdate','off');
hold on

sdNoisy = sbiosampleerror(sd,'constant',20);
% Plot the noisy simulation data.

plot(t,sdNoisy);
hold off
%%
figure()
em = @(y,p1,p2) y+p1+p2*randn(size(y));
t = Data.Time;
sdNoisy1 = sbiosampleerror(Data.Drug_nM,em,{0.5,2});
plot(t,sdNoisy1,t,Data.Drug_nM);