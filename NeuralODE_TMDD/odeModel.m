function y = odeModel(~,y,theta)

y = tanh(theta.fc1.Weights*y + theta.fc1.Bias);
y = tanh(theta.fc2.Weights*y + theta.fc2.Bias);
y = tanh(theta.fc3.Weights*y + theta.fc3.Bias);
y = tanh(theta.fc4.Weights*y + theta.fc4.Bias);
%y = tanh(theta.fc5.Weights*y + theta.fc5.Bias);
%y = tanh(theta.fc6.Weights*y + theta.fc6.Bias);
y = theta.fc5.Weights*y + theta.fc5.Bias;

end