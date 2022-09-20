function plotTrueAndPredictedSolutions(t, xTrue,xPred)

xPred = squeeze(xPred)';

err = mean(abs(xTrue(2:end,:) - xPred), "all");

plot(t,xTrue(:,1),'-k',t,xTrue(:,2),'-b',t,xTrue(:,3),"-r",LineWidth=1)
hold on
plot(t(1:105:end),xPred(1:105:end,1),'*k',t(1:105:end),xPred(1:105:end,2),'*b',t(1:105:end),xPred(1:105:end,3),'*r',LineWidth=1)

title("Absolute Error = " + num2str(err,"%.4f"))
xlabel("x(1)")
ylabel("x(2)")

% xlim([-2 3])
% ylim([-2 3])

legend("Ground Truth","Predicted")
hold off
end