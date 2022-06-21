clear all;
results_likelihood=load('/media/data/Dropbox/PhD Heriot-Watt/Codigo Ana/JournalAlain/results/germanLogit/true_likelihood/MultipleThetas__sigma5_300000WU_thinning1_600000samp_dScale600_dExp0.8_g0.00029324_Lf6820.29/results_manyThetasCentered_61.mat');
%res_likeli_0_200=load('/media/data/Dropbox/PhD Heriot-Watt/Codigo Ana/JournalAlain/results/germanLogit/true_likelihood/MultipleThetas__sigma5_300000WU_thinning1_600000samp_dScale600_dExp0.8_g0.00029324_Lf6820.29/results_manyThetasCentered_0to200_31.mat');
results_manyrunsEB=load('/media/data/Dropbox/PhD Heriot-Watt/Codigo Ana/JournalAlain/results/germanLogit/many_runs_EB_var/JRSSB__sigma5_20000WU_thinning1_300000samp_dScale600_dExp0.8_gFrac2/results_numRuns_1000.mat');
results_EB_estim=load('/media/data/Dropbox/PhD Heriot-Watt/Codigo Ana/JournalAlain/results/germanLogit/EB_1MCMC_LOGIT/JRSSB_singleTheta_thInit0sigma5_20000WU_thinning1_2000000samp_dScale600_dExp0.8_g0.00036695_Lf5450.29/results.mat');

%% Compute true maximiser
% A=-(smooth(log(results_likelihood.margLikEstim)))
% minIndex=find(A==min(A));
% trueMaxLikTheta=results_likelihood.thetas(minIndex);
p = polyfit(results_likelihood.thetas,(log(results_likelihood.margLikEstim)),2);
trueMaxLikTheta=-p(2)/(2*p(1)); 
%%
% Defaults for this blog post
    width = 3;     % Width in inches
    height = 2.8;    % Height in inches
    alw = 0.75;    % AxesLineWidth
    fsz = 15;      % Fontsize
    lw = 1.5;      % LineWidth
    msz = 8;       % MarkerSize
    
    figMargLikJoined=figure;
        
    hax=axes; 
    semilogy(results_likelihood.thetas,(exp(smooth(log(results_likelihood.margLikEstim)))),'color','black','MarkerSize',msz);hold on;  
    hold on;
    grid on;
    ylabel('$\hat{p}(y|\theta)$','Interpreter','latex')
    xlabel('\theta')
    
    pos = get(gcf, 'Position');
    set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
    set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

%plot histogram
true_th=trueMaxLikTheta; 
histogram(results_manyrunsEB.thetas_weighted,20,'Normalization','pdf','EdgeColor','none','FaceColor',[0 0 1]);%'FaceColor',[0.85 0.85 0.9]
line([true_th true_th],get(hax,'YLim'),'Color',[1 0 0],'LineStyle','--','LineWidth',1.5);


% histogram(results_manyrunsEB.thetas_mean,20,'Normalization','pdf');
% histogram(results_manyrunsEB.thetas_last,20,'Normalization','pdf');

h=legend('$\hat{p}(y|\theta)$','$\hat{\theta}_{N}$ histogram ','$\theta^*$');
set(h,'Interpreter','latex','fontsize',fsz)
axis([-4 5.943 .52387405341E-41 .26980107722E-36 ])



%% Cuadratic fit

    width = 3;     % Width in inches
    height = 2.8;    % Height in inches
    alw = 0.75;    % AxesLineWidth
    fsz = 13;      % Fontsize
    lw = 1.5;      % LineWidth
    msz = 8;       % MarkerSize
    
    
% plot(results_likelihood.thetas,polyval(p,results_likelihood.thetas),'MarkerSize',msz,'LineWidth',lw);hold on;
   
    figMargLikJoined=figure;
    hax=axes; 
    stem(results_likelihood.thetas,(log(results_likelihood.margLikEstim)),'.','color',[0,0,1],'LineStyle','none','MarkerSize',12);
    hold on;
    %set(gca,'yscal','log')
    plot(results_likelihood.thetas,(polyval(p,results_likelihood.thetas)),'color',[0.3,0.3,0.3],'MarkerSize',msz,'LineWidth',1);
    grid on;grid minor;grid minor;
    ylabel('$\log \hat{p}(y|\theta)$','Interpreter','latex','fontsize',fsz)
    xlabel('\theta','fontsize',fsz)    
    pos = get(gcf, 'Position');
    set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
    set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

%plot histogram
true_th=trueMaxLikTheta; 
%histogram(results_manyrunsEB.thetas_weighted,20,'Normalization','pdf','EdgeColor','none','FaceColor',[0 0 1]);%'FaceColor',[0.85 0.85 0.9]
line([true_th true_th],get(hax,'YLim'),'Color',[1 0 0],'LineStyle','--','LineWidth',2);
% semilogy(results_likelihood.thetas,exp(polyval(p,results_likelihood.thetas)),'MarkerSize',msz,'LineWidth',lw);hold on;


% histogram(results_manyrunsEB.thetas_mean,20,'Normalization','pdf');
% histogram(results_manyrunsEB.thetas_last,20,'Normalization','pdf');

h=legend('$\log \hat{p}(y|\theta)$','Cuadratic fit ','$\theta^*$');
set(h,'Interpreter','latex','fontsize',fsz)
%axis([-4 5.943 .52387405341E-41 .26980107722E-36 ]);

%% Histograms of different EB estimators (mean , last , weighted)
 
hisplot=figure;
hax=axes;
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
%plot histogram
histogram(results_manyrunsEB.thetas_weighted,20,'Normalization','pdf','FaceColor',[0 0 1]);%'FaceColor',[0.85 0.85 0.9]
hold on;
% histogram(results_manyrunsEB.thetas_mean,20,'Normalization','pdf');
% histogram(results_manyrunsEB.thetas_last,20,'Normalization','pdf');
xlabel('\theta','fontsize',fsz);
ylabel('Estimated $\hat{\theta}_{N}$ histogram','Interpreter','latex','fontsize',fsz);
xlim([0.81 0.96])
%title(['bias=' char(num2str(bias,2)) '      std=' char(num2str(stdev,2)) ]);

%plot line true theta
true_th=trueMaxLikTheta; %your point goes here 
line([true_th true_th],get(hax,'YLim'),'Color',[1 0 0],'LineStyle','--','LineWidth',1.5);
%plot gaussian fit
% x = [0.96:.0005:1.08];
% norm = normpdf(x,1-bias,stdev);
% plot(x,norm,'b','Color',[0 0 1])
h=legend('$\hat{\theta}_{N}$ ','$\theta^*$');
set(h,'Interpreter','latex','fontsize',fsz)


%%


% Defaults for this blog post
width = 3;     % Width in inches
height = 2.8;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 15;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize



% THETA FOR PAPER

burnin=50;
op=results_EB_estim.options;dimBeta=numel(results_EB_estim.betaMean);
total_iter=op.samples;
delta = @(i) op.delta_scale*( (i^(-op.delta_exp)) )/dimBeta;
deltas=arrayfun(delta,burnin:total_iter);
delta_theta=results_EB_estim.thetas(burnin:total_iter).*deltas;
weightedMeanValues=cumsum(delta_theta,2)./cumsum(deltas);

figTheta=figure;
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

semilogx(results_EB_estim.thetas, 'color',[0.9290, 0.6940, 0.1250]); hold on;
startXaxis=burnin+5;
 axis([0 inf -1 1.3 ])
xlim([1 1e4])
grid on;grid minor; grid minor;
xlabel('Iteration (n)')
ylabel('$\hat{\theta}_{n}$','Interpreter','latex');
%title('\theta through iterations');hold on;
%semilogy(cumsum(theta)./(1:total_iter),'color',[0,0.8,0.3],'LineWidth',1.5); 
semilogx( burnin:total_iter,weightedMeanValues,'b','LineWidth',lw,'MarkerSize',msz);
fplot(trueMaxLikTheta,'r--','LineWidth',lw,'MarkerSize',msz);
hold off; 
% h=legend('$\theta^{(t)}$','$\overline{\theta}^{(t)}$','$\hat{\theta}_{t}$');
h=legend('$\theta^{(n)}$','$\hat{\theta}_{n}$','$\theta^*$');
set(h,'Interpreter','latex','fontsize',fsz);



fig = figure(1);
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

%% Evolution of likelihood with thetas
figLikeliEvol=figure;
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

likelihoodEvolution=real(spline(results_likelihood.thetas,(exp(smooth(log(results_likelihood.margLikEstim)))),results_EB_estim.thetas));
likelihoodEvolutionFit=exp(polyval(p,results_EB_estim.thetas));
plot(real(-log(likelihoodEvolution)),'LineWidth',lw,'MarkerSize',msz);hold on;
plot(real(-log(likelihoodEvolutionFit)),'LineWidth',lw,'MarkerSize',msz);
minLogLik=min(-(smooth(log(results_likelihood.margLikEstim))));
fplot(minLogLik,'--','LineWidth',lw,'MarkerSize',msz);
grid on;
axis([1 inf 84 85 ])
xlabel('Iteration (n)')
ylabel('$-\log p(y|\theta)$','Interpreter','latex');
h=legend('$-\log p(y|\theta^{(n)})$','fit','$ min -\log p(y|\theta) $');
set(h,'Interpreter','latex','fontsize',12);

%% Evolution of likelihood with thetas for weighted average

figLikeliEvolWeighted=figure;
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

%likelihoodEvolution=real(spline(results_likelihood.thetas,(exp(smooth(log(results_likelihood.margLikEstim)))),weightedMeanValues));
likelihoodEvolutionFit=exp(polyval(p,weightedMeanValues));
likelihoodEvolutionFitTheta=exp(polyval(p,results_EB_estim.thetas));

plot(log(likelihoodEvolutionFitTheta),'color',[0.9290, 0.6940, 0.1250]); hold on;
plot(burnin:total_iter,log(likelihoodEvolutionFit),'b','LineWidth',lw,'MarkerSize',msz); hold on;
%plot(real(log(likelihoodEvolutionFit)),'LineWidth',lw,'MarkerSize',msz); 
minLogLik=polyval(p,trueMaxLikTheta)%max((smooth(log(results_likelihood.margLikEstim))));

fplot(minLogLik,'r--','LineWidth',lw,'MarkerSize',msz);
grid on;
axis([1 inf  -84.64  -84.58])
xlabel('Iteration (n)')
ylabel('$\log \hat{p}(y|\theta)$','Interpreter','latex');
h=legend('$\log \hat{p}(y|\theta^{(n)})$','$\log \hat{p}(y|\hat{\theta}_{n})$','sup $\log \hat{p}(y|\theta) $');
set(h,'Interpreter','latex','fontsize',12);
%%
fig = figure(2);
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];