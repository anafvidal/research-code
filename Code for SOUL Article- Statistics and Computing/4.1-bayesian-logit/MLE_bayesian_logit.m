%% Bayesian logistic regression  (Experimet 4.1 in SOUL article [1])
% 
% In experiment we demonstrate the SOUL algorithm with an empirical 
% Bayesian logistic regression problem for the Wisconsin Diagnostic Breast 
% Cancer dataset [2]. For more information see [1].
% 
% [1] De Bortoli, A. Durmus, M. Pereyra and A. F. Vidal, 
% “Efficient stochastic optimisation by unadjusted Langevin Monte Carlo. 
% Application to maximum marginal likelihood and empirical Bayesian estimation”, 
% Statistics and Computing, 31(3), 1-18, Mar. 2021.
% 
% [2] Available online: https://archive.ics.uci.edu/ml/datasets/Breast+
% Cancer+Wisconsin+(Diagnostic)
% -----------------------------------------------------------------------
% Code Authors: Ana Fernandez Vidal - Marcelo Pereyra
% Date: June, 2019.
% -----------------------------------------------------------------------
addpath ./bayesian-logit
addpath ./bayesian-logit/cancer-dataset

clear all;
clc;
% Set rnd generators (for repeatability)
randn('state',2);
rand('state',2);
%----------------------------------------------------------------------
% Load Wisconsin Breast Cancer Dataset
%----------------------------------------------------------------------
load('./bayesian-logit/cancer-dataset/cleanData.mat')

%----------------------------------------------------------------------
% Define covariate matrix X and split train and test data
%----------------------------------------------------------------------

numFeatures=10;
Xorig  = dataset(:,1:numFeatures);    %covariates

numTotSamples = length(Xorig); %number of observations (d_y)
percentTrain = 0.80 ; % 80% training samples
idx = randperm(numTotSamples);
Xtrain = Xorig(idx(1:round(numTotSamples*percentTrain)),:);
Xtest = Xorig(idx(round(numTotSamples*percentTrain)+1:end),:);
yOrig=dataset(:,numFeatures+1);
yTrain=yOrig(idx(1:round(numTotSamples*percentTrain)));
yTest=yOrig(idx(round(numTotSamples*percentTrain)+1:end));

%Normalize the covariates
Xnorm=0*Xtrain;
for i=1:numFeatures
    Xnorm(:,i)=(Xtrain(:,i)-mean(Xtrain(:,i)))/std(Xtrain(:,i));
end
% X=Xnorm;
X=Xnorm;
num_train_samp   =length(X);          %number of observations used for training

% We assume the regression coefficients beta follow a Gaussian prior
% distribution with variance=25 and unknown mean theta. 
op.sigma = 5;         %Deviation of the prior of beta. 

% We want to estimate theta using the SOUL algorithm. 


%----------------------------------------------------------------------
% Parameters for SOUL algorithm
%----------------------------------------------------------------------
op.thinning = 1;     
op.warmupSteps = 100;
op.samples=1e6; %note: the histograms in [1] were computed with 2e6 samples
op.theta_init=0;
op.delta_exp =  0.8;%Exponent of delta step for theta update
op.delta_scale = 600; %10;%Scaling of delta step for theta update
op.gammaFrac=2;%Fraction of maximum allowed gamma that we will use
dimParameters=1; %total dimension of parameters to estimate only theta CHECK if this should be 24. 
op.simName='JRSSB_';

% Compute bound for Lipschitz constant
sumX=0;
for i=1:num_train_samp
    sumX=sumX+norm(X(i,:))^2;
end
Lf=1/4+sumX+1/op.sigma^2
gammaBound =  1/(Lf);%max possible gamma
gamma=op.gammaFrac*gammaBound;%1e-2; %

%Define projection interval for theta
op.theta_min=-100;
op.theta_max=100;

% Setup
dimBeta=numFeatures;
total_iter=op.samples*op.thinning;    
delta = @(i) op.delta_scale*( (i^(-op.delta_exp)) )/dimBeta;

%----------------------------------------------------------------------
% Define functions related to Bayesian model
%----------------------------------------------------------------------
s=@(p) (1+exp(-p)).^(-1);
proj=@(x,x_min,x_max) min(x_max,max(x_min,x));

% Gradient of logPiBeta 
gradLogPibeta = @(beta,theta) (X'*(yTrain-s(X*beta)))-(beta-theta)/op.sigma^2;

% Function equal to LogPiBeta up to a constant:
lc=@(beta)  sum(yTrain.*(X*beta)-log(1+exp(X*beta)));
logPibeta= @(beta,theta) lc(beta)- norm(beta-theta)^2/op.sigma^2;

%Stochastic gradient of the likelihood
gradTheta1D=@(beta,theta) ((beta-theta)/op.sigma^2)'*ones(numFeatures,1);

%----------------------------------------------------------------------
% Warm-up for SOUL with fixed parameter theta
%----------------------------------------------------------------------
fix_theta=op.theta_init; 
betalog_wu(numFeatures,op.warmupSteps)=0;
% Run single MCMC chain sampling from the posterior of Beta: 
Beta_wu = randn(numFeatures,1);
fprintf("Running Warm up     \n");
for ii = 2:op.warmupSteps
    Beta_wu =  Beta_wu +gamma*gradLogPibeta(Beta_wu,fix_theta) + sqrt(2*gamma)*randn(size(Beta_wu));
    betalog_wu(:,ii)=Beta_wu;
    %Display Progress
    if mod(ii, round(op.warmupSteps / 100)) == 0
        fprintf("\b\b\b\b%2d%%\n", round(ii / (op.warmupSteps) * 100));
    end
end

%----------------------------------------------------------------------
% Run SOUL algorithm to estimate theta
%----------------------------------------------------------------------
tic;
theta(dimParameters,total_iter)=0;
betalog(numFeatures,total_iter)=0;
B = Beta_wu;%initialise chain with last sample from warm-up phase
fprintf("Running SOUL to estimate theta    \n");
for ii = 2:total_iter    
                
    % Sample from posterior of Beta:
    B =  B +gamma*gradLogPibeta(B,theta(ii-1)) + sqrt(2*gamma)*randn(size(B));
   
    %Update estimated parameter with stochastic gradient
    thetaii = theta(ii-1) + delta(ii)*gradTheta1D(B,theta(ii-1));
    
    %Apply projection step to theta 
    theta(ii)=proj(thetaii,op.theta_min,op.theta_max); 
    
    %Save current state 
    betalog(:,ii)=B;

    %Display Progress
    if mod(ii, round(total_iter / 100)) == 0
        fprintf('\b\b\b\b%2d%%\n', round(ii / (total_iter) * 100));
    end
end

results.execTime=toc;
results.betaMean=mean(betalog');
results.x_orig=Xtrain;
results.x_norm=Xnorm;
results.y=yTrain;
results.options=op;
results.betalog=betalog;
results.thetas=theta;

%----------------------------------------------------------------------
% Paper plots 
%----------------------------------------------------------------------
burnin=20000;
deltas=arrayfun(delta,burnin:total_iter);
delta_theta=theta(burnin:total_iter).*deltas;
weightedMeanValues=cumsum(delta_theta,2)./cumsum(deltas);

% plot defaults 
width = 3;     % Width in inches
height = 2.8;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 15;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

%---------------------------------------------------
% Plot evolution of estimated theta with iterations
trueMaxLikTheta=0.90927;%Computed with another script. For reference.
burnin=50;
dimBeta=numel(results.betaMean);
total_iter=op.samples;
delta = @(i) op.delta_scale*( (i^(-op.delta_exp)) )/dimBeta;
deltas=arrayfun(delta,burnin:total_iter);
delta_theta=results.thetas(burnin:total_iter).*deltas;
weightedMeanValues=cumsum(delta_theta,2)./cumsum(deltas);

figTheta=figure;
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) 507, 310]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

semilogx(results.thetas, 'color',[0.9290, 0.6940, 0.1250]); hold on;
startXaxis=burnin+5;
 axis([0 inf -100 100 ])
xlim([1 100])
grid on;grid minor; grid minor;
xlabel('Iteration (n)')
ylabel('$\hat{\theta}_{n}$','Interpreter','latex');
semilogx( burnin:total_iter,weightedMeanValues,'b','LineWidth',lw,'MarkerSize',msz);
fplot(trueMaxLikTheta,'r--','LineWidth',lw,'MarkerSize',msz);
hold off; 
h=legend('$\theta_{n}$','$\hat{\theta}_{n}$','$\theta^*$');
set(h,'Interpreter','latex','fontsize',fsz);


%----------------------------------
%Plot histograms for all components

%Note: this script is using op.samples=1e6, but the histograms in the paper
%were computed  with op.samples=2e6 for a better approximation; 
scrsz = get(0,'ScreenSize');
figHistograms12=figure('Position',[1 1 scrsz(3)/1.2 scrsz(4)/2])    
title('Histogram for different components of \beta');
ax1=subplot(2,5,1)
    histogram(betalog(1,:),'DisplayStyle','stairs','Normalization','pdf','LineWidth',lw)
    title('\beta_1')  
ax2=subplot(2,5,2)     
    histogram(betalog(2,:),'DisplayStyle','stairs','Normalization','pdf','LineWidth',lw)
    title('\beta_2') 
ax3=subplot(2,5,3)
    histogram(betalog(3,:),'DisplayStyle','stairs','Normalization','pdf','LineWidth',lw)
    title('\beta_3')   
ax4=subplot(2,5,4)
    histogram(betalog(4,:),'DisplayStyle','stairs','Normalization','pdf','LineWidth',lw)   
    title('\beta_4') 
ax5=subplot(2,5,5)
    histogram(betalog(5,:),'DisplayStyle','stairs','Normalization','pdf','LineWidth',lw)
    title('\beta_5')   
ax6=subplot(2,5,6)
    histogram(betalog(6,:),'DisplayStyle','stairs','Normalization','pdf','LineWidth',lw)
    title('\beta_6')   
ax7=subplot(2,5,7)
    histogram(betalog(7,:),'DisplayStyle','stairs','Normalization','pdf','LineWidth',lw)
   title('\beta_7')   
ax8=subplot(2,5,8)
    histogram(betalog(8,:),'DisplayStyle','stairs','Normalization','pdf','LineWidth',lw)
    title('\beta_8')  
ax9=subplot(2,5,9)
    histogram(betalog(9,:),'DisplayStyle','stairs','Normalization','pdf','LineWidth',lw)
    title('\beta_{9}')    
ax10=subplot(2,5,10)
    histogram(betalog(10,:),'DisplayStyle','stairs','Normalization','pdf','LineWidth',lw)
    title('\beta_{10}')       

    subplots = get(figHistograms12,'Children'); % Get each subplot in the figure
for i=1:length(subplots) % for each subplot
    caxis(subplots(i),[0,1]); 
end

%-------------------------------------------------------
% Plot detailed histogram for the 5th component of \beta
figHistogram=figure;
histogram(betalog(5,:),'DisplayStyle','stairs'); title('Histogram for the 5th component of \beta');

%----------------------------------------------------------------------
% Test estimated beta
%----------------------------------------------------------------------
betaEstim=mean(betalog')';
XtestNorm=0*Xtest;
for i=1:numFeatures
    XtestNorm(:,i)=(Xtest(:,i)-mean(Xtrain(:,i)))/std(Xtrain(:,i));
end

yEstim=(s(XtestNorm*betaEstim)>0.5);
errorPercent=sum(abs(yEstim-yTest))/length(yTest)*100;% =2.1898 percent errors 
fprintf('\n Error percent: %2.2f ',errorPercent);
results.errorPercent=errorPercent;

%----------------------------------------------------------------------
% Save results
%----------------------------------------------------------------------
dirname = 'results';
subdirname='bayesian-logit';
if ~exist(['./' dirname],'dir')
    mkdir('results');
end
if ~exist(['./' dirname '/' subdirname ],'dir')
    mkdir(['./' dirname '/' subdirname ]);
end
save(['./' dirname '/' subdirname '/results.mat'], '-struct','results');
%close all;