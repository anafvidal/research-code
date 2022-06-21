clear all;clc;
load('./bayesian-logit/cancer-dataset/cleanData.mat')
% Set rnd generators (for repeatability)
randn('state',2);
rand('state',2);
%% Generate covariate matrix X and split train and test data:
% We generate N obervations each of dim p. We generate every column as the
% realization of an AR process (dim N). Run p steps of AR process. 
numFeatures=10;
Xorig  = dataset(:,1:numFeatures);    %covariates

numTotSamples = length(Xorig);
percentTrain = 0.80 ; results.percentTrain=percentTrain;
idx = randperm(numTotSamples);
Xtrain = Xorig(idx(1:round(numTotSamples*percentTrain)),:);
Xtest = Xorig(idx(round(numTotSamples*percentTrain)+1:end),:);
yOrig=dataset(:,numFeatures+1);
yTrain=yOrig(idx(1:round(numTotSamples*percentTrain)));
yTest=yOrig(idx(round(numTotSamples*percentTrain)+1:end));
%normalize
Xnorm=0*Xtrain;
for i=1:numFeatures
    Xnorm(:,i)=(Xtrain(:,i)-mean(Xtrain(:,i)))/std(Xtrain(:,i));
end
% X=Xnorm;
X=Xnorm;
N   =length(X);          %number of observations you need
p   = length(X(1,:));    % dimension of every observed covariate x_i

op.sigma = 1000000;              %check if this is sigma or sigma²


%Plot the series
figure;
plot(Xtrain(1,:));hold on;
plot(Xtrain(2,:));
plot(Xtrain(3,:));
plot(Xtrain(4,:));
plot(Xtrain(5,:));
plot(Xtrain(6,:));
title('6 samples of X');
xlabel('i')
ylabel('X(i)')


%Plot the series
figure;
plot(Xnorm(1,:));hold on;
plot(Xnorm(2,:));
plot(Xnorm(3,:));
plot(Xnorm(4,:));
plot(Xnorm(5,:));
plot(Xnorm(6,:));
title('6 samples of Xnorm');
xlabel('i')
ylabel('Xnorm(i)')
%% Lasso penalty for beta and sigma
% alpha=1;%the paper said 1 but it might be a typo CHECK becuase it means
% %the l2 norm term is eliminated.  
% lambda=30;
% c=2-alpha;
% % Proximal operator of non-regular part (soft-thresholding) 
% % The penalty is  (1-alpha)*0.5*||beta||² +alpha||beta||_1 when computing
% % equivalence for quadratic term, scale c is equal to 1 so nothing changes.
% proxG = @(x,step) sign(x*c).*(max(abs(x*c),step*alpha*lambda*c)-step*alpha*lambda*c);
% %to-do marcelo said we should CHECK if alpha=1 was a typo. becuase it means
% %the l2 norm term is eliminated. 

% % We use this scalar summary to monitor convergence
% logPi = @(xw,theta) -(norm(y-A(xw),'fro')^2)/(2*sigma2) -theta*sum(abs(xw(:)));
%% MCMC MYULA sampler for Beta
%Parameters for EB algorithm
op.thinning = 1;     
op.warmupSteps = 3000;
op.samples=2e6; 
op.theta_init=0;%zeros(numFeatures,1);  %we start with 1 dimensional theta
op.delta_exp =  0.8;%Exponent of delta step for theta update
op.delta_scale = 600; %10;%Scaling of delta step for theta update
op.gammaFrac=2;%Fraction of maximum allowed gamma that we will use
dimParameters=1; %total dimension of parameters to estimate only theta CHECK if this should be 24. 
op.simName='BayesLowSamp_';
%% Define functions and gradients
s=@(p) (1+exp(-p)).^(-1);
proj=@(x,x_min,x_max) min(x_max,max(x_min,x));
% Gradient of logPiBeta
gradLogPibeta = @(beta,theta) (X' * (yTrain-s(X*beta))) - (beta-theta)/op.sigma^2;
betaTest=ones(numFeatures,1);%DEBUG

% Something equal to LogPiBeta up to a constant:
lc=@(beta)  sum(yTrain.*(X*beta)-log(1+exp(X*beta)));
logPibeta= @(beta,theta) lc(beta)- norm(beta-theta)^2/op.sigma^2;

%normdist=@(u,mu,sig)1/sqrt((2*pi*sig^2)^length(u))*exp(-(u-mu)'*(u-mu)/(2*sig^2));
%normdist2=@(u,mu,sig)exp(-(u-mu)'*(u-mu)/(2*sig^2));
gradTheta=@(beta,theta) (theta-beta)/op.sigma^2; %gradient is 1x1. 
gradTheta1D=@(beta,theta) ((beta-theta)/op.sigma^2)'*ones(numFeatures,1); %gradient is 1x1. 
gradTheta1Dv2=@(beta,theta) -((beta-theta)/op.sigma^2);
% Setup
dimBeta=numFeatures;
total_iter=op.samples*op.thinning;    

% scalingDeltaMatrix=scalingDeltaMatrix/length(beta_true);%scale for beta. 
% scalingDeltaMatrix(1,1)=1;%scale for sigma
% delta = @(i) scalingDeltaMatrix*op.delta_scale*( (i^(-op.delta_exp)) ); %delta(i) for theta update. %CHECK which dimension we use to normalize. 
delta = @(i) op.delta_scale*( (i^(-op.delta_exp)) )/dimBeta;
%evMax=(N*1.25);% Maximum eigenvalue of operator A. 

% Compute Lf Valentin
sumX=0;
for i=1:N
    sumX=sumX+norm(X(i,:))^2;
end
Lf=1/4+sumX+1/op.sigma^2
%op.Lf = numFeatures*554 ; %this is a rough bound  %(evMax/sigma)^2;%Lip const. 
op.Lf=2.3976e+04;%Calculo cota valentin
gammaBound =  1/(op.Lf);%max possible gamma
gamma=op.gammaFrac*gammaBound;%1e-2; %
op.gamma=gamma;


op.theta_min=-5;
op.theta_max=5;
%% Run MCMC sampler with fix theta to warm up the chain
fix_theta=op.theta_init; 
logPiTrace_WU(op.warmupSteps)=0;%logPiTrace for warm up iterations
betalog_wu(numFeatures,op.warmupSteps)=0;
% Run single MCMC chain sampling from the posterior of Beta: 
Beta_wu = randn(numFeatures,1);%start beta with zero...maybe not good idea. 
fprintf("Running Warm up     \n");
for ii = 2:op.warmupSteps
    Beta_wu =  Beta_wu +gamma*gradLogPibeta(Beta_wu,fix_theta) + sqrt(2*gamma)*randn(size(Beta_wu));
%     Uwu =  Uwu +gamma*gradLogPiu(Uwu,fix_beta,fix_sigma) ;
    logPiTrace_WU(ii) = logPibeta(Beta_wu,fix_theta);
    betalog_wu(:,ii)=Beta_wu;
    %Display Progress
    if mod(ii, round(op.warmupSteps / 100)) == 0
        fprintf("\b\b\b\b%2d%%\n", round(ii / (op.warmupSteps) * 100));
    end
end
% Plot evolution of components of beta
close all;
figure;
results.logPiTrace_WU=logPiTrace_WU;
plot(betalog_wu');xlabel('iter');ylabel('\beta_i');title('Beta components during warm up');

figure;
plot(logPiTrace_WU); title('Log pi trace Warmup theta=0; ');
%% Run MCMC sampler after warm up
% Using two MCMC chains for estimating the normalization constant as well.
tic;
theta(dimParameters,total_iter)=0;
logPiTraceBeta(total_iter)=0;%posterior
betalog(numFeatures,total_iter)=0;
% Run sampler for Double MCMC case
B = Beta_wu;%posterior
close all; 
fprintf('\nRunning double MCMC chains    \n');
for ii = 2:total_iter    
                
    % a)Sample from posterior U:
    B =  B +gamma*gradLogPibeta(B,theta(ii-1)) + sqrt(2*gamma)*randn(size(B));

    %Update MCMC param estimation   
    thetaii =theta(ii-1) + delta(ii)*gradTheta1D(B,theta(ii-1));
    
    %Apply prox step on theta 
    theta(ii)=proj(thetaii,op.theta_min,op.theta_max);  %CHECH THIS!!!!!!!
    
    %Save current state 
    logPiTraceBeta(ii) = logPibeta(B,theta(ii));
    betalog(:,ii)=B;
    %betaNorm(ii)=sum(abs(theta(2:end,ii)));
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
%% Save data and plots
op.subdirID = [op.simName 'singleTheta_thInit' char(num2str(op.theta_init)) 'sigma' char(num2str(op.sigma)) '_' char(num2str(op.warmupSteps)) 'WU_thinning' char(num2str(op.thinning))  '_'  char(num2str(op.samples)) 'samp_dScale' char(num2str(op.delta_scale)) '_dExp' char(num2str(op.delta_exp)) '_g' char(num2str(op.gamma)) '_Lf' char(num2str(op.Lf))];
dirname = 'results';
subdirname=['germanLogit/EB_1MCMC_LOGIT/' op.subdirID];
if ~exist(['./' dirname],'dir')
    mkdir('results');
end
if ~exist(['./' dirname '/' subdirname ],'dir')
    mkdir(['./' dirname '/' subdirname ]);
end



% Plots:
figBetaEvolution=figure;
plot(betalog');xlabel('iter');ylabel('\beta_i');title('\beta components during theta estimation');
saveas(figBetaEvolution,['./' dirname '/' subdirname '/betaEvolComponent.png']);

figTheta=figure;
plot(theta); legend('\theta');xlabel('iter');ylabel('\theta');title('\theta through iterations');hold on;
plot(cumsum(theta)./(1:total_iter)); legend('Theta','Average Theta');
hold off; 
saveas(figTheta,['./' dirname '/' subdirname '/theta.png']);

figLogPiTrace=figure;
plot(logPiTraceBeta); legend('logPi+const.');xlabel('iter');ylabel('logpi(u)');title('LogPi(U) up to a constant');
saveas(figLogPiTrace,['./' dirname '/' subdirname '/logPiTrace.png']);

figLogPiTraceWU=figure; 
plot(logPiTrace_WU);title('LogPiTrace Warm-up');legend('logPiTrace WU');
saveas(figLogPiTraceWU,['./' dirname '/' subdirname '/LogPiTraceWU.png']);

figHistogram=figure;
histogram(betalog(5,:)); title('Histogram for the 5th component of \beta');
saveas(figHistogram,['./' dirname '/' subdirname '/figHistogram5.png']);

%%
%Plot histograms
 %Generate subplots with orig and recovered end members
    scrsz = get(0,'ScreenSize');
    figHistograms12=figure('Position',[1 1 scrsz(3)/1.2 scrsz(4)/2])    
    ax1=subplot(2,5,1)
        histogram(betalog(1,:))
        title('\beta_1')  
    ax2=subplot(2,5,2)     
        histogram(betalog(2,:))
        title('\beta_2') 
    ax3=subplot(2,5,3)
        histogram(betalog(3,:))
        title('\beta_3')   
    ax4=subplot(2,5,4)
        histogram(betalog(4,:))   
        title('\beta_4') 
    ax5=subplot(2,5,5)
        histogram(betalog(5,:))
        title('\beta_5')   
    ax6=subplot(2,5,6)
        histogram(betalog(6,:))
        title('\beta_6')   
    ax7=subplot(2,5,7)
        histogram(betalog(7,:))
       title('\beta_7')   
    ax8=subplot(2,5,8)
        histogram(betalog(8,:))
        title('\beta_8')  
    ax9=subplot(2,5,9)
        histogram(betalog(9,:))
        title('\beta_{9}')    
    ax10=subplot(2,5,10)
        histogram(betalog(10,:))
        title('\beta_{10}')       

        subplots = get(figHistograms12,'Children'); % Get each subplot in the figure
    for i=1:length(subplots) % for each subplot
        caxis(subplots(i),[0,1]); % set the clim
    end
title('Histogram for different components of \beta');
saveas(figHistograms12,['./' dirname '/' subdirname '/figHistograms12.png']);

%% Test estimated beta
betaLogReduced=betalog(:,1000000:end);
betaEstim=mean(betalog')';
betaEstimRed=mean(betaLogReduced')';
betaEstim=betaEstimRed;
XtestNorm=0*Xtest;
for i=1:numFeatures
    XtestNorm(:,i)=(Xtest(:,i)-mean(Xtrain(:,i)))/std(Xtrain(:,i));
end

yEstim=(s(XtestNorm*betaEstim)>0.5);
errorPercent=sum(abs(yEstim-yTest))/length(yTest)*100;%=29.5% of error....pretty bad
fprintf('\n Error percent: %2.2f ',errorPercent);
% Quick check
%1) percent of ones in yTrain...
(sum(abs(yTrain)))/length(yTrain)*100 ;%  =29.1% are 1 in train samples
%2) if I just estimate all zero...
sum(abs(yEstim*0-yTest))/length(yTest)*100 ;% if I estimate y=0, my error is of 33.5% because that is the % of 1 in the test data

results.errorPercent=errorPercent;

% Save data:
save(['./' dirname '/' subdirname '/results.mat'], '-struct','results');
close all;