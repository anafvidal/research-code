%% Bayesian logistic regression with random effects (Experimet 4.3 in SOUL article [1])
% 
% In experiment we demonstrate the SOUL algorithm with a sparse
% Bayesian logistic regression with random effects problem. In
% this experiment p(y|θ ) is not log-concave (as it is in Experiment 4.1), 
% so SOUL can potentially get trapped in local maximisers.
% This experiment was previously considered by Atchadé et al. [2] and we 
% replicate their setup. For more information see [1].
% 
% [1] De Bortoli, A. Durmus, M. Pereyra and A. F. Vidal, 
% “Efficient stochastic optimisation by unadjusted Langevin Monte Carlo. 
% Application to maximum marginal likelihood and empirical Bayesian estimation”, 
% Statistics and Computing, 31(3), 1-18, Mar. 2021.
% 
% [2] Atchadé, Y.F., Fort, G., Moulines, E.: On perturbed proximal gradient
% algorithms. J. Mach. Learn. Res 18(1), 310–342, 2017.
%
% -----------------------------------------------------------------------
% Code Authors: Ana Fernandez Vidal - Marcelo Pereyra
% Date: June, 2019.
% -----------------------------------------------------------------------

clear all;clc;
% Set rnd generators (for repeatability)
randn('state',2);
rand('state',2);
%% Generate covariate matrix X:
% We generate N obervations each of dim p. We generate every column as the
% realization of an AR process (dim N). Run p steps of AR process. 

N   = 500;          % number of observations you need
p   =1000;      	% dimension of every observed covariate x_i
X   = 0*ones(N,p);  % vector to store covariates
rho = 0.8;          % set the value of rho (coefficient on y(t-1))

% In this code we mostly follow the notation from [2] which is different
% from the one used in [1]. Notice that what we here define as X and u, 
% is defined as V and x respectively in [1].

sigma = sqrt(1-rho^2);              
X (:,1)  = sigma*randn(N,1);        %Set initial point
mu_e  = 0;                          %Set the value of the mean of the error term                           

for t=2:p                        %Start the loop running from obs. 2 to p 
    X(:,t) = rho*X(:,t-1) + normrnd(mu_e, sigma, N, 1);    %The AR model
end

%Plot the series
figure;
plot(X(1,:));hold on;
plot(X(50,:));
plot(X(400,:));
title('Covariate matrix X');
xlabel('t')
ylabel('y(t)')

%% Generate true parameters Beta and Sigma
% Beta regression weights:
beta_true=1+4*rand(p,1);
numNullElem=ceil(p*98/100);
ii = sort(randsample(1:p,numNullElem));  
beta_true(ii)=0;

figure;plot(beta_true);title('True \beta');xlabel('i ');ylabel('\beta_i');

% Define sigma
sigma_true=0.1;

% Generate loading matrix for random effects
q=5; %dimension of the random effect
Z=0*ones(q,N);
id5=eye(5);
for i = 1:N
    Z(:,i) = id5(:,ceil(i*q/N)) ; 
end
%figure; imagesc(Z);

%% Lasso penalty for beta and sigma
alpha=1;%this means that the l2 norm component of the penalty term is eliminated.  
lambda=30;
c=2-alpha;
% Proximal operator of non-regular part (soft-thresholding) 
% The penalty is  (1-alpha)*0.5*||beta||² +alpha||beta||_1 
% Notice that scale c is equal to 1.
proxG = @(x,step) sign(x*c).*(max(abs(x*c),step*alpha*lambda*c)-step*alpha*lambda*c);

%% Define observation
s=@(p) (1+exp(-p)).^(-1);
u_true=randn(q,1);
pFun=@(u,beta,sigma)X*beta+sigma*Z'*u;
p_true=pFun(u_true,beta_true,sigma_true);
sp=s(p_true); %Vector with bernouli probs
y(500)=0; for i=1:N , y(i)=binornd(1,sp(i)); end;y=y';

%% MCMC MYULA sampler for U

%----------------------------------------------------------------------
% Parameters for SOUL algorithm
%----------------------------------------------------------------------
op.thinning = 1;     
op.warmupSteps = 10000;
op.samples=2e4; 
op.beta_init=ones(size(beta_true)); 
op.sigma_init=1;
op.delta.exp = 0.95 ;%Exponent of delta step for theta update
op.delta.scale = 1; %Scaling of delta step for theta update
op.gammaFrac=0.999;%Fraction of maximum allowed gamma that we will use
dimParameters=length(sigma_true)+length(beta_true); %total dimension of parameters to estimate

% Gradient of logPiu
gradLogPiu = @(u,beta,sigma) (sigma * Z * (y-s(pFun(u,beta,sigma)))-u);
% We use this scalar summary to monitor convergence:
lc=@(u,beta,sigma)  sum(y.*(pFun(u,beta,sigma))-log(1+exp(pFun(u,beta,sigma))));
logPiu= @(u,beta,sigma) lc(u,beta,sigma)-0.5* u'*u;

gradBetaSigma=@(u,beta,sigma)[u'*Z;X']*(y-s(pFun(u,beta,sigma))); %gradient is 1001x1. First element is sigma component. 1000 last are beta components. 

% Setup
dimU=numel(u_true);
total_iter=op.samples*op.thinning;    
scalingDeltaMatrix=eye(dimParameters);
delta = @(i) op.delta.scale*( (i^(-op.delta.exp)) )/dimU;
evMax=(N*1.25);% Maximum eigenvalue of operator A.
Lf = @(sigma) (evMax/sigma)^2;%Lip const. 
gammaBound = @(sigma) 1/(Lf(sigma));%max possible gamma
gamma=1e-2;%op.gammaFrac*gammaBound(sigma);
op.gamma=gamma;%set gamma after calculating sigma. 

tic;

%----------------------------------------------------------------------
% Warm-up for SOUL with fixed parameter theta (i.e. fixed beta and sigma)
%----------------------------------------------------------------------
fix_beta=op.beta_init; 
fix_sigma=op.sigma_init; 
logPiTrace_WU(op.warmupSteps)=0;%logPiTrace for warm up iterations
ulog_wu(5,op.warmupSteps)=0;

% Run single MCMC chain sampling from the posterior of U: 
Uwu = 0*randn(q,1);
fprintf("Running Warm up     \n");
for ii = 2:op.warmupSteps
    Uwu =  Uwu +gamma*gradLogPiu(Uwu,fix_beta,fix_sigma) + sqrt(2*gamma)*randn(size(Uwu));
%     Uwu =  Uwu +gamma*gradLogPiu(Uwu,fix_beta,fix_sigma) ;
    logPiTrace_WU(ii) = logPiu(Uwu,fix_beta,fix_sigma);
    ulog_wu(:,ii)=Uwu;
    %Display Progress
    if mod(ii, round(op.warmupSteps / 100)) == 0
        fprintf("\b\b\b\b%2d%%\n", round(ii / (op.warmupSteps) * 100));
    end
end

%----------------------------------------------------------------------
% Run SOUL algorithm to estimate theta
%----------------------------------------------------------------------
theta(dimParameters,total_iter)=0;%theta=[sigma;beta]=[sigma;b1;b2;...;b1000] dim=1001x1
theta(:,1)=[fix_sigma;fix_beta];

logPiTraceX(total_iter)=0;%posterior
ulog(5,total_iter)=0;
betaNorm(total_iter)=0;
U = Uwu;
fprintf('\nRunning SOUL to estimate theta    \n');
for ii = 2:total_iter    
                
    % Sample from posterior U:
    U =  U +gamma*gradLogPiu(U,theta(2:end,ii-1),theta(1,ii-1)) + sqrt(2*gamma)*randn(size(U));

    % Update theta estimation using U 
    thetaii = theta(:,ii-1) + delta(ii)*(gradBetaSigma(U,theta(2:end,ii-1),theta(1,ii-1))   );
    
    %Apply prox step on theta 
    theta(1,ii) = max(thetaii(1),1e-5);%Project sigma on positive subset
    theta(2:end,ii)=thetaii(2:end)-theta(2:end,ii-1) + proxG(theta(2:end,ii-1),delta(ii));  %Soft thresholding (or elastic) for beta. 
    
    %Save current state
    logPiTraceX(ii) = logPiu(U,theta(2:end,ii),theta(1,ii));%to monitor convergence
    ulog(:,ii)=U;
    betaNorm(ii)=sum(abs(theta(2:end,ii)));
    %Display Progress
    if mod(ii, round(total_iter / 100)) == 0
        fprintf('\b\b\b\b%2d%%\n', round(ii / (total_iter) * 100));
    end
end

time=toc;

burnin=600;
deltas=arrayfun(delta,burnin:total_iter);
delta_theta=theta(:,burnin:total_iter).*deltas;
weightedMeanValues=cumsum(delta_theta,2)./cumsum(deltas);

%----------------------------------------------------------------------
% Paper plots
%----------------------------------------------------------------------

% Defaults for plots
width = 3;     % Width in inches
height = 2.8;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 15;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

%-------------------------------
% Sigma evolution - Fig. 6 paper

fig=figure;  
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

plot(weightedMeanValues(1,:),'b','LineWidth',lw,'MarkerSize',msz);
axis([0 inf  0 1])
hold on; 
fplot(sigma_true, 'r--','LineWidth',lw,'MarkerSize',msz);
grid on;
h=legend('$\hat{\sigma}_{n}$','$\sigma_{true}$')
set(h,'Interpreter','latex','fontsize',12)
xlabel('Iteration (n)')
ylabel('$\hat{\sigma}_{n}$','Interpreter','latex')
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

%-------------------------------
% L0 norm of beta - Fig. 6 paper
betaa=weightedMeanValues(2:end,:);
betaa(abs(betaa)<0.005)=0;
fun=@(ii) sum(betaa(2:end,ii)~=0);
rangex=1:length(betaa);
yrange=fun(rangex);

fig=figure;  
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

semilogy(rangex, yrange,'b','LineWidth',lw,'MarkerSize',msz);
axis([0 inf  10 1000])
hold on; 
fplot(2/100*p, 'r--','LineWidth',lw,'MarkerSize',msz);
grid on;
h=legend('$\|\hat{\beta}_n\|_0$','$\|\beta_{true}\|_0$')
set(h,'Interpreter','latex','fontsize',12)
xlabel('Iteration (n)')
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

%-----------------------------
% Support beta - Fig. 7 paper
avgBetaTotal=sum(theta(2:end,:),2)/(length(theta(2,:)));
fig=figure;
sc=0.5;
betaa=weightedMeanValues(2:end,end);
betaa(abs(betaa)<0.006)=0;

stem((beta_true~=0)*sc,'Marker','none','LineWidth',lw);
hold on;
stem(-sc*(betaa~=0),'Marker','none','LineWidth',lw);
axis([0 inf -sc sc])
ylabel('$\hat{\beta}_{N}    \qquad \beta_{true}$','Interpreter','latex');
xlabel('Component index $i$','Interpreter','latex');
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

%--------------------------------------------
% Plots to monitor convergence (not in paper)
figure;
plot(betaNorm); legend('||\beta||_1');xlabel('iter');ylabel('||\beta||_1');title('\beta through iterations');
figure;
plot(logPiTraceX); legend('logPi+const.');xlabel('iter');ylabel('logpi(u)');title('LogPi(U) up to a constant');

