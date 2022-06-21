%%  Statistical audio compression  (Experimet 4.2 in SOUL article [1])
% 
% We consider the audio compressive sensing demonstration presented in [2].
% We use the SOUL algorithm to estimate the regularisation parameter that
% controls the degree of sparsity enforced, and where a main difficulty is the high-dimensionality of the
% latent space ($d = 2,900$). For more information see [1].
% 
% 
% [1] De Bortoli, A. Durmus, M. Pereyra and A. F. Vidal, 
% “Efficient stochastic optimisation by unadjusted Langevin Monte Carlo. 
% Application to maximum marginal likelihood and empirical Bayesian estimation”, 
% Statistics and Computing, 31(3), 1-18, Mar. 2021.
% 
% [2] Balzano, L., Nowak, R. and Ellenberg, J. (2010) 
% Compressed sensing audio demonstration. 
% website http://web.eecs.umich.edu/~girasole/csaudio.
% -----------------------------------------------------------------------
% Based on the original demonstration by Balzano et. Al [2]  

% Code Authors: Ana Fernandez Vidal - Marcelo Pereyra
% Date: June, 2019.
% -----------------------------------------------------------------------
addpath ./audio-cs
addpath ./audio-cs/GPSR

clear all;
clc;
% Set rnd generators (for repeatability)
randn('state',1);
rand('state',1);

%----------------------------------------------------------------------
%  Load mary's song data
%----------------------------------------------------------------------
songName='mary';
load pianoBasis.mat
load marySong.mat

% Import z (mary had a little lamb midi song)
z=marySong;
n=numel(z);

%----------------------------------------------------------------------
% Generate observation vector 'y'
%----------------------------------------------------------------------
sigma =0.015;% noise intensity

% Here we create two measurement matrices which simulate various
% sampling operators. We either undersample by taking only 1200 random
% samples (that's less than 4% of the samples in the file)
k =456;%  456;                             % Number of random samples
ii = sort(randsample(1:n,k));               % index of samples
R_meas = sparse(1:k,ii,ones(size(ii)),k,n); % Measurement matrix
Ax=R_meas*marySong;
sigma2=sigma^2;
R = R_meas*pianoBasis';%R_meas is obs matrix. pianoBasis is our dictionary.
y = Ax+sigma*randn(size(Ax));%noisy observation

% These are input functions to GPSR, the code we're using for l1
% minimization.
A = @(x) R*x;%playes role of operator A.
AT = @(x) R'*x;

%----------------------------------------------------------------------
% Parameters for SOUL algorithm
%----------------------------------------------------------------------
%Fixed parameters
op.thinning = 1;
op.warmupSteps = 250;
op.N=1000; %(number of iterations of SOUL algo.)
op.delta.exp =  0.8;%Exponent of delta step for theta update
op.delta.scale = 20; %10;%Scaling of delta step for theta update
op.gammaFrac=0.999;%Fraction of maximum allowed gamma step that we will use
%(gamma is selected as a function of the Lipschitz constant of the gradient)
op.lambdaMax=2;%Maximum allowed smoothing parameter for huber function

% Automatically set parameters
dimFrame=length(pianoBasis(:,1));
evMax=max_eigenval(A, AT, [dimFrame,1],0.0001,200,0);% Maximum eigenvalue of operator A.
Lf = @(sigma) (evMax/sigma)^2;%Lipschitz constant
gammaUpBound = @(sigma,lambda) 1/(Lf(sigma)+(1/lambda));%max possible gamma
lambdaFun= @(sigma) min((5/Lf(sigma)),op.lambdaMax);
lambda=lambdaFun(sigma);
gamma=op.gammaFrac*gammaUpBound(sigma,lambda);%set gamma after calculating sigma.


op.th_init=0.1*max(abs(R'*y))/sigma^2;%Default theta for compressive sensing
op.min_th=1e-4/sigma2;
op.max_th=0.5/sigma2;
op.eta_init=log(op.th_init);
op.min_eta=log(op.min_th);
op.max_eta=log(op.max_th);

%----------------------------------------------------------------------
% Define functions related to Bayesian model
%----------------------------------------------------------------------
% Gradient of smooth part
gradF = @(xw) -real(AT(A(xw)-y)/sigma2);

% Proximal operator of non-regular part (soft-thresholding)
proxG = @(xw,lambda,theta) sign(xw).*(max(abs(xw),theta*lambda)-theta*lambda);

%----------------------------------------------------------------------
% Warm-up for SOUL with fixed parameter theta
%----------------------------------------------------------------------
tic;
[dimFrame,numSampOrig]=size(pianoBasis);
total_iter=op.N*op.thinning;
delta = @(i) op.delta.scale*( (i^(-op.delta.exp)) / (dimFrame) ); %delta(i) for proximal gradient algorithm
%We define theta=e^{eta} and use a logarithmic scale;
eta_init=log(op.th_init);
min_eta=log(op.min_th);
max_eta=log(op.max_th);
fix_theta=op.th_init;

% Run single MCMC chain sampling from the posterior of X:
Xwu = AT(y); %Initial value of X during warm up
fprintf("Running Warm up     \n");
for ii = 2:op.warmupSteps
    Xwu =  Xwu + gamma* (proxG(Xwu,lambda,fix_theta)-Xwu)/lambda +gamma*gradF(Xwu) + sqrt(2*gamma)*randn(size(Xwu));
    %Display Progress
    if mod(ii, round(op.warmupSteps / 100)) == 0
        fprintf("\b\b\b\b%2d%%\n", round(ii / (op.warmupSteps) * 100));
    end
end

%----------------------------------------------------------------------
% Run SOUL algorithm to estimate theta
%----------------------------------------------------------------------
theta(op.N)=0;
theta(1)=fix_theta;
eta(op.N)=0;
eta(1)=eta_init;

X =Xwu; %initialize with last sample from warm-up
fprintf("Running SOUL to estimate theta    \n");
for ii = 2:op.N
    %Sample from posterior of X:
    for iv=1:op.thinning
        X =  X + gamma* (proxG(X,lambda,theta(ii-1))-X)/lambda +gamma*gradF(X) + sqrt(2*gamma)*randn(size(X));
    end
    
    %Update parameter
    etaii = eta(ii-1) + delta(ii)*(numel(X)/theta(ii-1)- sum(abs(X(:))))*exp(eta(ii-1));
    eta(ii) = min(max(etaii,min_eta),max_eta);
    theta(ii)=exp(eta(ii));
   
    %Display Progress
    if mod(ii, round(op.N / 100)) == 0
        fprintf("\b\b\b\b%2d%%\n", round(ii / (op.N) * 100));
    end
end
%----------------------------------------------------------------------
% Compute the weighted mean
%----------------------------------------------------------------------
burnin=100; %skip first 100 iterations
deltas=arrayfun(delta,1:total_iter);
ponderedMeanFun = @(i) (theta(burnin:i)*deltas(burnin:i)')/sum(deltas(burnin:i));
ponderedMeanValues=arrayfun(ponderedMeanFun,burnin:total_iter);

results.execTimeFindLambda=toc;
theta_found = ponderedMeanValues(end);
results.theta_found= theta_found;
results.thetas=theta;
results.options=op;

%----------------------------------------------------------------------
% Solve MAP problem with the weighter average of theta
%----------------------------------------------------------------------
tau = theta_found*sigma2;
debias = 1;
stopCri = 3;

fprintf('Running GPSR...\n')
[x_coefs_estim,x_debias,obj,...
    times,debias_start,mse]= ...
    GPSR_BB(y,A,tau,...
    'Debias',debias,...
    'AT',AT,...
    'Monotone',1,...
    'Initialization',0,...
    'MaxiterA', 2000, ...
    'StopCriterion',stopCri,...
    'ToleranceD',0.001,...
    'ToleranceA',0.001,...
    'Verbose' ,0);

z_estim=pianoBasis'*x_coefs_estim;
if(numel(x_debias)>0)
    z_estim_debias=pianoBasis'*x_debias;
else
    z_estim_debias=pianoBasis'*x_coefs_estim;
end
mse = 10*log10(norm(z-z_estim_debias,'fro')^2 /(numel(z_estim_debias)));
results.mse=mse;
results.zMAPdebias=z_estim_debias;
results.zMAP=z_estim;


%% Plot Theta (log scale)
% Defaults for this blog post
width = 3;     % Width in inches
height = 2.8;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 15;      % Fontsize
lw = 1.5;      % LineWidth
msz = 15;       % MarkerSize
%---------------------
figThetaLog=figure;
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

semilogy(theta,'r','LineWidth',lw,'MarkerSize',msz);
title(['Figure 5 SOUL paper ']);
hold on;
semilogy(ponderedMeanValues,'b','LineWidth',lw,'MarkerSize',msz);
grid on;
h=legend('$\theta_{n}$','$\hat{\theta}_n$ (weighted average)');
set(h,'Interpreter','latex','fontsize',12)
xlim([1,200]);
xlabel('Iteration (n)')
ylabel('\theta')
hold off;

%% Save Audio
%Save Results
[~,name,~] = fileparts(songName);
dirname = 'results';
if ~exist(['./' dirname],'dir')
    mkdir('results');
end

audiowrite(['./' dirname  '/mary_' char(num2str(k)) 'random.wav'],results.zMAP,fs,'BitsPerSample',32);

