
%% Maximum Marignal Likelihood estimation of regularisation parameters
%  Algorithm 2 applied to hyperspectral unmixing with TV-SUnSAL prior
%  used in Experiment 4.3 of the SIAM article [1].
%  
%  This function sets the regularisation parameters theta_1 and theta_2 by 
%  maximising the marginal likelihood p(y|theta) with Algorithm 2 proposed
%  in [1]. For further details about the algorithm see the paper:
%
%  [1] A. F. Vidal, V. De Bortoli, M. Pereyra, and D. Alain, Maximum 
%  likelihood estimation of regularisation parameters in high-dimensional
%  inverse problems: an empirical bayesian approach. Part I
%  
%  =================================================================== 
%  Authors: Ana F. Vidal, Marcelo Pereyra
%  Date: Mar, 2018. 
%  Revised: Sep, 2020
%  For any technical queries/comments/bug reports, please contact:
%  anafernandezvidal (at) gmail (dot) com
%  -------------------------------------------------------------------
%  Copyright (2020): Ana F. Vidal, Marcelo Pereyra
%  ===================================================================
%{
   Usage:
   [theta1_EB,theta2_EB, results] = SAPG_algorithm_2_hyper_unmix(y, op)    

%  ===== Required inputs =============
   y: hyperspectral observation [nf x np] 
   ----------------
   FOR THE SAPG SCHEME
   op.samples:  max iterations for SAPG algorithm to estimate theta   
   op.th1_init:	theta1_0 initialisation of the SAPG algorithm
   op.th2_init:	theta2_0 initialisation of the SAPG algorithm
   op.min_th1: 	projection interval Theta (min theta1)
   op.min_th2: 	projection interval Theta (min theta2)
   op.max_th1:	projection interval Theta (max theta1)
   op.max_th2:	projection interval Theta (max theta2)
   op.d_exp:    exponent for delta(i) = op.d_scale*( (i^(-op.d_exp)) / numel(x) )
   We admit different scales for theta1 and theta2:
   op.d1_scale:  scale for delta1(i) = op.d1_scale*( (i^(-op.d_exp)) / dimension(g1) )
   op.d2_scale:  scale for delta2(i) = op.d2_scale*( (i^(-op.d_exp)) / dimension(g2) )
   ----------------
   MYULA PARAMETERS
   op.lambda:    smoothing parameter for MYULA sampling from posterior
   op.gamma:    discretisation step MYULA sampling from posterior
   ----------------
   HYPERSPECTRAL UNMIXING PROBLEM
   op.A:     dictionary with end member spectral signatures
   op.sigma: noise standard deviation
   
%  ===== Optional inputs =============
 op.X0:      by default we assume dim(x)=dim(y) and use X0=y if op.X0 is absent
 op.stopTol: tolerance in relative change of theta_EB to stop the algorithm
             If absent, algorithm stops after op.samples iterations.
 op.burnin:  iterations we ignore before taking the average over iterates theta_n
             If absent, default value is 30.
 op.warmupSteps:  number of warm-up iterations with fixed theta for MYULA sampler
                  If absent, default value is 100. 
%  ===== Outputs =============
 theta1_EB: estimated theta1_EB computed by averaging the iterates theta1_n
           skipping the appropriate burn-in iterations
 theta2_EB: estimated theta2_EB computed by averaging the iterates theta1_n
           skipping the appropriate burn-in iterations
 results: structure containing the following fields: 
  -execTimeFindTheta: time it took to compute theta_EB in seconds
  -last_samp: number of iteration where the SAPG algorithm was stopped
  -logPiTrace_WU_X: an array with k*log pi(x_n|y,theta_0) during warm-up
   (k is some unknown proportionality constant)
  -logPiTraceX: an array with k*log pi(x_n|y,theta_n) for the SAPG algo
  -mean_th1: theta1_EB computed taking the average over all iterations
  -mean_th2: theta2_EB computed taking the average over all iterations
   skipping the appropriate burn-in iterations
  -last_th1: simply the last value of the iterate theta1_n computed
  -last_th2: simply the last value of the iterate theta2_n computed
  -th1_list: full array of theta1_n for all iterations
  -th2_list: full array of theta2_n for all iterations  
  -log_g1: saves the regulariser evolution through iterations g1(x_n)
  -log_g2: saves the regulariser evolution through iterations g2(x_n)
  -options: structure with all the options used to run the algorithm

%}

function [theta1_EB,theta2_EB, results] = SAPG_algorithm_2_hyper_unmix(y, op)
 tic;
%% Setup 
function x = clip_to_positive(x)
  x(x<0) = 0;
end
%%%% Assign default options for parameteres that were note specified
% Initialisation for MYULA sampler
if not(isfield(op, 'X0')) 
    op.X0 = clip_to_positive(pinv(op.A)*y); %by default we use pseudoinverse solution as initial condition
end
% Stop criteria (relative change tolerance)
if not(isfield(op, 'stopTol'))
    op.stopTol = -1; % if no stopTol was defined we set it to a negative value so it's ignored.
    % in this case the SAPG algo. will stop after op.samples iterations
end
% Burn-in (number of skipped initial samples)
if not(isfield(op, 'burnin'))
    op.burnin = 30; % if no burnin was defined we set it to 30
end
% Warm-up steps for MYULA sampler
if not(isfield(op, 'warmupSteps'))
    op.warmupSteps=100;       %use 100 by default 
end


%Define dimensions of X and of each semi-separable regulariser
dimX=numel(op.X0);
%Since we will do variable splitting, becuase TV and L1 norm are applied on
%differen variables, we define the dimension for each part:
dimL1=dimX; % Due to the positivity constraint the dimension for this subspace is equal to dimX
dimTV=dimX-1; % TV norm dependes on the differences not on the mean so it's one less than dimX. 

%Rename some variables to simplify notation
sigma2=op.sigma^2;
lambda = op.lambda; gamma=op.gamma;
A=op.A;

% Compute inverse of AtA for gradient preconditioning in MYULA 
AtA=A'*A;
invAtA=inv(AtA);% 
sqrtInvAta=sqrtm(invAtA);

%%%% Setup for SAPG algorithm 2
total_iter=op.samples;

% We work on a logarithmic scale, so we define an axiliary variable 
% eta such that theta=exp{eta}. 
min_eta1=log(op.min_th1);
max_eta1=log(op.max_th1);
min_eta2=log(op.min_th2);
max_eta2=log(op.max_th2);

%Now the delta step for the prox grad algorithm will be different for both
%regularizers becuase they have different dimensions
delta1 = @(i) op.d1_scale*( (i^(-op.d_exp)) / (dimL1) ); %delta(i) for proximal gradient algorithm 
delta2 = @(i) op.d2_scale*( (i^(-op.d_exp)) / (dimTV) ); %delta(i) for proximal gradient algorithm 

%% Definition of proximal operator using SUnSAL solver
% We will use sunsal to compute the prox operator of TV+L1 norm and non-negativity constraint. 
%  SUNSAL_TV solves the following l_2 + l_{1,1} + TV optimization problem:
%
%     Definitions
%
%      A  -> L * n_em; Mixing matrix (Library) n_em is number of end members
%      X  -> n_em * np; 
%
%      Optimization problem
%
%    min  (1/2) ||A X-Y||^2_F  + k1  ||X||_{1,1}  + k2 TV(X);
%     X
%If we choose A=eye(n_em) and   and Y=X0 we will solve
%    min  (1/2) ||X-X0||^2_F  +k1 ||X||_1 + k2 TV(X);
%     X

%where k1=thL1*lambdaProx, k2=thtv*lambdaProx amd n_em is the number of 
%endmembers in the dictionary A
imDimensions=[op.nl,op.nc];
sunsal_prox = @(X0,k1,k2)  sunsal_tv(eye(op.n_em),X0,'MU',0.05,'POSITIVITY','yes','ADDONE','no', ...
                               'LAMBDA_1',k1,'LAMBDA_TV', k2, 'TV_TYPE','iso',...
                               'IM_SIZE',imDimensions,'AL_ITERS',op.itersSUnSAL,  'VERBOSE','no');
                           %returns [X_hat_tv_i,res,rmse_i] AL_ITERS=150
%% Functions related to Bayesian model
%%%% Likelihood (data fidelity)
f = @(x) (norm(y-A*x,'fro')^2)/(2*sigma2);% p(y|x)∝ exp{-f(x)}
gradF = @(x) (A'*(A*x-y)/sigma2); % Gradient of smooth part f

%%%% Regulariser
g1= @(x) sum(abs(x(:))); %L1 reg
g2= @(x) tvNormVect(x,op.nl,op.nc); %TV reg
% Proximal operator of non-regular part (TV+L1+non-negativity)
proxG = @(x,lambda,th1,th2) sunsal_prox(x,lambda*th1,lambda*th2);    
                         
% We use this scalar summary to monitor convergence
logPi = @(x,th1,th2) -f(x) -th1*g1(x)-th2*g2(x);

%% MYULA Warm-up
 X_wu = op.X0;  %posterior chain initialization 
if (op.warmupSteps>0)
    fix_th1=op.th1_init;
    fix_th2=op.th2_init;
    logPiTrace_WU_X(op.warmupSteps)=0;%logPiTrace for warm up iterations posterior chain
  
    % Run MYULA sampling from the posterior of X:     
    fprintf('Running Warm up     \n');
    proxG_X=proxG(X_wu,lambda,fix_th1,fix_th2);
    for ii = 2:op.warmupSteps 
        %Sample from posterior with MYULA:
        X_wu =  X_wu -invAtA*gamma*gradF(X_wu)+ invAtA*gamma* (proxG_X-2*X_wu+clip_to_positive(X_wu))/lambda  + sqrtInvAta*sqrt(2*gamma)*randn(size(X_wu));
        proxGX_wu = proxG(X_wu,lambda,fix_th1,fix_th2);         
        %Save current state to monitor convergence
        logPiTrace_WU_X(ii) = logPi(X_wu,fix_th1,fix_th2);
        %Display Progress        
        fprintf('\b\b\b\b%2d%%\n', round(ii / (op.warmupSteps) * 100));        
    end
    results.logPiTrace_WU_X=logPiTrace_WU_X;          
end


%% Run SAPG algorithm 2 to estimate theta_1 and theta_2
% We use a single MCMC chain sampling from the posterior of X.
th1(total_iter)=0;      th2(total_iter)=0;
th1(1)=op.th1_init;     th2(1)=op.th2_init;
% We work on a logarithmic scale, so we define an axiliary variable 
% eta_i such that theta_i=exp{eta_i}, i={1,2}. 
eta1(total_iter)=0;     eta2(total_iter)=0;
eta1(1)=log(th1(1));    eta2(1)=log(th2(1));

% Logs                  
logPiTraceX(total_iter)=0; % to monitor convergence
log_g1(total_iter)=0;  % to monitor how the L1 regularisation function evolves
log_g2(total_iter)=0;  % to monitor how the TV regularisation function evolves

fprintf('\nRunning SAPG algorithm    \n');
X = X_wu;           % start MYULA markov chain from last sample after warmup
proxGX=proxG(X,lambda,th1(1),th2(1));    

for ii = 2:total_iter
        
    %Sample from posterior with MYULA:        
    X =  X + gamma*invAtA*((proxGX-2*X+clip_to_positive(X))/lambda - gradF(X)) + sqrtInvAta*sqrt(2*gamma)*randn(size(X));
    proxGX=proxG(X,lambda,th1(ii-1),th2(ii-1));
    
    %Update theta_1 and theta_2    
    eta1ii = eta1(ii-1) + delta1(ii)*(dimL1/th1(ii-1)- g1(X))*exp(eta1(ii-1));    
    eta1(ii) = min(max(eta1ii,min_eta1),max_eta1);
    th1(ii)=exp(eta1(ii)); 
    
    eta2ii = eta2(ii-1) + delta2(ii)*(dimTV/th2(ii-1)- g2(X))*exp(eta2(ii-1));    
    eta2(ii) = min(max(eta2ii,min_eta2),max_eta2);
    th2(ii)=exp(eta2(ii));
    
    %Save current state to monitor convergence  
    logPiTraceX(ii) = logPi(X,th1(ii-1),th2(ii-1));
    log_g1(ii-1)=g1(X);    
    log_g2(ii-1)=g2(X);
    
    %Display Progress
    fprintf('\b\b\b\b%2d%%\n', round(ii / (total_iter) * 100)); 
    
    %Check stop criteria. If relative error is smaller than op.stopTol stop
    if(ii>op.burnin)
        relErrTh1=abs(mean(th1(op.burnin:ii))-mean(th1(op.burnin:ii-1)))/mean(th1(op.burnin:ii-1));
        relErrTh2=abs(mean(th2(op.burnin:ii))-mean(th2(op.burnin:ii-1)))/mean(th2(op.burnin:ii-1));
        if(max(relErrTh1,relErrTh2)<op.stopTol)
            break;
        end
    end
end %end for loop sapg algo

%Save results in structure
results.execTimeFindTheta=toc;
last_samp=ii;
results.last_samp=last_samp;
results.logPiTraceX=logPiTraceX;
theta1_EB = mean(th1(op.burnin:last_samp));
theta2_EB = mean(th2(op.burnin:last_samp));
results.mean_th1= theta1_EB;
results.mean_th2= theta2_EB;
results.last_th1= th1(last_samp);
results.last_th2= th2(last_samp);
results.th1_list=th1;
results.th2_list=th2;
results.log_g1=log_g1;
results.log_g2=log_g2;
results.options=op;

end