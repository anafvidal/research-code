
%% Maximum Marignal Likelihood estimation of regularisation parameters
%  Algorithm 3 applied to denoising with Total-Generalised-Variation prior
%  used in Experiment 4.4 of the SIAM article [1].
%  
%  This function sets the regularisation parameters theta_1 and theta_2 by 
%  maximising the marginal likelihood p(y|theta) with Algorithm 3 proposed
%  in [1]. For further details about the algorithm see the paper:
%
%  [1] A. F. Vidal, V. De Bortoli, M. Pereyra, and D. Alain, Maximum 
%  likelihood estimation of regularisation parameters in high-dimensional
%  inverse problems: an empirical bayesian approach. Part I
%  
%  =================================================================== 
%  Authors: Ana F. Vidal, Marcelo Pereyra
%  Date: Mar, 2018. 
%  Revised: May, 2020
%  For any technical queries/comments/bug reports, please contact:
%  anafernandezvidal (at) gmail (dot) com
%  -------------------------------------------------------------------
%  Copyright (2020): Ana F. Vidal, Marcelo Pereyra
%  ===================================================================
%{
   Usage:
   [theta1_EB,theta2_EB, results] = SAPG_algorithm_3_denois_tgv(y, op)    

%  ===== Required inputs =============
   y: 1D vector or 2D array (image) of observations 
   ----------------
   FOR THE SAPG SCHEME
   op.samples:  max iterations for SAPG algorithm to estimate theta   
   op.burnIn:   iterations we ignore before taking the average over iterates theta_n
   op.th1_init:	theta1_0 initialisation of the SAPG algorithm
   op.th2_init:	theta2_0 initialisation of the SAPG algorithm
   op.min_th1: 	projection interval Theta (min theta1)
   op.min_th2: 	projection interval Theta (min theta2)
   op.max_th1:	projection interval Theta (max theta1)
   op.max_th2:	projection interval Theta (max theta2)
   op.d_exp:    exponent for delta(i) = op.d_scale*( (i^(-op.d_exp)) / numel(x) )
   We admit different scales for theta1 and theta2:
   op.d1_scale:  scale for delta1(i) = op.d1_scale*( (i^(-op.d_exp)) / numel(x) )
   op.d2_scale:  scale for delta2(i) = op.d2_scale*( (i^(-op.d_exp)) / numel(x) )
   ----------------
   MYULA PARAMETERS
   op.lambdaX    smoothing parameter for MYULA sampling from posterior
   op.lambdaU    smoothing parameter for MYULA sampling from prior
   op.gammaX:    discretisation step MYULA sampling from posterior
   op.gammaU:    discretisation step MYULA sampling from prior
   op.thinningU: thinning factor for prior chain (i.e., iterations per sample)                 
        
%  ===== Optional inputs =============
 op.X0:      by default we assume dim(x)=dim(y) and use X0=y if op.X0 is absent
 op.U0:      by default we assume dim(u)=dim(y) and use U0=y if op.X0 is absent
 op.stopTol: tolerance in relative change of theta_EB to stop the algorithm
             If absent, algorithm stops after op.samples iterations.
 op.warmup:  number of warm-up iterations with fixed theta for MYULA sampler
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
  -logPiTrace_WU_U: an array with k*log pi(u_n|theta_0) during warm-up
  -logPiTraceX: an array with k*log pi(x_n|y,theta_n) for the SAPG algo
  -logPiTraceU: an array with k*log pi(u_n|theta_n) for the SAPG algo 
  -mean_th1: theta1_EB computed taking the average over all iterations
  -mean_th2: theta2_EB computed taking the average over all iterations
   skipping the appropriate burn-in iterations
  -last_th1: simply the last value of the iterate theta1_n computed
  -last_th2: simply the last value of the iterate theta2_n computed
  -th1_list: full array of theta1_n for all iterations
  -th2_list: full array of theta2_n for all iterations  
  -tgvnorm_x: saves the regulariser evolution through iterations g(x_n)
  -tgvnorm_u: saves the regulariser evolution through iterations g(u_n)  
  -options: structure with all the options used to run the algorithm

%}

function [theta1_EB,theta2_EB, results] = SAPG_algorithm_3_denois_tgv(y, op)
 tic;
%% Setup     
%%%% assign default options for parameteres that were note specified
% initialisation for MYULA sampler
if not(isfield(op, 'X0')) % for posterior chain
    op.X0 = y; %by default we assume dim(x)=dim(y) if nothing is specified and use X0=y
end
if not(isfield(op, 'U0'))% for prior chain
    op.U0 = y; %by default we assume dim(x)=dim(y) if nothing is specified and use X0=y
end
% Stop criteria (relative change tolerance)
if not(isfield(op, 'stopTol'))
    op.stopTol = -1; % if no stopTol was defined we set it to a negative value so it's ignored.
    % in this case the SAPG algo. will stop after op.samples iterations
end
dimX=numel(op.X0);
sigma=op.sigma;
%%%% MYULA sampler
if not(isfield(op, 'warmupSteps'))
    op.warmupSteps=100;       %use 100 by default 
end

%%%% SAPG algorithm 3
total_iter=op.samples;
% We work on a logarithmic scale, so we define an axiliary variable 
% eta such that theta=exp{eta}. 
min_eta1=log(op.min_th1);
max_eta1=log(op.max_th1);
min_eta2=log(op.min_th2);
max_eta2=log(op.max_th2);
% If initialisation is really bad e.g. th=40, we need different scales,
% for instance delta1 should be 1000 times larger than the current value.

delta1 = @(i) op.d1_scale*( (i^(-op.d_exp)) / (dimX) ); %delta(i) for proximal gradient algorithm
delta2 = @(i) op.d2_scale*( (i^(-op.d_exp)) / (dimX) ); %delta(i) for proximal gradient algorithm


%% Functions related to Bayesian model
%%%% Likelihood (data fidelity)
f = @(x) (norm(x-y,'fro')^2)/(2*sigma^2);% p(y|x)∝ exp{-f(x)}
gradF = @(x) (x-y)/sigma^2; % Gradient of smooth part f

%%%% Regulariser
%This is an approximation to TGV norm, for details see article [1]
%g1 g2 TGV norm
tgvProxIter=100;
getProx_g1_g2= @(x,th1,th2,lambda) TGVprox(x,th1,th2,lambda,tgvProxIter);%TGV norm prox

% We use this scalar summary to monitor convergence
logPiX = @(x,th1,th2,g1,g2) -f(x) -th1*g1-th2*g2;
logPiU = @(u,th1,th2,g1,g2) -th1*g1-th2*g2;

epsil=1e-10; %small constant to add a cuadratic penalty for when we sample 
             %from the improper prior, to make it proper.

%% MYULA Warm-up
 X_wu = op.X0;  %posterior chain initialization
 U_wu =op.U0;   %prior chain initialization
if (op.warmupSteps>0)
    fix_th1=op.th1_init;
    fix_th2=op.th2_init;
    logPiTrace_WU_X(op.warmupSteps)=0;%logPiTrace for warm up iterations posterior chain
    logPiTrace_WU_U(op.warmupSteps)=0;%logPiTrace for warm up iterations prior chain

    % Run MYULA sampling from the posterior (X chain) and prior (U chain):     
    fprintf('Running Warm up     \n');
    proxTGV_X=getProx_g1_g2(X_wu,fix_th1,fix_th2,op.lambdaX) ;
    proxTGV_U=getProx_g1_g2(U_wu,fix_th1,fix_th2,op.lambdaU) ;
    for ii = 2:op.warmupSteps 
        %Sample from posterior with MYULA:
        X_wu =  X_wu - op.gammaX*gradF(X_wu) -op.gammaX* (X_wu-proxTGV_X)/op.lambdaX+ sqrt(2*op.gammaX)*randn(size(X_wu)); 
        [proxTGV_X,g1_x,g2_x]=getProx_g1_g2(X_wu,fix_th1,fix_th2,op.lambdaX) ;
        for iv=1:op.thinningU
            %Sample from prior with MYULA:
            U_wu =  U_wu  -op.gammaU*epsil*2*U_wu -op.gammaU* (U_wu-proxTGV_U)/op.lambdaU + sqrt(2*op.gammaU)*randn(size(U_wu));
            [proxTGV_U,g1_u,g2_u]=getProx_g1_g2(U_wu,fix_th1,fix_th2,op.lambdaU) ;
        end
        %Save current state to monitor convergence
        logPiTrace_WU_X(ii) = logPiX(X_wu,fix_th1,fix_th2,g1_x,g2_x);
        logPiTrace_WU_U(ii) = logPiU(U_wu,fix_th1,fix_th2,g1_u,g2_u);
        %Display Progress        
        fprintf('\b\b\b\b%2d%%\n', round(ii / (op.warmupSteps) * 100));        
        
    end
    results.logPiTrace_WU_X=logPiTrace_WU_X;    
    results.logPiTrace_WU_U=logPiTrace_WU_U;    
end


%% Run SAPG algorithm 3 to estimate theta_1 and theta_2
% We use two MCMC chains for estimating the normalization constant as well.
th1(total_iter)=0;      th2(total_iter)=0;
th1(1)=op.th1_init;     th2(1)=op.th2_init;
% We work on a logarithmic scale, so we define an axiliary variable 
% eta_i such that theta_i=exp{eta_i}, i={1,2}. 
eta1(total_iter)=0;     eta2(total_iter)=0;
eta1(1)=log(th1(1));    eta2(1)=log(th2(1));

%posterior                  %prior
logPiTraceX(total_iter)=0;  logPiTraceU(total_iter)=0;% to monitor convergence
tgvnorm_x(total_iter)=0;    tgvnorm_u(total_iter)=0;% to monitor how the regularisation function evolves
X = X_wu;                   U = U_wu;  % start MYULA markov chain from last sample after warmup
   

fprintf('\nRunning SAPG algorithm    \n');
[proxTGV_X,g1_X,g2_X]=getProx_g1_g2(X,th1(1),th2(1),op.lambdaX);
[proxTGV_U,g1_U,g2_U]=getProx_g1_g2(U,th1(1),th2(1),op.lambdaU);

             
for ii = 2:total_iter
        
    %Sample from posterior with MYULA:    
    Z = randn(size(X));    
    X =  X - op.gammaX*gradF(X) -op.gammaX* (X-proxTGV_X)/op.lambdaX + sqrt(2*op.gammaX)*Z;
    for iv=1:op.thinningU
        %Sample from prior with MYULA:
        U =  U -op.gammaU*epsil*2*U -op.gammaU* (U-proxTGV_U)/op.lambdaU + sqrt(2*op.gammaU)*randn(size(U));
        [proxTGV_U,g1_U,g2_U]=getProx_g1_g2(U,th1(ii-1),th2(ii-1),op.lambdaU);
    end
    [proxTGV_X,g1_X,g2_X]=getProx_g1_g2(X,th1(ii-1),th2(ii-1),op.lambdaX);
    
    %Update theta_1 and theta_2    
    eta1ii = eta1(ii-1) + delta1(ii)*(g1_U- g1_X)*exp(eta1(ii-1));    
    eta1(ii) = min(max(eta1ii,min_eta1),max_eta1);
    th1(ii)=exp(eta1(ii));    
    eta2ii = eta2(ii-1) + delta2(ii)*(g2_U- g2_X)*exp(eta2(ii-1));    
    eta2(ii) = min(max(eta2ii,min_eta2),max_eta2);
    th2(ii)=exp(eta2(ii));
    
    %Save current state to monitor convergence  
    logPiTraceX(ii) = logPiX(X,th1(ii-1),th2(ii-1),g1_X,g2_X);
    logPiTraceU(ii) = logPiU(U,th1(ii-1),th2(ii-1),g1_U,g2_U);
    tgvnorm_x(ii-1)=th1(ii-1)*g1_X+th2(ii-1)*g2_X;
    tgvnorm_u(ii-1)=th1(ii-1)*g1_U+th2(ii-1)*g2_U;
    
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
results.logPiTraceU=logPiTraceU;
theta1_EB = mean(th1(op.burnin:last_samp));
theta2_EB = mean(th2(op.burnin:last_samp));
results.mean_th1= theta1_EB;
results.mean_th2= theta2_EB;
results.last_th1= th1(last_samp);
results.last_th2= th2(last_samp);
results.th1_list=th1;
results.th2_list=th2;
results.tgvnorm_x=tgvnorm_x;
results.tgvnorm_u=tgvnorm_u;
results.options=op;

end
