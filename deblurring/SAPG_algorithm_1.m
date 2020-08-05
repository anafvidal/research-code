%% Maximum Marignal Likelihood estimation of regularisation parameters
%  Algorithm 1: for homogeneous regularisers and scalar theta
%  
%  This function sets the regularisation parameter theta by maximising
%  the marginal likelihood p(y|theta) with Algorithm 1 proposed in [1].
%  For further details about the algorithm see the paper:
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
%  Copyright (2020): Ana F. Vidal, Marcelo Pereyra ??????????? 
%  ===================================================================
%{
   Usage:
   [theta_EB, results] = SAPG_algorithm_1(y, op)    

%  ===== Required inputs =============
   y: 1D vector or 2D array (image) of observations 
   ----------------
   FOR THE SAPG SCHEME
   op.samples:  max iterations for SAPG algorithm to estimate theta   
   op.burnIn:   iterations we ignore before taking the average over iterates theta_n
   op.th_init:	theta_0 initialisation of the SAPG algorithm
   op.min_th: 	projection interval Theta (min theta)
   op.max_th:	projection interval Theta (max theta)
   op.d_exp:    exponent for delta(i) = op.d_scale*( (i^(-op.d_exp)) / numel(x) )
   op.d_scale:  scale for delta(i) = op.d_scale*( (i^(-op.d_exp)) / numel(x) )
   ----------------
   MYULA PARAMETERS
   op.lambda    smoothing parameter for MYULA
   op.gamma:    discretisation step MYULA 
   ----------------
   BAYESIAN MODEL OPERATORS
   op.g:     function handle for regulariser g(x)  
   op.proxG: function handle for proximal operator of g(x)  
   op.f:     function handle for log likelihood such that p(y|xw)∝ exp{-op.f(xw)}
   op.gradF: function handle for gradient of smooth part f(x)
   op.logPi: function handle for scalar summary to monitor convergence;                  
        
%  ===== Optional inputs =============
 op.X0:      by default we assume dim(x)=dim(y) and use X0=y if op.X0 is absent
 op.stopTol: tolerance in relative change of theta_EB to stop the algorithm
             If absent, algorithm stops after op.samples iterations.
 op.warmup:  number of warm-up iterations with fixed theta for MYULA sampler
             If absent, default value is 100.

%  ===== Outputs =============
 theta_EB: estimated theta_EB computed by averaging the iterates theta_n
           skipping the appropriate burn-in iterations
 results: structure containing the following fields: 
  -execTimeFindTheta: time it took to compute theta_EB in seconds
  -last_samp:      number of iteration where the SAPG algorithm was stopped
  -logPiTrace_WU: an array ∝log pi(x_n|y,theta_0) during warm-up
                  (up to some unknown proportionality constant)
  -logPiTraceX: an array  ∝log pi(x_n|y,theta_n) for the SAPG
  -gXTrace: saves the regulariser evolution through iterations g(x_n)
  -mean_theta: theta_EB computed taking the average over all iterations
   skipping the appropriate burn-in iterations
  -last_theta: simply the last value of the iterate theta_n computed
  -thetas: full array of theta_n for all iterations
  -options: structure with all the options used to run the algorithm             

%}
function [theta_EB, results] = SAPG_algorithm_1(y, op)
    tic;
    %% Setup     
    %%%% assign default options for parameteres that were note specified
    % initialisation for MYULA sampler
    if not(isfield(op, 'X0'))
        op.X0 = y; %by default we assume dim(x)=dim(y) if nothing is specified and use X0=y
    end
    % Stop criteria (relative change tolerance)
    if not(isfield(op, 'stopTol'))
        op.stopTol = -1; % if no stopTol was defined we set it to a negative value so it's ignored.
        % in this case the SAPG algo. will stop after op.samples iterations
    end
    dimX=numel(op.X0);
    
    %%%% MYULA sampler
    if not(isfield(op, 'warmup'))
        op.warmup=100;       %use 100 by default 
    end
    warmupSteps=op.warmup;   
    lambda = op.lambda;  
    gamma=op.gamma;
    %%%% SAPG algorithm 1
    total_iter=op.samples;
    % We work on a logarithmic scale, so we define an axiliary variable 
    % eta such that theta=exp{eta}. 
    eta_init=log(op.th_init);
    min_eta=log(op.min_th);
    max_eta=log(op.max_th);
    % delta(i) for SAPG algo 
    delta = @(i) op.d_scale*( (i^(-op.d_exp)) /dimX ); 
    
    %%%% Functions related to Bayesian model
    gradF = op.gradF;% Gradient of smooth part 
    proxG =op.proxG;% Proximal operator of non-smooth part
    logPi = op.logPi;% We use this scalar summary to monitor convergence
    g = op.g;
    
    %% MYULA Warm-up
    X_wu = op.X0;
    if(warmupSteps>0)
        % Run MYULA sampler with fix theta to warm up the markov chain
        fix_theta=op.th_init;
        logPiTrace_WU(op.warmup)=0;%logPiTrace to monitor convergence

        % Run MYULA sampling from the posterior of X:         
        proxGX_wu = proxG(X_wu,lambda,fix_theta);
        fprintf('Running Warm up     \n');
        for ii = 2:warmupSteps
            %Sample from posterior with MYULA:
            X_wu =  X_wu + gamma* (proxGX_wu-X_wu)/lambda -gamma*gradF(X_wu) + sqrt(2*gamma)*randn(size(X_wu));
            proxGX_wu = proxG(X_wu,lambda,fix_theta);
            %Save current state to monitor convergence
            logPiTrace_WU(ii) = logPi(X_wu,fix_theta);
            %Display Progress
            if mod(ii, round(op.warmup / 100)) == 0
                fprintf('\b\b\b\b%2d%%\n', round(ii / (op.warmup) * 100));
            end
        end
        results.logPiTrace_WU=logPiTrace_WU;
    end
    %% Run SAPG algorithm 1 to estimate theta
    theta(total_iter)=0;
    theta(1)=op.th_init;
    eta(total_iter)=0;
    eta(1)=eta_init;
    logPiTraceX(total_iter)=0;% to monitor convergence
    gX(total_iter)=0; % to monitor how the regularisation function evolves

    X = X_wu;% start MYULA markov chain from last sample after warmup
    logPiTraceX(1)=logPi(X,theta(1));
    proxGX = proxG(X,lambda,theta(1));
    fprintf('\nRunning SAPG algorithm     \n');

    for ii = 2:total_iter 
        %Sample from posterior with MYULA:
        Z = randn(size(X));
        X =  X + gamma* (proxGX-X)/lambda -gamma*gradF(X) + sqrt(2*gamma)*Z;
        proxGX = proxG(X,lambda,theta(ii-1));    

        %Update theta   
        etaii = eta(ii-1) + delta(ii)*(dimX/theta(ii-1)- g(X))*exp(eta(ii-1));    
        eta(ii) = min(max(etaii,min_eta),max_eta);
        theta(ii)=exp(eta(ii));   

        %Save current state to monitor convergence
        logPiTraceX(ii) = logPi(X,theta(ii-1));
        gX(ii-1) = g(X);

        %Display Progress
        if mod(ii, round(total_iter / 100)) == 0
            fprintf('\b\b\b\b%2d%%\n', round(ii / (total_iter) * 100));
        end

        %Check stop criteria. If relative error is smaller than op.stopTol stop
        relErrTh1=abs(exp(mean(eta(op.burnIn:ii)))-exp(mean(eta(op.burnIn:ii-1))))/exp(mean(eta(op.burnIn:ii-1)));
        if(relErrTh1<op.stopTol) && (ii>op.burnIn+1)
            break;
        end
    end
    
    %Save logs in results struct
    results.execTimeFindTheta=toc;
    last_samp=ii;results.last_samp=last_samp;
    results.logPiTraceX=logPiTraceX(1:last_samp);
    results.gXTrace=gX(1:last_samp);
    theta_EB = exp(mean(eta(op.burnIn:last_samp)));
    results.mean_theta= theta_EB;
    results.last_theta= theta(last_samp);
    results.thetas=theta(1:last_samp);    
    results.options=op;

end