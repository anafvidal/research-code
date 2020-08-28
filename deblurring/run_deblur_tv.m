%% Experiment 4.2.1 - Deblurring with Total-Variation prior
%{
  Experiment 4.2.1 of the SIAM article [1]:
  Deblurring with Total-Variation prior

  Computes the maximum marignal likelihood estimation of the
  regularisation parameter using Algorithm 1 from [1], i.e.
  the algorithm for homogeneous regularisers and scalar theta.
  
  It runs Algorithm 1 for 10 test images, and computes the MAP estimator
  using the estimated value theta_EB and the SALSA solver.

  All results are saved in a <img_name>_results.mat file in a 'results'
  directory. This .mat file contains a structure with:
  
  -execTimeFindTheta: time it took to compute theta_EB in seconds
  -last_samp: number of iteration where the SAPG algorithm was stopped
  -logPiTrace_WU: an array with k*log pi(x_n|y,theta_0) during warm-up
   (k is some unknown proportionality constant)
  -logPiTraceX: an array with k*log pi(x_n|y,theta_n) for the SAPG
  -gXTrace: saves the regulariser evolution through iterations g(x_n)
  -mean_theta: theta_EB computed taking the average over all iterations
   skipping the appropriate burn-in iterations
  -last_theta: simply the last value of the iterate theta_n computed
  -thetas: full array of theta_n for all iterations
  -options: structure with all the options used to run the algorithm
  -mse: mean squarred error after computing x_MAP with theta_EB
  -xMAP: estimated x_MAP
  -x: original image
  -y: observations (blurred and noisy image)

  [1] A. F. Vidal, V. De Bortoli, M. Pereyra, and D. Alain, Maximum
  likelihood estimation of regularisation parameters in high-dimensional
  inverse problems: an empirical bayesian approach. Part I:Methodology and 
  Experiments.
  
  Prerequired libraries:  
  - SALSA: http://cascais.lx.it.pt/~mafonso/salsa.html

%}
%  ===================================================================
%% Create array with test images names
clear all;clc;
testImg=dir('images');
testImg={testImg(3:end).name};
%Check that test images are accesible
if sum(size(testImg)) == 0
    error('No images found, please check that the images directory is in the MATLAB path');
end
%Check that SALSA solver is in the MATLAB path
if exist('SALSA_v2')==0
    error('SALSA package not found. Please make sure that the salsa solver is added to your MATLAB path.  The code can be obtained from http://cascais.lx.it.pt/~mafonso/salsa.html');
end
% Set to true to save additional plots
save_logpi_plots= true; % to check stabilisation of the MYULA sampler
save_gx_plot=true; % to check evolution of the regulariser g and the gradient approx

%% Parameter Setup
op.samples =1500; % max iterations for SAPG algorithm to estimate theta
op.stopTol=1e-3; % tolerance in relative change of theta_EB to stop the algorithm
op.burnIn=20;	% iterations we ignore before taking the average over iterates theta_n

op.th_init = 0.01 ; % theta_0 initialisation of the SAPG algorithm
op.min_th=1e-3; % projection interval Theta (min theta)
op.max_th=1; % projection interval Theta (max theta)

% delta(i) for SAPG algorithm defined as: op.d_scale*( (i^(-op.d_exp)) / numel(x) );
op.d_exp =  0.8;
op.d_scale =  0.1/op.th_init;

%MYULA parameters
op.warmup = 300 ;% number of warm-up iterations with fixed theta for MYULA sampler
op.lambdaMax = 2; % max smoothing parameter for MYULA
lambdaFun= @(Lf) min((5/Lf),op.lambdaMax);
gamma_max = @(Lf,lambda) 1/(Lf+(1/lambda));%Lf is the lipschitz constant of the gradient of the smooth part
op.gammaFrac=0.98; % we set gamma=op.gammaFrac*gamma_max

randn('state',1); % Set rnd generators (for repeatability)
%% RUN EXPERIMENTS
for i_im = 1:length(testImg)
    for bsnr=[30]
        fprintf('SNR=%d\n',bsnr); op.BSNRdb=bsnr;
        filename=testImg{i_im};
        %% Generate a noisy and blurred observation vector 'y'
        %%%%%  original image 'x'
        fprintf(['File:' filename '\n']);
        x = double(imread(filename));
        dimX = numel(x);
        
        %%%%%  blur operator
        blur_length=9;
        [A,AT,H_FFT,HC_FFT]=uniform_blur(length(x),blur_length);
        evMax=max_eigenval(A,AT,size(x),1e-4,1e4,0);%Maximum eigenvalue of operator A.
        
        %%%%% observation y
        Ax = A(x);
        sigma = norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNRdb/10));
        sigma2 = sigma^2; op.sigma=sigma;
        y = Ax + sigma*randn(size(Ax));
        
        %% Experiment setup: functions related to Bayesian model
        %%%% Regulariser
        % TV norm
        op.g = @(x) TVnorm(x); %g(x) for TV reg
        chambolleit = 25;
        % Proximal operator of g(x)         
        op.proxG = @(x,lambda,theta) chambolle_prox_TV_stop(x,'lambda',lambda*theta,'maxiter',chambolleit);
        Psi = @(x,th) op.proxG(x,1,th); % define this format for SALSA solver
                         
        %%%% Likelihood (data fidelity)
        op.f = @(x) (norm(y-A(x),'fro')^2)/(2*sigma2); % p(y|x)∝ exp{-op.f(x)}
        op.gradF = @(x) real(AT(A(x)-y)/sigma2);  % Gradient of smooth part f
        Lf = (evMax/sigma)^2; % define Lipschitz constant of gradient of smooth part

        % We use this scalar summary to monitor convergence
        op.logPi = @(x,theta) -op.f(x) -theta*op.g(x);
        
        %%%% Set algorithm parameters that depend on Lf
        op.lambda=lambdaFun(Lf);%smoothing parameter for MYULA sampler
        op.gamma=op.gammaFrac*gamma_max(Lf,op.lambda);%discretisation step MYULA
        
        %% Run SAPG Algorithm 1 to compute theta_EB
        [theta_EB,results]=SAPG_algorithm_1(y,op);
        
        %% Solve MAP problem with theta_EB         
        % Setup for SALSA solver
        global calls;
        calls = 0;
        A = @(x) callcounter(A,x);
        AT = @(x) callcounter(AT,x);
        inneriters = 1;
        outeriters = 500;%500 is used in salsa demo.
        tol = 1e-5%taken from salsa demo.
        mu = theta_EB/10;
        muinv = 1/mu;
        filter_FFT = 1./(abs(H_FFT).^2 + mu);
        invLS = @(x) real(ifft2(filter_FFT.*fft2( x )));
        invLS = @(xw) callcounter(invLS,xw);
        fprintf('Running SALSA solver...\n')
        % SALSA
        [xMAP, ~, ~, ~, ~, ~, ~] = ...
            SALSA_v2(y, A, theta_EB*sigma^2,...
            'MU', mu, ...
            'AT', AT, ...
            'StopCriterion', 1, ...
            'True_x', x, ...
            'ToleranceA', tol, ...
            'MAXITERA', outeriters, ...
            'Psi', Psi, ...
            'Phi', op.g, ...
            'TVINITIALIZATION', 1, ...
            'TViters', 10, ...
            'LS', invLS, ...
            'VERBOSE', 0);
        %Compute MSE with xMAP
        mse = 10*log10(norm(x-xMAP,'fro')^2 /dimX);
        results.mse=mse;
        results.xMAP=xMAP;
        results.x=x;
        results.y=y;
        
        %% Save Results
        %Create directories
        %subdirID: creates aditional folder with this name to store results
        subdirID = ['algo1_tol' char(num2str(op.stopTol)) '_maxIter' char(num2str(op.samples)) ];
        dirname = 'results/4.2.1-deblur-TV';
        subdirname=['BSNR' char(num2str(op.BSNRdb)) '/' subdirID ];
        if ~exist(['./' dirname],'dir')
            mkdir('results');
        end
        if ~exist(['./' dirname '/' subdirname ],'dir')
            mkdir(['./' dirname '/' subdirname ]);
        end
        
        % Save data
        [~,name,~] = fileparts(char(filename));
        save(['./' dirname '/' subdirname '/' name '_results.mat'], '-struct','results');
        close all;
        
        % Save figures (degraded and recovered)
        f1=figure; imagesc(results.x), colormap gray, axis equal, axis off;
        f2=figure; imagesc(results.y),  colormap gray, axis equal, axis off;
        f3=figure; imagesc(results.xMAP), colormap gray, axis equal, axis off;
        
        saveas(f2,['./' dirname '/' subdirname '/' name '_degraded.png']);
        saveas(f3,['./' dirname '/' subdirname '/' name '_estim.png']);
        saveas(f1,['./' dirname '/' subdirname '/' name '_orig.png']);
        close all;
        if save_logpi_plots
            if(op.warmup>0)
                % Save images of logpitrace
                figLogPiTraceWarmUp=figure;
                plot(results.logPiTrace_WU(2:end),'b','LineWidth',1.5,'MarkerSize',8);   hold on;
                title('\propto log p(X^n|y,\theta) during warm-up');xlabel('Iteration (n)');grid on; hold off;
                saveas(figLogPiTraceWarmUp,['./' dirname '/' subdirname '/' name '_logPiTraceX_warmup.png' ]);
            end
            
            figLogPiTrace=figure;
            plot(results.logPiTraceX(2:end),'r','LineWidth',1.5,'MarkerSize',8); hold on;
            title('\propto log p(X^n|y,\theta) in SAPG algorithm');xlabel('Iteration (n)');grid on;hold off;
            saveas(figLogPiTrace,['./' dirname '/' subdirname '/' name '_logPiTraceX.png' ]);
        end
        if save_gx_plot
            figSum=figure;
            plot(results.gXTrace(1:results.last_samp-1),'r','LineWidth',1.5,'MarkerSize',8);hold on;
            plot(dimX*((results.thetas(1:results.last_samp-1)).^(-1)),'b','LineWidth',1.5,'MarkerSize',8);
            legend('g(x)= TV(x)','dimX/\theta_n');grid on; xlabel('Iteration (n)');hold off;
            saveas(figSum,['./' dirname '/' subdirname '/' name '_gX.png']);
        end
        figTheta=figure;
        plot(results.thetas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);
        xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');title('\theta_n')
        saveas(figTheta,['./' dirname '/' subdirname '/' name '_thetas.png' ]);
        close all;
    end
end

