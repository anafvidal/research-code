%% Experiment 4.2.3 - Wavelet deconvolution with synthesis prior
%{
  Experiment 4.2.3 of the SIAM article [1]:
  Wavelet deconvolution with synthesis prior

  Computes the maximum marignal likelihood estimation of the
  regularisation parameter using Algorithm 1 from [1], i.e.
  the algorithm for homogeneous regularisers and scalar theta.
  
  It runs Algorithm 1 for 10 test images, and computes the MAP estimator
  using the estimated value theta_EB and the SALSA solver.

  All results are saved in a <img_name>_results.mat file in a 'results'
  directory. This .mat file contains a structure with:
  
  -execTimeFindTheta: time it took to compute theta_EB in seconds
  -last_samp: number of iteration where the SAPG algorithm was stopped
  -logPiTrace_WU: an array ∝log pi(x_n|y,theta_0) during warm-up
   (up to some unknown proportionality constant)
  -logPiTraceX: an array  ∝log pi(x_n|y,theta_n) for the SAPG
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
  - Rice Wavelet Toolbox: https://github.com/ricedsp/rwt
  
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
%Check that rice-wavelet solver is in the MATLAB path
if exist('daubcqf')==0
    error('Rice Wavelet Toolbox not found. Please make sure that this toolbox is added to your MATLAB path.  The code can be obtained from https://github.com/ricedsp/rwt');
end
% Set to true to save additional plots
save_logpi_plots= true; % to check stabilisation of the MYULA sampler
save_gx_plot=true; % to check evolution of the regulariser g and the gradient approx

%% Parameter Setup
op.samples =3000;%1500; % max iterations for SAPG algorithm to estimate theta
op.stopTol=1e-3; % tolerance in relative change of theta_EB to stop the algorithm. 
%                  Set to 1e-4 for improved accuracy but longer computation
op.burnIn=20;	% iterations we ignore before taking the average over iterates theta_n

op.th_init = 0.01 ; % theta_0 initialisation of the SAPG algorithm
op.min_th=1e-3; % projection interval Theta (min theta)
op.max_th=1; % projection interval Theta (max theta)

% delta(i) for SAPG algorithm defined as: op.d_scale*( (i^(-op.d_exp)) / numel(x) );
op.d_exp =  0.8;
op.d_scale =  0.1/op.th_init;

%MYULA parameters
op.warmup = 0;%300 % number of warm-up iterations with fixed theta for MYULA sampler
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
        [B,BT,H_FFT,HC_FFT]=uniform_blur(length(x),blur_length);
        evMax=max_eigenval(B,BT,size(x),1e-4,1e4,0);%Maximum eigenvalue of operator A.
        
        %%%%%  wavelet representation
        % We assume that xw=WT(x) represents x in an redundant 4-level Haar
        % wavelet representation $\Psi$, with dimension d = 10 * d_y. 
        wav = daubcqf(2); % Haar wavelet
        levels = 4;
        W = @(xw) mirdwt_TI2D(xw, wav, levels); % inverse transform, xw is x in wavelet coefs. representation
        WT = @(x) mrdwt_TI2D(x, wav, levels); % forward transform

        %%%% true value (in wavelet representation)
        WTx = WT(x);
        [Mw, Nw] = size(WTx);
        dimXw=Mw*Nw;
        %%%% function handles
        A = @(xw) B(W(xw));
        AT = @(x) WT(BT(x));
        
        %%%%% observation y
        Bx = B(x);
        sigma = norm(Bx-mean(mean(Bx)),'fro')/sqrt(dimX*10^(op.BSNRdb/10));
        sigma2 = sigma^2; op.sigma=sigma;
        y = Bx + sigma*randn(size(Bx));
        
        %% Experiment setup: 
        % Functions related to Bayesian model
        % We assume a Laplace prior on the the wavelet coeficients xw  
        % with unknown parameter \theta. Accordingly, we have that
        % f_y(xw)= \norm{y-  A(xw)}^2_2/(2\sigma^{2}) and  \g(xw) = |xw|_1.
        % Note that alogorithm 1 will run directly on the wavelet domain so
        % all operators are defined in terms of the wavelet coefs xw
              
        %%%% Regulariser    
        % L1 norm of the wavelet coefs        
        op.g = @(xw) sum(abs(xw(:))); %g(xw)=|xw|_1 
        % Proximal operator of g(xw)=|xw|_1   (soft-thresholding)
        op.proxG = @(xw,lambda,theta) sign(xw).*(max(abs(xw),theta*lambda)-theta*lambda);
        Psi = @(xw,th)  op.proxG(xw,1,th); % define this format for SALSA solver
                
        %%%% Likelihood (data fidelity)  
        op.f = @(xw) (norm(y-A(xw),'fro')^2)/(2*sigma2); % p(y|xw)∝ exp{-op.f(xw)}
        op.gradF = @(xw) real(AT(A(xw)-y)/sigma2);% Gradient of smooth part
        Lf = (evMax/sigma)^2; % define Lipschitz constant of gradient of smooth part
       
        % We use this scalar summary to monitor convergence
        op.logPi = @(xw,theta) -op.f(xw) -theta*op.g(xw);
        
        %%%% Set algorithm parameters that depend on Lf
        op.lambda=lambdaFun(Lf);%smoothing parameter for MYULA sampler
        op.gamma=op.gammaFrac*gamma_max(Lf,op.lambda);%discretisation step MYULA
        
        %% Run SAPG Algorithm 1 to compute theta_EB
        op.X0=WT(y); % Set initial point for MYULA sampler
        % Note that alogorithm 1 will run directly on the wavelet domain:
        % MYULA will sample the wavelet coefficients
        [theta_EB,results]=SAPG_algorithm_1(y,op);
        
        %% Solve MAP problem with theta_EB     
        %Setup for SALSA solver
        global calls;
        calls = 0;
        A = @(xw) callcounter(A,xw);
        AT = @(xw) callcounter(AT,xw);
        inneriters = 1;
        outeriters = 500;%500 is used in salsa demo.
        tol = 1e-4;%taken from salsa demo.
        mu = theta_EB; %CHECK IF mu=theta_EB/10
        muinv = 1/mu;
        filter_FFT = HC_FFT./(abs(H_FFT).^2 + mu).*H_FFT;%Revisar porque no me queda claro lo de abajo
        invLS = @(xw) muinv*( xw - WT( real(ifft2(filter_FFT.*fft2( W(xw) ))) ) );%inversa de (Wtr Btr B W+ mu I)=xw
        invLS = @(xw) callcounter(invLS,xw);
        fprintf('Running SALSA solver...\n')
        % SALSA
        [xwMAP, ~, ~, ~, ~, ~, ~] = ...
            SALSA_v2(y, A, theta_EB*sigma^2,...
            'MU', mu, ...
            'AT', AT, ...
            'True_x', WTx, ...
            'ToleranceA', tol,...
            'MAXITERA', outeriters, ...
            'LS', invLS, ...
            'VERBOSE', 0);
        %Compute MSE with xMAP
        xMAP = W(xwMAP); %inverse transform to recover estimated image
        mse = 10*log10(norm(x-xMAP,'fro')^2 /dimX);
        results.mse=mse;
        results.xMAP=xMAP;
        results.x=x;
        results.y=y;
        
        %% Save Results
        %Create directories
        %subdirID: creates aditional folder with this name to store results
        subdirID = ['algo1_tol' char(num2str(op.stopTol)) '_maxIter' char(num2str(op.samples)) ];
        dirname = 'results/4.2.3-wav-deconv-synthL1';
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
                title('\propto log p(Xw^n|y,\theta) during warm-up');xlabel('Iteration (n)');grid on; hold off;
                saveas(figLogPiTraceWarmUp,['./' dirname '/' subdirname '/' name '_logPiTraceX_warmup.png' ]);
            end
            
            figLogPiTrace=figure;
            plot(results.logPiTraceX(2:end),'r','LineWidth',1.5,'MarkerSize',8); hold on;
            title('\propto log p(Xw^n|y,\theta) in SAPG algorithm');xlabel('Iteration (n)');grid on;hold off;
            saveas(figLogPiTrace,['./' dirname '/' subdirname '/' name '_logPiTraceX.png' ]);
        end
        if save_gx_plot
            figSum=figure;
            plot(results.gXTrace(1:results.last_samp-1),'r','LineWidth',1.5,'MarkerSize',8);hold on;
            plot(dimXw*((results.thetas(1:results.last_samp-1)).^(-1)),'b','LineWidth',1.5,'MarkerSize',8);
            legend('g(xw)= |xw|_1','dimXw/\theta_n');grid on; xlabel('Iteration (n)');hold off;
            saveas(figSum,['./' dirname '/' subdirname '/' name '_gX.png']);
        end
        figTheta=figure;
        plot(results.thetas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);
        xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');title('\theta_n')
        saveas(figTheta,['./' dirname '/' subdirname '/' name '_thetas.png' ]);
        close all;
    end
end

