
%% Experiment 4.4 - Denoising with a Total-Generalised-Variation prior
%{
  Experiment 4.4 of the SIAM article [1]:
  Denoising with a Total-Generalised-Variation prior

  Computes the maximum marignal likelihood estimation of the
  regularisation parameter using Algorithm 3 from [1], i.e.
  the most general algorithm for unknown partition function.
  
  It runs Algorithm 3 for 10 test images, and computes the MAP estimator
  using the estimated values of theta1_EB and theta2_EB, and the TGV denoiser 
  implemented by Laurent Condat, Version 1.0, Oct. 12, 2016 available at:
  https://www.gipsa-lab.grenoble-inp.fr/~laurent.condat/download/TGVdenoise.m.

  All results are saved in a <img_name>_results.mat file in a 'results'
  directory organised by SNR level. 
  This .mat file contains a structure with:
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
  -options: structure with all the options used to run the algorithm
  -tgvnorm_x: saves the regulariser evolution through iterations g(x_n)
  -tgvnorm_u: saves the regulariser evolution through iterations g(u_n)
  -mse: mean squarred error computing x_MAP with theta1_EB and theta2_EB
  -xMAP: estimated x_MAP computed using theta1_EB and theta2_EB
  -x: original image
  -y: observations (blurred and noisy image)

  [1] A. F. Vidal, V. De Bortoli, M. Pereyra, and D. Alain, Maximum
  likelihood estimation of regularisation parameters in high-dimensional
  inverse problems: an empirical bayesian approach. Part I:Methodology and 
  Experiments.
  
  Prerequired functions:  
  - TGV primal-dual denoiser (already included in this directory)

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

% Set to true to save additional plots
save_logpi_plots= true; % to check stabilisation of the MYULA sampler
save_gx_plot=true; % to check evolution of the regulariser g and the gradient approx

%To reduce the computing times, we use a cropped image in Algorithm 3.
%Set size of image to be cropped. Comment for working with full images
op.cropSize=255;
op.cropImage=[op.cropSize op.cropSize op.cropSize op.cropSize];

%% Parameter Setup
op.samples =2000;%if stop criteria is defined this is taken as max samples
op.stopTol=1e-3; %relative change in theta. normally e-4 or e-3
op.burnin=20; % iterations we ignore before taking the average over iterates theta_n

%Init values for theta_1 and theta_2
op.th1_init=10;% theta1_0 initialisation of the SAPG algorithm
op.th2_init=10;% theta2_0 initialisation of the SAPG algorithm
op.max_th1=100;% projection interval theta_1 
op.max_th2=100;% projection interval theta_2 
op.min_th1=1e-4;% projection interval theta_1
op.min_th2=1e-4;% projection interval theta_2 

% delta(i) for SAPG algorithm defined as: op.d_scale*( (i^(-op.d_exp)) / numel(x) );
op.d_exp =  0.8;
op.d1_scale = 20/(op.th1_init);
op.d2_scale = 20/(op.th2_init);
    
%MYULA parameters
op.thinningU =6;%thinning factor for prior chain (i.e., iterations per sample) 
op.warmupSteps = 25;% number of warm-up iterations with fixed theta for MYULA sampler
op.lambdaMax =1; % max smoothing parameter for MYULA
lambdaFun= @(Lf) min((5/Lf),op.lambdaMax);
gamma_max = @(Lf,lambda) 1/(Lf+(1/lambda));%Lf is the lipschitz constant of the gradient of the smooth part
op.gammaFrac=0.98; % we set gamma=op.gammaFrac*gamma_max

randn('state',1); % Set rnd generators (for repeatability)
%% RUN EXPERIMENTS
for bsnr= [8  ]
    op.BSNRdb=bsnr;    
    for i_im = 1:length(testImg)    
        fprintf('SNR=%d\n',bsnr); op.BSNRdb=bsnr;
        filename=testImg{i_im};
        %% Generate a noisy and blurred observation vector 'y'
        %%%%%  original image 'x'
        fprintf(['File:' filename '\n']);
        x = double(imread(filename))/255;%rescale from 0-255 to 0-1. Required by solver
        dimX = numel(x);                  
        %%%%% observation y        
        sigma = norm(x-mean(mean(x)),'fro')/sqrt(dimX*10^(op.BSNRdb/10));
        sigma2 = sigma^2; op.sigma=sigma;
        y = x + sigma*randn(size(x));
        
        %%%%%  crop image if cropImage option is set
        % To keep computing times shorter theta can be estimated on a
        % smaller fragment of the original image.
        if isfield(op, 'cropImage')
            x_crop = imcrop(x,op.cropImage);%test small im   
            y_crop = imcrop(y,op.cropImage);
        else
            op.cropSize=0;
        end
        dimXcrop=numel(x_crop);
        
        %%%% Set algorithm parameters that depend on Lf
        Lf = (1/sigma)^2; % define Lipschitz constant of gradient of smooth part
        op.lambdaX=lambdaFun(Lf);%smoothing parameter for MYULA sampler: posterior dist
        op.gammaX=op.gammaFrac*gamma_max(Lf,op.lambdaX);%discretisation step MYULA: posterior dist
        op.lambdaU= op.lambdaX; %smoothing parameter for MYULA sampler: prior dist
        op.gammaU =0.9* op.lambdaU; %discretisation step MYULA: prior dist
      
        %% Run SAPG Algorithm 1 to compute theta_EB
        [theta1_EB,theta2_EB,results]=SAPG_algorithm_3_denois_tgv(y_crop, op);       
        
        %% Solve MAP problem with theta1_EB and theta2_EB          
        xMAP=TGVprox(y,theta1_EB,theta2_EB,sigma^2,120);
        
        %Compute MSE with xMAP
        mse = 10*log10(norm(x-xMAP,'fro')^2 /dimX);
        results.mse=mse;          
        results.xMAP=xMAP;
        results.x=x;
        results.y=y;
        
        %% Save Results        
        %Create directories 
        %subdirID: creates aditional folder with this name to store results        
        if isfield(op, 'stopTol')
            subdirID = ['algo3_tol' char(num2str(op.stopTol)) '_maxIter' char(num2str(op.samples)) '_th1-init=' char(num2str(op.th1_init)) '_th2-init=' char(num2str(op.th2_init)) '_th-min=' char(num2str(op.min_th1))];
        else
           subdirID = ['algo3_maxIter' char(num2str(op.samples)) '_th1-init=' char(num2str(op.th1_init)) '_th2-init=' char(num2str(op.th2_init)) '_th-min=' char(num2str(op.min_th1))];
        end
            dirname = 'results/4.4-denois-TGV';
        subdirname=['BSNR' char(num2str(op.BSNRdb)) '/' subdirID];
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
            if(op.warmupSteps>0)
                % Save images of logpitrace
                figLogPiTraceWarmUp=figure;
                plot(results.logPiTrace_WU_X(2:end),'b','LineWidth',1.5,'MarkerSize',8);   hold on;
                plot(results.logPiTrace_WU_U(2:end),'r','LineWidth',1.5,'MarkerSize',8); 
                title('  Monitoring convergence during warm-up');
                legend('\propto log p(X^n|y,\theta)','\propto log p(U^n|\theta)','Location','east');
                xlabel('Iteration (n)');grid on; hold off;
                saveas(figLogPiTraceWarmUp,['./' dirname '/' subdirname '/' name '_logPiTraceX_warmup.png' ]);
            end
            
            figLogPiTrace=figure;
            plot(results.logPiTraceX(2:end),'b','LineWidth',1.5,'MarkerSize',8); hold on;
            plot(results.logPiTraceU(2:end),'r','LineWidth',1.5,'MarkerSize',8); 
            title('Monitoring convergence during SAPG');
            legend('\propto log p(X^n|y,\theta)','\propto log p(U^n|\theta)','Location','east');
            xlabel('Iteration (n)');grid on;hold off;
            saveas(figLogPiTrace,['./' dirname '/' subdirname '/' name '_logPiTraceX.png' ]);
        end
        if save_gx_plot
            figSum=figure;
            plot(results.tgvnorm_x(1:results.last_samp-1),'b','LineWidth',1.5,'MarkerSize',8);hold on;
            plot(results.tgvnorm_u(1:results.last_samp-1),'r','LineWidth',1.5,'MarkerSize',8);
            legend('g(X^n)= TGV(X^n)','g(U^n)= TGV(U^n)');grid on; xlabel('Iteration (n)');hold off;
            saveas(figSum,['./' dirname '/' subdirname '/' name '_gX.png']);
        end
        figTheta=figure;
        plot(results.th1_list(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
        plot(results.th2_list(1:results.last_samp),'r','LineWidth',1.5,'MarkerSize',8);
        legend('\theta^1_n','\theta^2_n','Location','east');
        xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');
        title('Evolution of iterates \theta_n^1 and \theta_n^2')
        saveas(figTheta,['./' dirname '/' subdirname '/' name '_thetas.png' ]);
        close all;
        
        
        
        
        
        
        
    end
end
