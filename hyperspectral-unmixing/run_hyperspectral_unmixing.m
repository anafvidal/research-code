%% Experiment 4.3 - Hyperspectral unmixing with SUnSAL prior
%{
  Experiment 4.3 of the SIAM article [1]:
  Hyperspectral unmixing with TV-SUnSAL prior

  Computes the maximum marignal likelihood estimation of the
  regularisation parameters using Algorithm 2 from [1], i.e.
  the algorithm for separably homogeneous regularisers.
  
  It runs Algorithm 2 a synthetic hyperspectral image, and computes the MAP 
  estimator of the fractional abundances using the estimated values of 
  theta1_EB and theta2_EB, and the SUnSAL-TV solver [2].

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
  inverse problems: an empirical bayesian approach. Part I

  [2] M.-D. Iordache, J. Bioucas-Dias, and A. Plaza, "Total variation spatial
  regularization for sparse hyperspectral unmixing", IEEE Transactions on
  Geoscience and Remote Sensing, vol. PP, no. 99, pp. 1-19, 2012.
  
  Prerequired functions and data:  
  -SUnSAL_tv: solver for sparse unmixing with TV. Available at 
   http://www.lx.it.pt/~bioucas/code/demo_sparse_tv.rar
   Recommendation: add a ';' in line 461 of sunsal_tv.m to avoid output.
  -USGS_1995_Library.mat (provided here but also available at
   http://speclab.cr.usgs.gov/spectral.lib06)
  -SALSA: http://cascais.lx.it.pt/~mafonso/salsa.html We only need this
   because we use the TV norm defined in the SALSA utils. 
  

%}
%  ===================================================================


%% General Setup
clear all;clc;
%Check that USGS_1995_Library.mat is accesible
try
    load USGS_1995_Library.mat;
catch
    error('USGS_1995_Library.mat file missing. Please make sure that this file is available in your MATLAB path.  The file can be obtained from http://speclab.cr.usgs.gov/spectral.lib06');
end
%Check that sunsal solver is in the MATLAB path
if exist('sunsal_tv')==0
    error('SUnSAL solver not found. Please make sure that this solver is added to your MATLAB path.  The code can be obtained from http://www.lx.it.pt/~bioucas/code/demo_sparse_tv.rar');
end
%Check that TVnorm in SALSA package is in the MATLAB path
if exist('TVnorm')==0
    error('TVnorm function not found. Please make sure that the SALSA package is added to your MATLAB path.  The code can be obtained from http://cascais.lx.it.pt/~mafonso/salsa.html');
end

% Set to true to save additional plots
save_logpi_plots= true; % to check stabilisation of the MYULA sampler
save_gx_plot=true; % to check evolution of the regulariser g and the gradient approx

% define random states
rand('state',10);
randn('state',10);
%% Hyperspectral unmixing problem definitions

op.SNR=40; %SNR in dB     
op.p = 5;  % number of end members that will be present in the test image
op.bandwidth = 1000; %% noise bandwidth (iid noise)
op.min_angle = 20; % min_angle (in deg) between any two spectral signatures 
% The larger the min_angle the easier the sparse regression problem

% Get compact representation of fractional abundances
[ X_compact,op.nl,op.nc] = get_fractional_abundances(op.p); 
% Get hyperspectral observation y and spectral signature dictionary A 
[y,X,op.A,op.n_em,op.sigma] = get_hyperspectral_observation(X_compact, op.p, op.SNR,op.bandwidth,op.min_angle);

op.np = op.nl*op.nc;     % number of pixels
dimX=op.n_em*op.np; %dimension of X
op.itersSUnSAL=20; %number of iterations to be used in SUnSAL solver
%% Parameter setup for SAPG algorithm 2
op.samples =50;%run 50 iterations of the sampler
op.burnin=30; % iterations we ignore before taking the average over iterates theta_n

%Init values for theta_1 and theta_2
op.th1_init=10;% 10 theta1_0 initialisation of the SAPG algorithm
op.th2_init=10;% 10 theta2_0 initialisation of the SAPG algorithm
op.max_th1=150;% projection interval theta_1 
op.min_th1=0.01;% projection interval theta_1
op.max_th2=250;% projection interval theta_2 
op.min_th2=10;% projection interval theta_2 

% delta(i) for SAPG algorithm defined as: op.d_scale*( (i^(-op.d_exp)) / numel(x) );
op.d_exp =  0.8;
op.d1_scale = 3e3/(op.max_th1);
op.d2_scale = 3e3/(op.max_th2);
    
%%%% MYULA parameters
op.warmupSteps = 50; % number of warm-up iterations with fixed theta for MYULA sampler

%Lf is the lipschitz constant of the gradient of the smooth part
Lf=1/(op.sigma^2);%we consider that the maximum sing. value of A is
% svMax=1 becuase we use preconditioning to normalize the geometry
%Without preconditioning this would be Lf=(svMax/(sigma^2)).
   
% Compute inverse of AtA for preconditioning 
AtA=op.A'*op.A;
invAtA=inv(AtA);
sqrtInvAta=sqrtm(invAtA);

%Special care needs to be taken when defining lambda with preconditioning:
%when taking the prox step we need to guarantee a step amplitude <1 that is:
% gamma*precondMatrix*(x-prox)/lambda the factor that multiplies the
% gradient has to be <1. When we have precond matrix this factor is
% ~~(gamma/lambda)*maxEval(precondMatrix). This means that when we select
% lambda, we should multiply the value by maxEval(precondMatrix) so that it
% cancels out and we still have sth <1. 
op.lambda=(1/Lf)*max(eig(invAtA));

op.gamma_max=2*1/(Lf+2*max(eig(invAtA))/op.lambda);%we use 2/lambda because we use  2 moreau envs, one for sunsal and one for the positivity constraint.
op.gammaFrac=0.9; % we set gamma=op.gammaFrac*gamma_max
op.gamma=op.gammaFrac*op.gamma_max;%discretisation step MYULA: posterior dist
   
%% Run SAPG Algorithm 2 to compute theta_EB
[theta1_EB,theta2_EB,results]=SAPG_algorithm_2_hyper_unmix(y, op);       
        
%% Solve MAP problem with theta1_EB and theta2_EB          
%Solve for X using SUnSAL-TV solver
imDimensions=[op.nl,op.nc];
[xMAP,res,rmse_i] = sunsal_tv(op.A,y,'MU',0.05,'POSITIVITY','yes','ADDONE','no', ...
                          'LAMBDA_1',theta1_EB*op.sigma^2,'LAMBDA_TV', theta2_EB*op.sigma^2, 'TV_TYPE','iso',...
                          'IM_SIZE',imDimensions,'AL_ITERS',200, 'TRUE_X', X,  'VERBOSE','no');

%Compute MSE with xMAP
mse = 10*log10(norm(X-xMAP,'fro')^2 /dimX);
results.SRE=20*log10(norm(X,'fro')/norm(xMAP-X,'fro')); 
results.mse=mse;
results.xMAP=xMAP;
results.x=X;
results.y=y;
results.options=op;

%% Save Results        
%%%% Create directories 
%subdirID: creates aditional folder with this name to store results        
if isfield(op, 'stopTol')
    subdirID = [char(num2str(op.nl)) 'x' char(num2str(op.nc)) 'x' char(num2str(op.n_em)) '/algo2_tol' char(num2str(op.stopTol)) '_maxIter' char(num2str(op.samples)) '_th1-init=' char(num2str(op.th1_init)) '_th2-init=' char(num2str(op.th2_init)) '_p=' char(num2str(op.p))  '_gamma=' char(num2str(op.gamma))];
else
    subdirID = [char(num2str(op.nl)) 'x' char(num2str(op.nc)) 'x' char(num2str(op.n_em)) '/algo2_maxIter' char(num2str(op.samples)) '_th1-init=' char(num2str(op.th1_init)) '_th2-init=' char(num2str(op.th2_init)) '_p=' char(num2str(op.p)) '_gamma=' char(num2str(op.gamma))];
end
    dirname = 'results/4.3-hyperspectral';
subdirname=['BSNR' char(num2str(op.SNR)) '/' subdirID];
if ~exist(['./' dirname],'dir')
    mkdir('results');
end
if ~exist(['./' dirname '/' subdirname ],'dir')
    mkdir(['./' dirname '/' subdirname ]);
end

% Save data
save(['./' dirname '/' subdirname '/results.mat'], '-struct','results');
close all;


%%%% Generate subplots with orig and recovered end members
scrsz = get(0,'ScreenSize');
fsubplots=figure('Position',[1 1 scrsz(4) scrsz(4)/3.3])    
ax1=subplot(2,6,1)
X_reshaped = reshape(X', op.nl,op.nc,op.n_em);           
    imagesc(X_reshaped(:,:,1))
    title('Original - EM1')  
ax2=subplot(2,6,2)     
    imagesc(X_reshaped(:,:,2))
    title('Original - EM2')  
ax3=subplot(2,6,3)
    imagesc(X_reshaped(:,:,3))
    title('Original - EM3')  
ax4=subplot(2,6,4)
    imagesc(X_reshaped(:,:,4))
    title('Original - EM4')  
ax5=subplot(2,6,5)
    imagesc(X_reshaped(:,:,5))
    title('Original - EM5')  
ax6=subplot(2,6,6)
    imagesc(X_reshaped(:,:,6))
    title('Original - EM6')  
ax7=subplot(2,6,7)
X_estim_reshaped = reshape(xMAP', op.nl,op.nc,op.n_em);           
    imagesc(X_estim_reshaped(:,:,1))
    title('SUnSAL-TV')  
ax8=subplot(2,6,8)
    imagesc(X_estim_reshaped(:,:,2))
    title('SUnSAL-TV')   
ax9=subplot(2,6,9)
    imagesc(X_estim_reshaped(:,:,3))
    title('SUnSAL-TV')   
ax10=subplot(2,6,10)
    imagesc(X_estim_reshaped(:,:,4))
    title('SUnSAL-TV')   
ax11=subplot(2,6,11)
    imagesc(X_estim_reshaped(:,:,5))
    title('SUnSAL-TV')   
ax12=subplot(2,6,12)
    imagesc(X_estim_reshaped(:,:,6))
    title('SUnSAL-TV')  

    subplots = get(fsubplots,'Children'); % Get each subplot in the figure
for i=1:length(subplots) % for each subplot
    caxis(subplots(i),[0,1]);
end
saveas(fsubplots,['./' dirname '/' subdirname '/estimVsOrigEndMembers.png']); 
 
    
%%%% Monitor convergence of the sampler by plotting logPiTrace
if save_logpi_plots
    %For warmup
    if(op.warmupSteps>0)
        % Save images of logpitrace
        figLogPiTraceWarmUp=figure;
        plot(results.logPiTrace_WU_X(2:end),'b','LineWidth',1.5,'MarkerSize',8);   hold on;
        title('  Monitoring convergence during warm-up');
        legend('\propto log p(X^n|y,\theta)','Location','east');
        xlabel('Iteration (n)');grid on; hold off;
        saveas(figLogPiTraceWarmUp,['./' dirname '/' subdirname '/logPiTraceX_warmup.png' ]);
    end
    
    %For SAPG algorithm during theta estimation
    figLogPiTrace=figure;
    plot(results.logPiTraceX(2:end),'b','LineWidth',1.5,'MarkerSize',8); hold on;
    title('Monitoring convergence during SAPG');
    legend('\propto log p(X^n|y,\theta)','Location','east');
    xlabel('Iteration (n)');grid on;hold off;
    saveas(figLogPiTrace,['./' dirname '/' subdirname '/logPiTraceX.png' ]);
end

%%%% Monitor the gradient estimates for theta_1 and theta_2
if save_gx_plot

    % The gradient estimate is given by the difference between each regulariser
    % g_i(X^n) and dim_i/theta^i_n for i=1,2; We plot both terms first for g1
    % and then for g2. Both terms should couple if the algorithm converged.

    figRegL1=figure;
    plot(results.log_g1(1:results.last_samp-1),'b','LineWidth',1.5,'MarkerSize',8);hold on;
    title('Regularization term g_1 - L1 norm');
    plot(dimX./results.th1_list(1:results.last_samp-1),'r','LineWidth',1.5,'MarkerSize',8);
    legend('g_1(X^n)= |X^n|_1' , 'dim_{L1}/\theta^1_n');
     xlabel('Iteration (n)');grid on;hold off;
    saveas(figRegL1,['./' dirname '/' subdirname '/g1_X.png']);
    
    figRegTV=figure;
    plot(results.log_g2(1:results.last_samp-1),'b','LineWidth',1.5,'MarkerSize',8);hold on;
    title('Regularization term g_2 - TV_{vec} norm');
    plot((dimX-1)./results.th2_list(1:results.last_samp-1),'r','LineWidth',1.5,'MarkerSize',8);
    legend('g_2(X^n)= TV_{vec}(X^n)' , 'dim_{TV}/\theta^2_n');
     xlabel('Iteration (n)');grid on;hold off;
    saveas(figRegTV,['./' dirname '/' subdirname '/g2_X.png']);

end

%%%% Plot the evolution of theta^1_n and theta^2_n
figTheta=figure;
plot(results.th1_list(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
plot(results.th2_list(1:results.last_samp),'r','LineWidth',1.5,'MarkerSize',8);
legend('\theta^1_n','\theta^2_n','Location','east');
xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');
title('Evolution of iterates \theta_n^1 and \theta_n^2')
saveas(figTheta,['./' dirname '/' subdirname '/thetas.png' ]);
close all;


%%%% Generate surface plot of SRE in terms of thetv and thL1
try
    loadedResults=load(['snr' char(num2str(op.SNR)) '-scanLambdaRes12materials.mat']);
catch
    error(['The SRE surface has not been precomputed for SNR=' char(num2str(op.SNR)) ])
end
resultsSRE=loadedResults.resultsSRE;
lambdasL1=resultsSRE(:,:,1);
lambdasTV=resultsSRE(:,:,2);
SRE_cls=resultsSRE(:,:,3);
SRE_l1=resultsSRE(:,:,4);
SRE_tv_ni=resultsSRE(:,:,5);
SRE_tv_i=resultsSRE(:,:,6);

[maxSRE_tv_i,pos_maxSRE_tv_i]=max(SRE_tv_i(:));
bestThTV=lambdasTV(pos_maxSRE_tv_i)/op.sigma^2;
bestThL1=lambdasL1(pos_maxSRE_tv_i)/op.sigma^2;
fsurf=figure;
A = axes;
surf(lambdasL1/op.sigma^2,lambdasTV/op.sigma^2,SRE_tv_i);
hold on; 
set(A,'XScale','log');set(A,'YScale','log');
xlabel('theta_{L1}') ;% x-axis label
ylabel('theta_{TV}'); % y-axis label
title(['SNR ' char(num2str(op.SNR)) '  -  SRE surface ']);
scatter3(bestThL1,bestThTV,maxSRE_tv_i, 'filled','green');
scatter3(theta1_EB,theta2_EB,results.SRE, 'filled','blue');legend('SRE surface','Best params','EB Params')
view(A,[-90.3 90])
grid(A,'on')
saveas(fsurf,['./' dirname '/' subdirname '/SREsurface.png']);