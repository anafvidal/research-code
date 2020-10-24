% This code is based on the code provided in
% http://www.lx.it.pt/~bioucas/code/demo_sparse_tv.rar by the authors of
% Iordache, M.D., Bioucas-Dias, J.M. and Plaza, A., 2012. Total variation 
% spatial regularization for sparse hyperspectral unmixing. 
% IEEE Transactions on Geoscience and Remote Sensing, 50(11), pp.4484-4502.

function [ Y,X,A,n_em,sigma] = get_hyperspectral_observation(X_compact, p, SNR,bandwidth,min_angle)
[p np]=size(X_compact);
%%%%% Define spectrum for each end member      

% Buid the dictionary with end member spectral information
load USGS_1995_Library.mat
[tmp index] = sort(datalib(:,1));%  order bands by increasing wavelength
Afull =  datalib(index,4:end);
% Reduce the size of the library (select some end members)
% min angle (in degres) defines the minimum angle between 
% any two spectral signatures . 
A = prune_library(Afull,min_angle); % 12  signatures in dictionary
% order  the columns of A by decreasing angles 
[A, index, angles] = sort_library_by_angle(A);
% select p endmembers  from A
supp = 1:p;
M = A(:,supp);%these are the 5 materials that will be present in the observation
[L,p] = size(M);  % L = number of bands; p = number of materials present in example

%%%% Generate hyperspectral measurement 'y'

randn('state',1);% Set rnd generators (for repeatability)
sigma=norm(M*X_compact-mean(mean(M*X_compact)),'fro')/sqrt(np*L*10^(SNR/10));% noise standard deviation
sigma2=sigma^2;
op.sigma=sigma;
noise = sigma*randn(L,np);% generate Gaussian iid noise

% make noise correlated by low pass filtering
% low pass filter (Gaussian)
filter_coef = exp(-(0:L-1).^2/2/bandwidth.^2)';
scale = sqrt(L/sum(filter_coef.^2));
filter_coef = scale*filter_coef;
noise = idct(dct(noise).*repmat(filter_coef,1,np));

%  observed spectral vector created with reduced lib M
Y = M*X_compact + noise;
% Y (224freq * np) = M(224 * 5) X_compact (5 * np)

% create  true X wrt  the library A instead of wrt reduced library M
n_em = size(A,2);%number of possible materials (12)
X = zeros(n_em,np);
X(supp,:) = X_compact;%X has 5625 pixeles with 12 materials each.

end