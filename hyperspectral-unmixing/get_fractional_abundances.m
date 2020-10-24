% This code is based on the code provided in
% http://www.lx.it.pt/~bioucas/code/demo_sparse_tv.rar by the authors of
% Iordache, M.D., Bioucas-Dias, J.M. and Plaza, A., 2012. Total variation 
% spatial regularization for sparse hyperspectral unmixing. 
% IEEE Transactions on Geoscience and Remote Sensing, 50(11), pp.4484-4502.

function [ X_compact,nl,nc] = get_fractional_abundances(p)
    %%%%% Generate original fractional abundances 'X_compact'       
    x1 = eye(p);% pure pixels   
    x2 = x1 + circshift(eye(p),[1 0]); % mixtures with two materials    
    x3 = x2 + circshift(eye(p),[2 0]); % mixtures with three materials   
    x4 = x3 + circshift(eye(p),[3 0]); % mixtures with four  materials   
    x5 = x4 + circshift(eye(p),[4 0]); % mixtures with five  materials
    x6 = [0.1149 0.0741  0.2003 0.2055, 0.4051]';  % background (random mixture)  
    % Normalize
    x2 = x2/2;    x3 = x3/3;    x4 = x4/4;    x5 = x5/5;

    % Build a matrix
    % There will be 26 different combinations of materials. The first 5 are
    % single elements (end members) x1. The second set of 5 are mixtures of 2
    % elements, etc. The last one is a random combination of the 5 elements. 
    xt = [x1 x2 x3 x4 x5 x6];

    % build image of indices to xt
    % We make an image that has indexes that say which combination of materials
    % goes in that pixel. There are 26 different types of "material combination",
    % so we can have 26 different indices. 
    imp = zeros(3);
    imp(2,2)=1;
    imind = [imp*1  imp*2 imp* 3 imp*4 imp*5;
        imp*6  imp*7 imp* 8 imp*9 imp*10;
        imp*11  imp*12 imp*13 imp*14 imp*15;
        imp*16  imp*17 imp* 18 imp*19 imp*20;
        imp*21  imp*22 imp* 23 imp*24 imp*25];

    imind = kron(imind,ones(5));
    imind(imind == 0) = 26;% set backround index
    % Generate fractional abundances for all pixels
    % We use imind to index xt which contains the possible material combinations. 
    % We asign each pixel a material combination thus defining the true image X_compact, 
    % where each pixel xi has a certian combination of materials or end members
    [nl,nc] = size(imind);
    np = nl*nc;     % number of pixels
    for i=1:np
        X_compact(:,i) = xt(:,imind(i));
        %X_compact has dimension 5 X_compact np. It has np columns, and in each column 5 rows 
        %with the abundance of each one of the 5 endmembers
        %imind was matrix but we use is with columns concatenated 
        %X_compact= x11 x12      ....x1_np
        %   x21 x22 x23  ....x2_np
        %   ...
        %   ...
        %   x51 x52 x53  ....x5_np
    end
    %Now we reshape X_compact to arrange the pixels into 2 dimensions and the materials
    %into a third dimension.
    Xim = reshape(X_compact',nl,nc,p); % array with 3 levels. (rwo,col, material%)
   
    
    %  Image of endmember 5 (uncomment for visualising fractional abundances)
    %     figure(1)
    %     imagesc(Xim(:,:,5))
    %     title('Fractional abundance of endmember 5')

    
    end