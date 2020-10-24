function ret = tvNormVect(x,nl,nc)
%TVNORMVECT This function computes the total TV norm of an abundance matrix
%It adds up all independent TV norms for each end member abundance 2d
%matrix. 
[n_em,np]=size(x);
ret=0;
    for i=1:n_em
        Ximage = reshape(x',nl,nc,n_em);
        ret=ret+TVnorm( Ximage(:,:,i));
    end 
end

