function [tgvNorm] = TGVnorm(x0,th1,th2,num_iter)
    if(nargin<4)
        num_iter=20;
    end
    tau=0.01;
	[x,l12NormZ1,l12NormZ2]=TGVdenoising(x0,th1,th2,tau,num_iter);%primal dual solver slow
    tgvNorm=th1*l12NormZ1+th2*l12NormZ2;
end

