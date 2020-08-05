function [x,l12NormZ1,l12NormZ2] = TGVprox(x0,th1,th2,lambda,num_iter)
    if(nargin<4)
        num_iter=20;
    end
    tau=0.01;
	[x,l12NormZ1,l12NormZ2]=TGVdenoising(x0,th1*lambda,th2*lambda,tau,num_iter);%primal dual solver slow
end

