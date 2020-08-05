function [A,AT,H_FFT,HC_FFT] = uniform_blur(tot_length,blur_length)
%UNIFORM_BLUR Summary of this function goes here
%   Detailed explanation goes here
%%%% function handle for uniform blur operator 
h = ones(1,blur_length);
lh = length(h);
h = h/sum(h);
h = [h zeros(1,tot_length-blur_length)];
h = cshift(h,-(lh-1)/2);
h = h'*h;
H_FFT = fft2(h);
HC_FFT = conj(H_FFT);

A = @(x) real(ifft2(H_FFT.*fft2(x)));
AT = @(x) real(ifft2(HC_FFT.*fft2(x)));
end

