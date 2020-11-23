This code generates the results presented in [1] available at https://epubs.siam.org/doi/pdf/10.1137/20M1339829 or https://arxiv.org/pdf/1911.11709.pdf for the arxiv version.

To replicate the deblurring experiments in [1] you can run the following scripts:

deblurring/run_deblur_tv.m
deblurring/run_deblur_synthesis_L1.m


To replicate the total generalised variation denoising experiment in [1] run the script
TGV-denoising/run_denoising_tgv.m

To replicate the hyperspectral unmixing experiment in [1] run:
hyperspectral-unmixing/run_hyperspectral_unmixing.m

Each script contains a detailed explanation of what it does and which dependencies are required. 
The results are stored in a "results" directory. 
The "images" directory contains all the test images. 

For the deblurring and denoising experiments, after all the test images have been processed and 
the results have been stored in the corresponding directory, the function "utils/runStats.m" can 
be used to compute the average MSE and computing time. 
runStats.m is a function that receives as an argument the full path to the folder where all the results (the *.mat files) are stored. 


[1] A. F. Vidal, V. De Bortoli, M. Pereyra, and A. Durmus (2020). Maximum Likelihood Estimation of Regularization Parameters in High-Dimensional Inverse Problems: An Empirical Bayesian Approach Part I: Methodology and Experiments. SIAM Journal on Imaging Sciences, 13(4), 1945-1989.
