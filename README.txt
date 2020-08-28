This code generates the results presented in [1] available at https://arxiv.org/pdf/1911.11709.pdf

To replicate the deblurring experiments in [1] you can run the following scripts:

deblurring/run_deblur_tv.m
deblurring/run_deblur_synthesis_L1.m

To replicate the total generalised variation denoising experiment in [1] run the script
TGV-denoising/run_denoising_tgv.m

Each script contains a detailed explanation of what it does and which dependencies are required. 
The results are stored in a "results" directory. 
The "images" directory contains all the test images. 
After all the test images are processed and the results are stored in the corresponding directory, 
the script "utils/runStats.m" can be used to compute the average MSE and computing time. 
runStats.m is a function that receives as an argument the full path to the folder where all the results (the *.mat files) are stored. 


[1] A. F. Vidal, V. De Bortoli, M. Pereyra, and D. Alain, Maximum
likelihood estimation of regularisation parameters in high-dimensional
inverse problems: an empirical bayesian approach. Part I: Methodology and Experiments
