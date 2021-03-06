# Diffusion Variational Autoencoders

The code consists of two directories: modules and run_scripts. The modules directory contains all the necessary python files needed for using the run_scripts. The run_scripts directory contains the ".py" files that are used for replicating some of the results for the relevant manifolds from the MNIST dataset and the synthetic dataset. 

- run_mnist file: This code runs the diffusion variational autoencoder for the manifolds: $$\mathcal{S}^2$$, flat torus embedded in $$\mathbb{R}^4$$, torus embedded in $$\mathbb{R}^3$$, $$\mathbb{R}\mathbb{P}^3$$, $$\mathbb{R}\mathbb{P}^2$$, $$\mathbb{R}^3$$, $$\mathbb{R}^2$$ with the MNIST dataset. It automatically creates subdirectories with the trained models and plots.
- run_fourier file: This code runs the diffusion variational autoencoder for the manifolds: $$\mathcal{S}^2$$, flat torus embedded in $$\mathbb{R}^4$$, torus embedded in $$\mathbb{R}^3$$, $$\mathbb{R}\mathbb{P}^3$$, $$\mathbb{R}\mathbb{P}^2$$, $$\mathbb{R}^3$$, $$\mathbb{R}^2$$ with the synthetic dataset. It automatically creates subdirectories with the trained models and plots.

This code is an outdated version of the Diffusion Variational Autoencoders paper: 
Perez Rey, L.A., Menkovski, V., Portegies, J.W. (2020). Diffusion Variational Autoencoders. Twenty-Ninth International Joint Conference on Artificial Intelligence.

The updated version can be found in [this link](https://github.com/luis-armando-perez-rey/diffusion_vae).
