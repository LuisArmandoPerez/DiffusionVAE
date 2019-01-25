#!/usr/bin/anaconda3/bin/python3

# Consistency with previous versions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# System imports
import os
import sys

sys.path.append(os.getcwd())

# Plot and numpy
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg') # for plotting in the cluster

from modules import DiffusionSphereVAE,\
                    DiffusionEmbeddedTorusVAE,\
                    DiffusionFlatTorusVAE,\
                    DiffusionRPNVAE,\
                    experiment_parameters,\
                    diffusion_vae_parameters,\
                    experiment,\
                    StandardVAE,\
                    standard_vae_parameters
import modules.dataset_creation
from keras.datasets import mnist



# Reading Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = modules.dataset_creation.flatten_normalize_images(x_train)

num_repetitions = 1
manifolds = np.array(["sphere","torus", "flat_torus","rpn", "rpn2","standard", "standard2"])
var_x_array = np.array([1.0])
repetitions = range(num_repetitions)
for repetition in repetitions:
    for manifold in manifolds:
        for var_x in var_x_array:
            if manifold == "standard":
                latent_dim = 3
            else:
                latent_dim = 2
            if manifold =="rpn":
                d = 3
            else:
                d = 2




            # Instatiate diffusion VAE parameters
            vae_params_dict = {"image_size": x_train.shape[1],
                               "var_x": var_x, "d":d}
            diff_vae_params = diffusion_vae_parameters.DiffusionVAEParams(**vae_params_dict)
            std_params_dict = {"image_size": x_train.shape[1],
                               "var_x": var_x, "latent_dim": latent_dim}
            standard_vae_params = standard_vae_parameters.StandardVAEParams(**std_params_dict)
            # Instantiate diffusion VAE
            if manifold=="sphere":
                vae = DiffusionSphereVAE(diff_vae_params)
            elif manifold=="standard" or manifold == "standard2":
                vae = StandardVAE(standard_vae_params)
            elif manifold=="torus":
                vae = DiffusionEmbeddedTorusVAE(diff_vae_params)
            elif manifold=="flat_torus":
                vae =DiffusionFlatTorusVAE(diff_vae_params)
            elif manifold =="rpn" or manifold == "rpn2":
                vae = DiffusionRPNVAE(diff_vae_params)
            else:
                vae = None


            # Experiment class
            training_params_dict = {"epochs": 500,
                                    "batch_size": 128*5}
            train_vae_params = experiment_parameters.ExperimentParams(**training_params_dict)
            exp = experiment.Experiment(vae, train_vae_params, x_train, os.path.join(os.getcwd(), "mnist/"+manifold+"/"+str(var_x)))

            # Train VAE
            exp.run()
            # Plot outcomes
            exp.plot_outcomes(x_train, y_train)
