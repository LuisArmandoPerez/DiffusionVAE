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

plt.switch_backend('agg')  # for plotting in the cluster

from modules import DiffusionSphereVAE, \
    DiffusionEmbeddedTorusVAE, \
    DiffusionFlatTorusVAE, \
    experiment_parameters, \
    diffusion_vae_parameters, \
    experiment, \
    StandardVAE, \
    standard_vae_parameters, \
    LowFreqGenerator, \
    plot_utils, \
    DiffusionRPNVAE

# Iteration arrays
num_repetitions = 1 # number of repetitions for each experiment
components_array = np.array([1]) # number of fourier components used
discounting_array = np.array([1]) # amount of discount for the fourier components
manifolds = np.array(["sphere","torus", "flat_torus","rpn", "rpn2","standard", "standard2"])
var_x_array = np.array([1])
repetitions = np.arange(num_repetitions)

# Calling generator
batch_size = 128
batches = 100
for discount in discounting_array:
    for component in components_array:
        # Create data
        low_freq_gen = LowFreqGenerator(batch_size, N=component, discounting=discount, constant_components=True)
        data = next(low_freq_gen.generate_shifts())
        data = next(low_freq_gen.generate_shifts())
        x_train = data[0]
        y_train = data[1]
        for batch in range(batches):
            data = next(low_freq_gen.generate_shifts())
            x_train = np.append(x_train, data[0], axis=0)
            y_train = np.append(y_train, data[1], axis=0)
        discount = discounting_array[0]
        component = components_array[0]
        for repetition in repetitions:
            for manifold in manifolds:
                for var_x in var_x_array:
                    if manifold == "rpn":
                        d = 3
                    else:
                        d = 2
                    # Instantiate diffusion VAE parameters
                    vae_params_dict = {"image_size": x_train.shape[1],
                                       "var_x": var_x, "constant_t": False,
                                       "log_t_fixed": -2.0, "d": d}
                    if manifold == "standard2":
                        latent_dim = 2
                    else:
                        latent_dim = 3

                    std_vae_params_dict = {"image_size": x_train.shape[1],
                                           "var_x": var_x, "latent_dim": latent_dim}
                    diff_vae_params = diffusion_vae_parameters.DiffusionVAEParams(**vae_params_dict)
                    std_vae_params = standard_vae_parameters.StandardVAEParams(**std_vae_params_dict)
                    # Instantiate diffusion VAE
                    if manifold == "sphere":
                        vae = DiffusionSphereVAE(diff_vae_params)
                    elif manifold == "standard" or manifold == "standard2":
                        vae = StandardVAE(std_vae_params)
                    elif manifold == "torus":
                        vae = DiffusionEmbeddedTorusVAE(diff_vae_params)
                    elif manifold == "flat_torus":
                        vae = DiffusionFlatTorusVAE(diff_vae_params)
                    elif manifold == "rpn2" or manifold == "rpn":
                        vae = DiffusionRPNVAE(diff_vae_params)
                    else:
                        vae = None

                    # Experiment class
                    training_params_dict = {"epochs": 500,
                                            "batch_size": 128 * 5}
                    train_vae_params = experiment_parameters.ExperimentParams(**training_params_dict)
                    exp = experiment.Experiment(vae, train_vae_params, x_train, os.path.join(os.getcwd(),
                                                                                             "sinusoid_components_manifolds/" + manifold + "/" + str(
                                                                                                 component) + "/" + str(
                                                                                                 var_x)))

                    # Train VAE
                    exp.run()

                    # Fourier_components_file
                    components_dir = os.path.join(exp.path, "fourier_components")
                    os.makedirs(components_dir, exist_ok=True)
                    components_file = os.path.join(components_dir, str(exp.time_stamp) + '.npy')
                    np.save(components_file, low_freq_gen._fourier_components)

                    # Plot outcomes
                    colors0 = plot_utils.YIQ_embedding_2(2*np.pi*y_train[:, 0], 2*np.pi*y_train[:,1])
                    exp.plot_outcomes(x_train, colors0, extra_label="0")
                    exp.plot_example_datapoint_generator(low_freq_gen)

