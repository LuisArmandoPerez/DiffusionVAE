'''
Created on Dec 7, 2018

@author: jportegi1
'''
import os
import time
import pandas as pd
import numpy as np
from modules import plot_utils


class Experiment(object):
    '''
    classdocs
    '''

    # initial path ?
    # manage csv file / database

    def __init__(self, diffusionVAE, experiment_params, train_data, path):
        '''
        Constructor
        '''
        self.diffusionVAE = diffusionVAE
        self.experiment_params = experiment_params
        self.train_data = train_data
        self.path = os.path.join(path, "output")
        self.csv_record = os.path.join(self.path, "diffusion_vae_experiments.csv")
        self.time_stamp = None

    def run(self):
        '''
        Runs experiment
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)

        self.diffusionVAE.train_vae(self.train_data,
                                    self.experiment_params.epochs,
                                    self.experiment_params.batch_size,
                                    weights_file,
                                    tensorboard_file)

        # Append record of experiments to csv file
        experiment_params_df = self.experiment_params.params_to_df()
        experiment_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df = self.diffusionVAE.diffusion_vae_params.params_to_df()
        diffusion_vae_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df["manifold"] = self.diffusionVAE.manifold
        merged_df = pd.merge(diffusion_vae_params_df, experiment_params_df, on="timestamp")
        if os.path.isfile(self.csv_record):
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=False, index=False)
        else:
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=True, index=False)

    def run_checkpoints(self):
        '''
        Runs experiment
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Checkpoints folder
        weights_dir_checkpoint = os.path.join(weights_dir, self.time_stamp)
        os.makedirs(weights_dir_checkpoint, exist_ok=True)

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)

        self.diffusionVAE.train_vae_checkpoints(self.train_data,
                                    self.experiment_params.epochs,
                                    self.experiment_params.batch_size,
                                    weights_file,
                                    tensorboard_file, weights_dir_checkpoint)

        # Append record of experiments to csv file
        experiment_params_df = self.experiment_params.params_to_df()
        experiment_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df = self.diffusionVAE.diffusion_vae_params.params_to_df()
        diffusion_vae_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df["manifold"] = self.diffusionVAE.manifold
        merged_df = pd.merge(diffusion_vae_params_df, experiment_params_df, on="timestamp")
        if os.path.isfile(self.csv_record):
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=False, index=False)
        else:
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=True, index=False)




    def run_generator(self, low_freq_generator, steps_per_epoch):
        '''
        Runs experiment
        '''
        generator = low_freq_generator.generate()
        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)

        # Fourier_components_file
        components_dir = os.path.join(self.path, "fourier_components")
        os.makedirs(components_dir, exist_ok=True)
        components_file = os.path.join(components_dir, self.time_stamp+'.npy')
        np.save(components_file, low_freq_generator._fourier_components)

        self.diffusionVAE.train_generator_vae(generator, steps_per_epoch,
                                              self.experiment_params.epochs,
                                              weights_file,
                                              tensorboard_file)

        # Append record of experiments to csv file
        experiment_params_df = self.experiment_params.params_to_df()
        experiment_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df = self.diffusionVAE.diffusion_vae_params.params_to_df()
        diffusion_vae_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df["manifold"] = self.diffusionVAE.manifold
        merged_df = pd.merge(diffusion_vae_params_df, experiment_params_df, on="timestamp")
        if os.path.isfile(self.csv_record):
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=False, index=False)
        else:
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=True, index=False)

    def plot_outcomes(self, x_train, y_train, extra_label=""):
        image_dir = os.path.join(self.path, "images")
        os.makedirs(image_dir, exist_ok=True)
        latent_space_image_filename = os.path.join(image_dir, self.time_stamp + "_latent" + extra_label + ".png")
        reconstruction_image_filename = os.path.join(image_dir, self.time_stamp + "reconstruction.png")
        self.diffusionVAE.plot_latent_space((x_train, y_train), 128, latent_space_image_filename)
        self.diffusionVAE.plot_image_reconstruction(128, reconstruction_image_filename, 20)
    def plot_latent(self, x_train, y_train, extra_label=""):
        image_dir = os.path.join(self.path, "images")
        os.makedirs(image_dir, exist_ok=True)
        latent_space_image_filename = os.path.join(image_dir, self.time_stamp + "_latent" + extra_label + ".png")
        reconstruction_image_filename = os.path.join(image_dir, self.time_stamp + "reconstruction.png")
        self.diffusionVAE.plot_latent_space((x_train, y_train), 128, latent_space_image_filename)

    def plot_outcomes_generator(self, low_freq_gen, batches):
        image_dir = os.path.join(self.path, "images")
        os.makedirs(image_dir, exist_ok=True)
        latent_space_image_filename0 = os.path.join(image_dir, self.time_stamp + "_latent1.png")
        latent_space_image_filename1 = os.path.join(image_dir, self.time_stamp + "_latent2.png")
        reconstruction_image_filename = os.path.join(image_dir, self.time_stamp + "reconstruction.png")
        generator = low_freq_gen.generate_shifts()
        data = next(generator)
        x_train = data[0]
        y_train = data[1]
        for batch in range(batches):
            data = next(generator)
            x_train = np.append(x_train, data[0], axis=0)
            y_train = np.append(y_train, data[1], axis=0)
        colors0 = plot_utils.labels_to_circular_colors(y_train[:, 0])
        colors1 = plot_utils.labels_to_circular_colors(y_train[:, 1])
        self.diffusionVAE.plot_latent_space((x_train, colors0), 128, latent_space_image_filename0)
        self.diffusionVAE.plot_latent_space((x_train, colors1), 128, latent_space_image_filename1)
        self.diffusionVAE.plot_image_reconstruction(128, reconstruction_image_filename, 20)

    def plot_example_datapoint_generator(self, low_freq_gen):
        image_dir = os.path.join(self.path, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = os.path.join(image_dir, self.time_stamp + "_datapoint.png")
        generator = low_freq_gen.generate_shifts()
        data = next(generator)
        x_train = data[0]
        plot_utils.plot_datapoint(x_train[0].reshape((low_freq_gen.im_size, low_freq_gen.im_size)), image_filename)

