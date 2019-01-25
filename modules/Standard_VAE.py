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
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.switch_backend('agg')  # for plotting in the cluster

import math

# Tensorflow and keras
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Reshape
from keras.models import Model
from keras import callbacks
from keras.losses import mse
from keras import backend as K
from keras import losses

# Plot and numpy
import numpy as np
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats, special


class StandardVAE():
    def __init__(self, standard_vae_params):

        self.diffusion_vae_params = standard_vae_params
        self.image_size = standard_vae_params.image_size
        self.input_shape = (self.diffusion_vae_params.image_size,)
        self.intermediate_dim = self.diffusion_vae_params.intermediate_dim
        # Added here automatic choice for latent dimension
        self.latent_dim = standard_vae_params.latent_dim
        self.manifold = "standard"

        self.var_x = standard_vae_params.var_x
        self.r_loss = standard_vae_params.r_loss
        self.encoder, self.decoder, self.vae = self.build_network()

        # Distributions and densities
        self.decoding_distribution = stats.multivariate_normal
        self.prior = stats.multivariate_normal
        self.posterior = stats.multivariate_normal


    def build_network(self):
        """
        This function build the variational autoencoder, the encoder and decoder
        :return: encoder, decoder, vae
        """
        print('type of image_size: ', type(self.image_size))
        ################################################################################################################
        # ENCODER
        ################################################################################################################
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = Dense(self.intermediate_dim, activation='relu')(inputs)
        x_2 = Dense(self.intermediate_dim, activation='relu')(x)
        x_3 = Dense(self.intermediate_dim, activation='relu')(x_2)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x_3)
        z_mean = Dense(self.latent_dim, name='z_mean')(x_3)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        # encoder.summary()

        ################################################################################################################
        # DECODING
        ################################################################################################################
        decoder_h1 = Dense(self.intermediate_dim, activation='relu')
        decoder_h2 = Dense(self.intermediate_dim, activation='relu')

        if self.r_loss == "mse":
            outputs_def = Dense(self.image_size)
        elif self.r_loss == "binary":
            outputs_def = Dense(self.image_size, activation='sigmoid')
        else:
            print("Loss not appropriately chosen")
            outputs_def = None

        z_h1 = decoder_h1(z)
        z_h2 = decoder_h2(z_h1)
        outputs = outputs_def(z_h2)
        # STANDALONE DECODING
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        _z_h1 = decoder_h1(latent_inputs)
        _z_h2 = decoder_h2(_z_h1)
        _outputs = outputs_def(_z_h2)
        decoder = Model(latent_inputs, _outputs, name='decoder')

        # decoder.summary()

        ################################################################################################################
        # LOSS FUNCTIONS
        ################################################################################################################
        def r_loss(inputs, outputs):
            if self.r_loss == "mse":
                print("Reconstruction loss is mean squared error")
                se = K.sum(K.square(outputs - inputs), axis=-1)
                loss = 0.5 * (se / self.var_x + self.image_size * np.log(2 * np.pi * self.var_x))
            elif self.r_loss == "binary":
                print("Reconstruction loss is binary cross entropy")
                print("Reconstruction loss is binary cross entropy")
                epsilon = K.epsilon()
                loss = inputs * tf.log(epsilon + outputs) \
                       + (1 - inputs) * tf.log(epsilon + 1 - outputs)
                loss = -tf.reduce_sum(loss, axis=-1)
            else:
                print("Loss not appropriately chosen")
                loss = None
            return loss

        def kl_loss(inputs, outputs):
            loss = 0.5 * tf.reduce_sum(K.exp(z_log_var) + K.square(z_mean) - z_log_var - 1, axis=-1)
            return loss

        def vae_loss(inputs, outputs):
            loss = K.mean(r_loss(inputs, outputs) + kl_loss(inputs, outputs))
            return loss

        def mean_squared_error(inputs, outputs):
            se = K.mean(K.pow(outputs - inputs, 2), axis=-1)
            se = K.mean(se, axis=-1)
            return se

        ################################################################################################################
        # COMPILE VARIATIONAL AUTOENCODER
        ################################################################################################################
        vae = Model(inputs, outputs, name='vae_mlp')
        vae.compile(optimizer='adam', loss=vae_loss, metrics=[r_loss, kl_loss, mean_squared_error])
        # vae.summary()

        return encoder, decoder, vae

    # LAMBDA LAYER FUNCTIONS
    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.

        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)

        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        z_sample = z_mean + tf.multiply(epsilon, tf.exp(0.5*z_log_var))
        return z_sample

    def train_vae(self, train_data, epochs, batch_size, weights, tensorboard_file):
        tensorboard_cb = callbacks.TensorBoard(log_dir=tensorboard_file)
        self.vae.fit(train_data, train_data,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=[tensorboard_cb],
                     verbose=2
                     )
        self.vae.save_weights(weights)

    def load_model(self, weights):
        self.vae.load_weights(weights)

    def encode(self, data, batch_size):
        encoded = self.encoder.predict(data, batch_size=batch_size)[0]
        return encoded

    def decode(self, latent, batch_size):
        decoded = self.decoder.predict(latent, batch_size=batch_size)
        return decoded

    def autoencode(self, x_test, batch_size):
        autoencoded = self.vae.predict(x_test, batch_size=batch_size)[0]
        return autoencoded

    def plot_latent_space(self, data, batch_size, filename):
        if self.latent_dim == 3:
            x_test, y_test = data
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            z_mean, _, _ = self.encoder.predict(x_test,
                                                batch_size=batch_size)
            fig = plt.figure(figsize=(12, 10))
            ax = Axes3D(fig)
            ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=y_test)
            ax.set_xlim([np.amin(z_mean[:, 0]), np.amax(z_mean[:, 0])])
            ax.set_ylim([np.amin(z_mean[:, 1]), np.amax(z_mean[:, 1])])
            ax.set_zlim([np.amin(z_mean[:, 2]), np.amax(z_mean[:, 2])])
            plt.savefig(filename)
        elif self.latent_dim == 2:
            x_test, y_test = data
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            z_mean, _, _ = self.encoder.predict(x_test,
                                                batch_size=batch_size)
            fig = plt.figure(figsize=(12, 10))
            ax = plt.gca()
            ax.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)

            plt.savefig(filename)
        else:
            print("Not implemented for other latent dimensions other than 2 and 3")

    def plot_latent_space_ax(self, data, batch_size, ax):
        if self.latent_dim == 3:
            x_test, y_test = data
            z_mean, _, _ = self.encoder.predict(x_test,
                                                batch_size=batch_size)

            ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=y_test, s = 5)
            min = np.amin(z_mean)
            max = np.amax(z_mean)
            ax.set_xlim([min, max])
            ax.set_ylim([min, max])
            ax.set_zlim([min, max])
            ax.set_aspect("equal")

        elif self.latent_dim == 2:
            x_test, y_test = data
            z_mean, _, _ = self.encoder.predict(x_test,
                                                batch_size=batch_size)

            ax.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, s = 5)
            ax.set_aspect("equal")

        else:
            print("Not implemented for other latent dimensions other than 2 and 3")
        return ax

    def plot_image_reconstruction(self, batch_size, filename, samples):
        limit = 0.02
        if self.latent_dim == 3:
            sampled_line = np.linspace(-limit, limit, samples)
            combinations = []
            for i in itertools.product(sampled_line, sampled_line):
                combinations.append(i)
            coordinates = np.array(combinations)
            coordinates = np.append(coordinates, np.zeros((len(combinations), 1)), axis=-1)
            decoded = self.decode(coordinates, batch_size)
            # Reshape reconstructions
            images_decoded = decoded.reshape(len(combinations), int(np.sqrt(self.image_size)),
                                             int(np.sqrt(self.image_size)))
            # Plot the reconstructed ciphers
            fig = plt.figure(figsize=(10, 10))
            for i in range(samples):
                for j in range(samples):
                    ax = fig.add_subplot(samples, samples, j * samples + i + 1)
                    ax.imshow(images_decoded[i * samples + j], cmap="gray")
                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.savefig(filename, bbox_inches='tight')

        if self.latent_dim == 2:
            sampled_line = np.linspace(-limit, limit, samples)
            combinations = []
            for i in itertools.product(sampled_line, sampled_line):
                combinations.append(i)
            coordinates = np.array(combinations)
            decoded = self.decode(coordinates, batch_size)
            # Reshape reconstructions
            images_decoded = decoded.reshape(len(combinations), int(np.sqrt(self.image_size)),
                                             int(np.sqrt(self.image_size)))
            # Plot the reconstructed ciphers
            fig = plt.figure(figsize=(10, 10))
            for i in range(samples):
                for j in range(samples):
                    ax = fig.add_subplot(samples, samples, j * samples + i + 1)
                    ax.imshow(images_decoded[i * samples + j], cmap="gray")
                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.savefig(filename, bbox_inches='tight')

        else:
            print("Not implemented")

    def plot_prior_reconstruction(self, num_samples, batch_size, filename):
        latent_samples = np.random.normal(0.0, 1.0, (num_samples ** 2, self.latent_dim))
        decoded = self.decode(latent_samples, batch_size=batch_size)
        decoded_reshaped = decoded.reshape((-1, int(np.sqrt(self.image_size)), int(np.sqrt(self.image_size))))
        fig = plt.figure(figsize=(10, 10))
        for i in range(len(decoded_reshaped)):
            plt.subplot(num_samples, num_samples, i + 1)
            plt.imshow(decoded_reshaped[i], cmap="gray")
            plt.xticks([])
            plt.yticks([])
        plt.savefig(filename, bbox_inches='tight')

    def sample_latent_posterior(self, x_test, batch_size=128, num_samples=1):
        assert num_samples >= 1, "Samples must be an integer greater equal than one"
        samples = np.zeros((len(x_test), num_samples, self.latent_dim))
        y, log_var_z, samples[:, 0, :] = self.encoder.predict(x_test, batch_size=batch_size)
        for sample in range(num_samples - 1):
            samples[:, sample + 1, :] = self.encoder.predict(x_test, batch_size=batch_size)[2]
        return y, log_var_z, samples

    def calculate_log_p_xgz(self, data, decoded_z_samples):
        if self.r_loss == "mse":
            data_size = data.shape[1]
            log_exponent = np.sum((data[:, np.newaxis, :] - decoded_z_samples) ** 2, axis=-1) / (2 * self.var_x)
            log_determinant = data_size * np.log(2 * np.pi * self.var_x) / 2
            log_p = -log_determinant - log_exponent
        elif self.r_loss == "binary":
            log_p = np.sum((1 - data[:, np.newaxis, :]) * (np.log(1 - decoded_z_samples + 1e-7)) + np.log(
                1e-7 + decoded_z_samples) * data[:, np.newaxis, :], axis=-1)
        return log_p

    def calculate_log_q_zgx(self, z_samples, log_var_z, encoded):
        covariance_diag = np.exp(log_var_z)
        inverse_covariance_diag = 1.0 / covariance_diag
        log_exponent = np.sum(
            (inverse_covariance_diag[:, np.newaxis, :] * (z_samples - encoded[:, np.newaxis, :]) ** 2), axis=-1) / 2.0
        log_determinant = np.log(np.prod(2 * np.pi*covariance_diag, axis=-1))[:, np.newaxis] / 2.0
        log_q = -log_determinant - log_exponent
        return log_q

    def estimate_log_likelihood(self, data, batch_size, num_samples):
        encoded, log_var_z, z_samples = self.sample_latent_posterior(data, batch_size, num_samples)
        decoded_z_samples = np.zeros((len(data), num_samples, data.shape[1]))
        prior = stats.multivariate_normal
        log_p_z = np.zeros((len(data), num_samples))
        for num_sample in range(num_samples):
            sample = z_samples[:,num_sample,:]
            decoded_z_samples[:,num_sample,:] = self.decode(sample, batch_size = len(data))
            log_p_z[:,num_sample] = prior.logpdf(sample, mean=np.zeros(self.latent_dim), cov=1.0)
        log_p_xgz = self.calculate_log_p_xgz(data, decoded_z_samples)
        log_q_zgx = self.calculate_log_q_zgx(z_samples, log_var_z, encoded)
        weight_estimate = log_p_xgz + log_p_z - log_q_zgx - np.log(num_samples)
        estimate = np.mean(special.logsumexp(weight_estimate, axis=-1), axis=-1)
        return estimate

    def estimate_log_likelihood2(self, data, batch_size, num_samples):
        mean_z, log_var_z, samples = self.sample_latent_posterior(data, 128, 100)
        estimate = np.zeros((len(data), num_samples))
        for num_sample in range(num_samples):
            sample = samples[:, num_sample, :]
            decoded_mean = self.decode(mean_z, 128)
            decoded_sample = self.decode(sample, 128)
            normal_distribution = self.stats.multivariate_normal
            log_posterior = normal_distribution.logpdf(sample[:, :], mean=mean_z[0, :], cov=np.exp(log_var_z[0, :]))
            log_prior = normal_distribution.logpdf(sample[:, :], mean=np.zeros(self.latent_dim), cov=1.0)
            log_encoder = normal_distribution.logpdf(decoded_sample[:, :], mean=decoded_mean[0, :], cov=1.0)
            estimate[:, num_sample] = log_encoder + log_prior - log_posterior
        return np.mean(special.logsumexp(estimate, axis=-1) - np.log(num_samples))

    def evaluate_metrics(self, x_train, batch_size):
        values = self.vae.evaluate(x=x_train, y=x_train, batch_size=batch_size)
        values = np.array(values)
        return values

    def squared_error(self, x_train, batch_size):
        encoded = self.autoencode(x_train, batch_size)
        squared_error = np.mean(np.sum((encoded - x_train) ** 2, axis=-1), axis=-1)
        return squared_error
