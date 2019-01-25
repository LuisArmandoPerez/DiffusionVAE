#!/usr/bin/anaconda3/bin/python3
# Consistency with previous versions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# System imports
import os
import sys

sys.path.append(os.getcwd())

# Tensorflow and keras
import tensorflow as tf
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras import callbacks
from keras import backend as K

# Plot and numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import math
plt.switch_backend('agg')  # for plotting in the cluster


class DiffusionVAE():
    def __init__(self, latent_dim, diffusion_vae_params, symmetrize = False):

        self.diffusion_vae_params = diffusion_vae_params
        self.symmetrize = symmetrize # this parameter is necessary for RPN
        self.image_size = diffusion_vae_params.image_size
        self.input_shape = (self.diffusion_vae_params.image_size,)
        self.intermediate_dim = self.diffusion_vae_params.intermediate_dim
        self.steps = self.diffusion_vae_params.steps  # how many steps to take in random-walk sampling
        self.truncation_radius = self.diffusion_vae_params.truncation_radius  # (tanh-implemented) truncation radius for sampling

        self.latent_dim = latent_dim # latent dimension depends on manifold

        self.max_log_t = diffusion_vae_params.max_log_t
        self.min_log_t = diffusion_vae_params.min_log_t
        self.constant_t = diffusion_vae_params.constant_t # boolean flag for constant time regime
        self.log_t_fixed = diffusion_vae_params.log_t_fixed # log time for constant time regime
        self.var_x = diffusion_vae_params.var_x # decoding distribution variance (normal distribution)
        self.r_loss = diffusion_vae_params.r_loss
        self.encoder, self.decoder, self.vae = self.build_network()



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
        if self.diffusion_vae_params.unconstrained_t:
            z_log_t = Dense(1, name="z_log_var")(x_3)
        else:
            z_log_var_pre = Dense(1, name="z_log_var", activation='tanh')(x_3)
            time_interval_length = self.max_log_t - self.min_log_t
            z_log_t = Lambda(lambda x: np.abs(time_interval_length) * x +self.min_log_t + np.abs(time_interval_length)/2)(z_log_var_pre)
        z_mean = Dense(self.latent_dim, name='z_mean')(x_3)
        z_mean_projected = Lambda(self.projection, output_shape=(self.latent_dim,), name="z_projected")(z_mean)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean_projected, z_log_t])

        encoder = Model(inputs, [z_mean_projected, z_log_t, z], name='encoder')
        ################################################################################################################
        # DECODING
        ################################################################################################################
        decoder_h1 = Dense(self.intermediate_dim, activation='relu')
        decoder_h2 = Dense(self.intermediate_dim, activation='relu')
        if self.r_loss=="mse":
            outputs_def = Dense(self.image_size)
        elif self.r_loss=="binary":
            outputs_def = Dense(self.image_size, activation='sigmoid')
        else:
            print("Loss not appropriately chosen")
            outputs_def = None
        # STANDALONE DECODING
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        _z_h1 = decoder_h1(latent_inputs)
        _z_h2 = decoder_h2(_z_h1)
        _outputs_pre = outputs_def(_z_h2)

        def average(args):
            z1, z2 = args
            avg = 0.5 * (z1 + z2)
            return avg

        # Symmetrize routine for RPN manifold
        if self.symmetrize:
            _latent_inputs_reflected = Lambda(lambda s : -s)(latent_inputs)
            _z_h1_reversed = decoder_h1(_latent_inputs_reflected)
            _z_h2_reversed = decoder_h2(_z_h1_reversed)
            _outputs_reversed = outputs_def(_z_h2_reversed)
            _outputs = Lambda(average)([_outputs_pre, _outputs_reversed])
            decoder = Model(latent_inputs, _outputs, name='decoder')
            outputs = decoder(z)
        else:
            z_h1 = decoder_h1(z)
            z_h2 = decoder_h2(z_h1)
            outputs = outputs_def(z_h2)
            decoder = Model(latent_inputs, _outputs_pre, name='decoder')
        ################################################################################################################
        # LOSS FUNCTIONS
        ################################################################################################################
        def r_loss(inputs, outputs):
            """
            Reconstruction loss part of the variational autoencoder
            :param inputs: input data
            :param outputs: output data from the autoencoder
            :return: r_loss tensor
            """
            if self.r_loss == "mse":
                print("Reconstruction loss is mean squared error")
                se = K.sum(K.pow(outputs - inputs, 2), axis=-1)
                loss = 0.5 * (se / self.var_x + self.image_size * np.log(2 * np.pi * self.var_x))
            elif self.r_loss == "binary":
                print("Reconstruction loss is binary cross entropy")
                epsilon = K.epsilon()
                loss = inputs * tf.log(epsilon + outputs) \
                       + (1 - inputs) * tf.log(epsilon + 1 - outputs)
                loss = -tf.reduce_sum(loss, axis=-1)
            else:
                print("Error, no renconstruction chosen")
                loss = None
            return loss

        def mean_squared_error(inputs, outputs):
            """
            Calculates the mean squared error between input data and output data
            :param inputs:
            :param outputs:
            :return: r_loss tensor
            """
            se = K.mean(K.pow(outputs - inputs, 2), axis=-1)
            se = K.mean(se, axis=-1)
            return se


        def kl_loss(inputs, outputs):
            """
            Kullback-Leibler divergence of posterior distribution. No necessary inputs and outputs are needed
            :param inputs :
            :param outputs:
            :return:
            """
            if self.constant_t:
                loss = self.kl_tensor(self.log_t_fixed, z_mean_projected)
            else:
                loss = self.kl_tensor(z_log_t, z_mean_projected)
            return loss

        def vae_loss(inputs, outputs):
            loss = K.mean(r_loss(inputs, outputs) + kl_loss(inputs, outputs))
            return loss

        ################################################################################################################
        # COMPILE VARIATIONAL AUTOENCODER
        ################################################################################################################
        vae = Model(inputs, outputs, name='vae_mlp')
        vae.compile(optimizer='adam', loss=vae_loss, metrics=[r_loss, kl_loss, mean_squared_error])
        return encoder, decoder, vae

    # LAMBDA LAYER FUNCTIONS
    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.

        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)

        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean_projected, z_log_var = args
        z_sample = z_mean_projected
        for k in range(self.steps):
            epsilon = K.random_normal(shape=K.shape(z_mean_projected))
            # Define the step taken
            if self.constant_t:
                step = K.exp(0.5 * self.constant_t) * epsilon / math.sqrt(self.steps)
            else:
                step = K.exp(0.5* z_log_var)*epsilon/math.sqrt(self.steps)
            # Project back to the manifold
            z_sample = self.projection(z_sample + step)
        return z_sample

    def train_vae(self, train_data, epochs, batch_size, weights_file, tensorboard_file):
        """

        :param train_data (numpy array): first dimension corresponds to number of datapoints
        while second dimension corresponds to the size of each the datapoint
        :param epochs (int) number of epochs the diffusion vae is trained
        :param batch_size (int) size of the batch used for training each epoch
        :param weights_file (str) complete path for saving the trained weights
        :param tensorboard_file (str) complete path for saving the tensorboard log
        :return:
        """
        tensorboard_cb = callbacks.TensorBoard(log_dir=tensorboard_file)
        self.vae.fit(train_data, train_data,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=[tensorboard_cb],
                     verbose=2
                     )
        self.vae.save_weights(weights_file)

    def train_vae_checkpoints(self, train_data, epochs, batch_size, weights_file, tensorboard_file, models_filepath):
        """
        Train diffusion variational autoencoder that can
        :param train_data (numpy array): first dimension corresponds to number of datapoints
        while second dimension corresponds to the size of the datapoint
        :param epochs (int) number of epochs the diffusion vae is trained
        :param batch_size (int) size of the batch used for training each epoch
        :param weights_file (str) complete path for saving the trained weights
        :param tensorboard_file (str) complete path for saving the tensorboard log
        :param models_filepath (str) path to where the diffusion vae models are to be saved
        :return:
        """
        checkpoint = callbacks.ModelCheckpoint(models_filepath, verbose=0, save_best_only=False,
                                        save_weights_only=True, mode='auto', period=10)

        tensorboard_cb = callbacks.TensorBoard(log_dir=tensorboard_file)
        self.vae.fit(train_data, train_data,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=[tensorboard_cb, checkpoint],
                     verbose=2
                     )
        self.vae.save_weights(weights_file)


    def train_generator_vae(self, generator, steps_per_epoch, epochs, weights, tensorboard_file):
        """
        Train the diffusion vae whose data is generated from a generator
        :param generator (generator): generator object that can be used with fit_generator
        :param steps_per_epoch:
        :param epochs:
        :param weights:
        :param tensorboard_file:
        :return:
        """
        tensorboard_cb = callbacks.TensorBoard(log_dir=tensorboard_file)
        self.vae.fit_generator(generator, steps_per_epoch, epochs, verbose=1, callbacks=[tensorboard_cb], workers=1,
                               use_multiprocessing=False)
        self.vae.save_weights(weights)

    def load_model(self, weight_file):
        """
        Reload the weights of previously trained models
        :param weight_file (str): path to the stored weights file
        :return:
        """
        self.vae.load_weights(weight_file)

    def encode(self, data, batch_size):
        """
        Encode into the latent space the input data
        :param data (numpy array) first dimension of array corresponds to the number of
        datapoints and the second correspond to the size of each datapoint
        :param batch_size (int):
        :return:
        """
        encoded = self.encoder.predict(data, batch_size=batch_size)[0]
        return encoded

    def encode_time(self, data, batch_size):
        time = np.exp(self.encoder.predict(data, batch_size=batch_size)[1])
        return time

    def decode(self, latent, batch_size):
        decoded = self.decoder.predict(latent, batch_size=batch_size)
        return decoded

    def autoencode(self, x_test, batch_size):
        autoencoded = self.vae.predict(x_test, batch_size=batch_size)
        return autoencoded

    def sample_latent_posterior(self, data, batch_size=128, num_samples=1):
        """
        Sample a certain number of latent variables from the latent space according to the
        posterior with respect to input data
        :param data (numpy array): input data for
        :param batch_size:
        :param num_samples:
        :return:
        """
        assert num_samples>=1 , "Samples must be an integer greater equal than one"
        samples = np.zeros((len(data), num_samples, self.latent_dim))
        y, log_t, samples[:,0,:] = self.encoder.predict(data, batch_size=batch_size)
        t = np.exp(log_t)
        for sample in range(num_samples-1):
            samples[:,  sample+1,:] = self.encoder.predict(data, batch_size=batch_size)[2]
        return y , t, samples


    def evaluate_metrics(self, x_train, batch_size):
        values = self.vae.evaluate(x=x_train, y=x_train, batch_size=batch_size)
        values = np.array(values)
        return values

    def squared_error(self, x_train, batch_size):
        encoded = self.autoencode(x_train, batch_size)
        squared_error = np.mean(np.sum((encoded - x_train) ** 2, axis=-1), axis=-1)
        return squared_error


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



    def calculate_log_q_zgx(self, z_samples, t, encoded):
        """
        Estimate the log_posterior for the z_samples obtained with respect to each of the encoded values
        """
        r = np.linalg.norm(z_samples - encoded[:, np.newaxis, :], axis=-1)
        log_term1 = -0.5*self.d * np.log(2 * np.pi * (t))
        log_term2 = -r ** 2 / (2 * (t))
        coefficient1 = self.S(z_samples) * t / (8 * self.d * (r) ** 2)
        coefficient2 = 3 - self.d + (self.d - 1) * r ** 2 + (self.d - 3) * r / np.tan(r)
        log_term3 = np.log(1 + coefficient1 * coefficient2)
        log_q = log_term1 + log_term2 + log_term3
        return log_q


    def estimate_log_likelihood(self, data, batch_size, num_samples):
        """
        Estimates the log-likelihood with weighted importnace for a given dataset
        with respect to a certain number of samples from the latent space accoding
        to the approximate posterior
        :param data (numpy array): first dimension corresponds to number of datapoints
        while second dimension corresponds to the size of each the datapoint
        :param batch_size (int): size of the batch for producing the samples
        :param num_samples (int): number of samples taken from the latent space according
        to the approximate posterior distribution
        :return: estimate: corresponds to the estimate log-likelihood value
        """
        encoded, t, z_samples = self.sample_latent_posterior(data, batch_size, num_samples)
        decoded_z_samples = np.zeros((len(data), num_samples, data.shape[1]))
        for num_sample in range(num_samples):
            sample = z_samples[:, num_sample, :]
            decoded_z_samples[:, num_sample, :] = self.decode(sample, batch_size=len(data))

        log_p_xgz = self.calculate_log_p_xgz(data,  decoded_z_samples)
        log_q_zgx = self.calculate_log_q_zgx(z_samples, t, encoded)
        log_p_z = self.log_prior
        weight_estimate = log_p_xgz + log_p_z - log_q_zgx - np.log(num_samples)

        estimate = np.mean(special.logsumexp(weight_estimate, axis=-1), axis=-1)
        return estimate


