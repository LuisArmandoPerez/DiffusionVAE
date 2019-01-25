'''
Created on Dec 6, 2018

@author: jportegi1
'''
from modules import DiffusionVAE
import math
import numpy as np
import tensorflow as tf
import keras.backend as K
# Plotting libraries
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import itertools


class DiffusionSphereVAE(DiffusionVAE):
    '''
    classdocs
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        self.manifold = "sphere"
        self.params = params
        latent_dim = self.params.d + 1
        self.d = self.params.d
        self.S = lambda x: self.d * (self.d - 1)
        self.volume = volume_sphere(self.d)

        # Distributions and densities
        self.decoding_distribution = stats.multivariate_normal
        self.log_prior = np.log(1 / self.volume)
        super(DiffusionSphereVAE, self).__init__(latent_dim, params)

    def kl_tensor(self, logt, y):
        d = self.params.d
        scalar_curv = d * (d - 1)
        volume = self.volume
        loss = -d * logt / 2.0 - d * np.log(2.0 * np.pi) / 2.0 - d / 2.0 + np.log(volume) \
               + (d + 4) * scalar_curv * K.exp(logt) / 24
        return loss

    def projection(self, z):
        """
        This function takes an input latent variable (tensor) in R^3 and projects it into the chosen
        manifold
        :param z: Input latent variable in R^3
        :return:
        """
        z_proj = tf.nn.l2_normalize(z, dim=-1)
        return z_proj

    def plot_latent_space(self, data, batch_size, filename):
        x_test, y_test = data
        root_dir = os.path.split(filename)[0]
        os.makedirs(root_dir, exist_ok=True)
        z_mean, _, _ = self.encoder.predict(x_test,
                                            batch_size=batch_size)
        fig = plt.figure(figsize=(12, 10))
        ax = Axes3D(fig)
        ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=y_test)
        samples = 20

        theta = 2 * np.pi * np.linspace(0, 1, samples)
        phi = np.pi * np.linspace(0, 1, samples)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        x = np.sin(phi_grid) * np.cos(theta_grid)
        y = np.sin(phi_grid) * np.sin(theta_grid)
        z = np.cos(phi_grid)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r', alpha=0.1, linewidth=0, shade=True)
        ax.set_aspect("equal")
        plt.savefig(filename)

    def plot_latent_space_ax(self, data, batch_size, ax):
        x_test, y_test = data
        z_mean, _, _ = self.encoder.predict(x_test,
                                            batch_size=batch_size)
        ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=y_test, s = 5)
        samples = 20

        theta = 2 * np.pi * np.linspace(0, 1, samples)
        phi = np.pi * np.linspace(0, 1, samples)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        x = np.sin(phi_grid) * np.cos(theta_grid)
        y = np.sin(phi_grid) * np.sin(theta_grid)
        z = np.cos(phi_grid)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        #ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r', alpha=0.1, linewidth=0, shade=True)
        ax.set_aspect("equal")
        return ax

    def plot_prior_reconstruction(self, num_samples, batch_size, filename):
        latent_samples = np.random.normal(0.0, 1.0, (num_samples**2, self.latent_dim))
        latent_samples_normalized = latent_samples/np.linalg.norm(latent_samples, axis = -1)[:,np.newaxis]
        decoded = self.decode(latent_samples_normalized, batch_size = batch_size)
        decoded_reshaped = decoded.reshape((-1, int(np.sqrt(self.image_size)), int(np.sqrt(self.image_size))))
        fig = plt.figure(figsize=(10, 10))
        for i in range(len(decoded_reshaped)):
            plt.subplot(num_samples, num_samples, i + 1)
            plt.imshow(decoded_reshaped[i],cmap = "gray")
            plt.xticks([])
            plt.yticks([])
        plt.savefig(filename, bbox_inches='tight')

    def plot_image_reconstruction(self, batch_size, filename, samples):

        theta = 2 * np.pi * np.linspace(0, 1, samples)
        phi = np.pi * np.linspace(0, 1, samples)
        combinations = []
        for i in itertools.product(theta, phi):
            combinations.append(i)
        combinations = np.array(combinations)
        coordinates = np.zeros((len(combinations), 3))
        coordinates[:, 0] = np.cos(combinations[:, 0]) * np.sin(combinations[:, 1])
        coordinates[:, 1] = np.sin(combinations[:, 0]) * np.sin(combinations[:, 1])
        coordinates[:, 2] = np.cos(combinations[:, 1])
        decoded = self.decode(coordinates, batch_size)
        # Reshape reconstructions
        images_decoded = decoded.reshape(len(combinations), int(np.sqrt(self.image_size)),
                                         int(np.sqrt(self.image_size)))
        # Plot the reconstructed ciphers
        fig = plt.figure(figsize=(5, 5))
        for i in range(samples):
            for j in range(samples):
                ax = fig.add_subplot(samples, samples, j * samples + i + 1)
                ax.imshow(images_decoded[i * samples + j], cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
        plt.savefig(filename, bbox_inches='tight')

    def estimate_ll(self, x_test, num_samples, batch_size=128):
        y_values, t_values, samples = self.sample_latent_posterior(x_test, batch_size=batch_size,
                                                                   num_samples=num_samples)
        estimates = np.zeros((len(y_values)))

        for num_y, y in enumerate(y_values):
            decoded = self.decode(samples[num_y, :, :], batch_size=batch_size)
            for num_z, z in enumerate(samples[num_y, :, :]):
                decoding_density = self.decoding_distribution.pdf(x_test[num_y], decoded[num_z], self.params.var_x)
                estimates[num_y] += self.prior_density * decoding_density / self.posterior_density_approximation(
                    t_values[num_y], y, z)

        estimate = np.mean(estimates)
        return estimate

    def posterior_density_approximation(self, t, y, z):
        r = np.linalg.norm(y - z, axis=-1)
        front_coefficient = (1 / (2 * np.pi * t) ** (self.d / 2)) * np.exp(-r ** 2 / (2 * t))
        second_coefficient = self.S * t / (8 * self.d * r ** 2)
        third_coefficient = 3 - self.d + (self.d - 1) * r ** 2 + (self.d - 3) * r * (1 / np.tan(r))
        posterior_density = front_coefficient * (1 + (second_coefficient * third_coefficient))
        return posterior_density


def volume_sphere(d):
    """ Compute volume of d-sphere
     eps = 0.00001

     np.abs(volume_sphere(1) - 2*math.pi) < eps
        True

     np.abs(volume_sphere(2) - 4*math.pi) < eps
        True

     np.abs(volume_sphere(3) - 2*math.pi**2) < eps
        True
    """

    latent_dim = d + 1

    if latent_dim % 2 == 0:
        k = latent_dim / 2
        volume = latent_dim * math.pi ** k / math.factorial(k)
    else:
        k = (latent_dim - 1) / 2
        volume = latent_dim * 2 * math.factorial(k) * (4 * math.pi) ** k / math.factorial(2 * k + 1)

    return volume


if __name__ == "__main__":
    import doctest

    doctest.testmod()
