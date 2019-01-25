'''
Created on Dec 6, 2018

@author: jportegi1
'''
from modules import DiffusionVAE
import numpy as np
import tensorflow as tf
import keras.backend as K
import math
# Plotting libraries
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

class DiffusionEmbeddedTorusVAE(DiffusionVAE):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        latent_dim = 3
        self.d = 2.0

        self.c = 3.0 # radius to the center of tube
        self.a = 0.6 # radius of tube
        self.S = self.calculate_curvature
        self.volume = 4 * math.pi * 2 * self.c * self.a
        self.log_prior = np.log(1 / self.volume)
        self.manifold = "torus"
        super(DiffusionEmbeddedTorusVAE,self).__init__(latent_dim, params)


    def calculate_curvature(self, z_samples):
        length = np.sqrt(np.sum(z_samples**2, axis = -1))
        S = (2*length)/(self.a*(self.c+self.a*length))
        return S


    def kl_tensor(self, logt, y):
        if self.constant_t:
            logt = self.log_t_fixed
        proj_matrix = K.constant([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        z_projected = K.dot(y, proj_matrix)
        proj_length = K.sqrt(K.sum(z_projected ** 2, axis=-1))
        scaled_proj_length = (proj_length - self.c) / self.a
        # Note: scalar curvature of 2-torus is twice Gauss curvature
        scalar_curv = 2 * scaled_proj_length / (self.a * (self.c + self.a * scaled_proj_length))
        d = 2  # dimension of manifold
        loss = - 0.5 * d * logt - 0.5 * d \
               + ((d + 4) / 24.) * scalar_curv * K.exp(logt) \
               + K.log(4 * math.pi * 2 * self.c * self.a)
        return loss
    
    def projection(self, z):
        """
        This function takes an input latent variable (tensor) in R^3 and projects it into the chosen
        manifold
        :param z: Input latent variable in R^3
        :return:
        """
        c = 3.0 # radius to center of tube
        a = 0.6 # radius of tube
        proj_matrix = K.constant([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        z_circle = c * tf.nn.l2_normalize(K.dot(z, proj_matrix), dim=-1)
        z_proj = a * tf.nn.l2_normalize(z - z_circle, dim=-1) + z_circle
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
        a = 0.6
        c = 3.0
        theta = 2 * np.pi * np.linspace(0, 1, samples)
        phi = 2 * np.pi * np.linspace(0, 1, samples)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        x = (c + a * np.cos(theta_grid)) * np.cos(phi_grid)
        y = (c + a * np.cos(theta_grid)) * np.sin(phi_grid)
        z = a * np.sin(theta_grid)
        ax.set_xlim([-c, c])
        ax.set_ylim([-c, c])
        ax.set_zlim([-c, c])
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r', alpha=0.1, linewidth=0, shade=True)
        ax.set_aspect("equal")
        plt.savefig(filename)

    def plot_latent_space_ax(self, data, batch_size, ax):
        x_test, y_test = data
        z_mean, _, _ = self.encoder.predict(x_test,
                                            batch_size=batch_size)
        ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=y_test, s = 5)
        samples = 20
        a = 0.6
        c = 3.0
        theta = 2 * np.pi * np.linspace(0, 1, samples)
        phi = 2 * np.pi * np.linspace(0, 1, samples)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        x = (c + a * np.cos(theta_grid)) * np.cos(phi_grid)
        y = (c + a * np.cos(theta_grid)) * np.sin(phi_grid)
        z = a * np.sin(theta_grid)
        ax.set_xlim([-c, c])
        ax.set_ylim([-c, c])
        ax.set_zlim([-c, c])
        #ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r', alpha=0.1, linewidth=0, shade=True)
        ax.set_aspect("equal")
        return ax

    def plot_image_reconstruction(self, batch_size, filename, samples):
        theta = 2 * np.pi * np.linspace(0, 1, samples)
        phi = 2 * np.pi * np.linspace(0, 1, samples)
        combinations = []
        for i in itertools.product(theta, phi):
            combinations.append(i)
        combinations = np.array(combinations)
        coordinates = np.zeros((len(combinations), 3))
        c = 3
        a = 0.6
        coordinates[:, 0] = (c + a * np.cos(combinations[:, 0])) * np.cos(combinations[:, 1])
        coordinates[:, 1] = (c + a * np.cos(combinations[:, 0])) * np.sin(combinations[:, 1])
        coordinates[:, 2] = np.sin(combinations[:, 0])
        decoded = self.decode(coordinates, batch_size)
        # Reshape reconstructions
        images_decoded = decoded.reshape(len(combinations),int(np.sqrt(self.image_size)), int(np.sqrt(self.image_size)))
        # Plot the reconstructed ciphers
        fig = plt.figure(figsize=(10, 10))
        for i in range(samples):
            for j in range(samples):
                ax = fig.add_subplot(samples, samples, j * samples + i + 1)
                ax.imshow(images_decoded[i * samples + j], cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
        plt.savefig(filename, bbox_inches='tight')