'''
Created on Dec 6, 2018

@author: jportegi1
'''
from modules import DiffusionVAE
import numpy as np
import tensorflow as tf
import keras.backend as K
# Plotting libraries
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import itertools

class DiffusionFlatTorusVAE(DiffusionVAE):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        self.manifold = "flat_torus"
        latent_dim = 4
        self.d = 2.0
        self.S = lambda x: 0.0
        self.volume = 4*np.pi**2
        self.log_prior = np.log(1/self.volume)
        self.decoding_distribution = stats.multivariate_normal
        super(DiffusionFlatTorusVAE,self).__init__(latent_dim, params)
    
    def kl_tensor(self, logt, y):
        loss = -0.5 * logt * self.d - 0.5 * self.d + np.log(4*np.pi**2)
        return loss
    
    def projection(self, z):
        """
        This function takes an input latent variable (tensor) in R^3 and projects it into the chosen
        manifold
        :param z: Input latent variable in R^3
        :return:
        """
        proj_matrix1 = K.constant([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 0],[0, 0 , 0, 0]])
        proj_matrix2 = K.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        z_proj1 = tf.nn.l2_normalize(K.dot(z, proj_matrix1), dim = -1)
        z_proj2 = tf.nn.l2_normalize(K.dot(z, proj_matrix2), dim=-1)
        z_proj = z_proj1+z_proj2
        return z_proj
    def plot_latent_space(self, data, batch_size, filename):
        x_test, y_test = data
        root_dir = os.path.split(filename)[0]
        os.makedirs(root_dir, exist_ok=True)
        z_mean, _, _ = self.encoder.predict(x_test,
                                            batch_size=batch_size)
        fig = plt.figure(figsize=(12, 10))
        ax = plt.gca()
        angle1 = np.arctan2(z_mean[:,1], z_mean[:,0])
        angle2 = np.arctan2(z_mean[:,3], z_mean[:,2])
        ax.scatter(angle1, angle2, c=y_test)
        ax.set_aspect("equal")
        plt.savefig(filename)

    def plot_prior_reconstruction(self, num_samples, batch_size, filename):
        angles = 2*np.pi*np.random.uniform(0.0, 1.0, (num_samples**2, 2))
        projected = np.zeros((len(angles), 4))
        projected[:,0] = np.cos(angles[:,0])
        projected[:,1] = np.sin(angles[:,0])
        projected[:,2] = np.cos(angles[:,1])
        projected[:,3] = np.sin(angles[:,1])
        decoded = self.decode(projected, batch_size = batch_size)
        decoded_reshaped = decoded.reshape((-1, int(np.sqrt(self.image_size)), int(np.sqrt(self.image_size))))
        fig = plt.figure(figsize=(10, 10))
        for i in range(len(decoded_reshaped)):
            plt.subplot(num_samples, num_samples, i + 1)
            plt.imshow(decoded_reshaped[i],cmap = "gray")
            plt.xticks([])
            plt.yticks([])
        plt.savefig(filename, bbox_inches='tight')

    def plot_latent_space_ax(self, data, batch_size, ax):
        x_test, y_test = data
        z_mean, _, _ = self.encoder.predict(x_test,
                                            batch_size=batch_size)
        angle1 = np.arctan2(z_mean[:,1], z_mean[:,0])
        angle2 = np.arctan2(z_mean[:,3], z_mean[:,2])
        ax.scatter(angle1, angle2, c=y_test, s = 5)
        ax.set_aspect("equal")
        # ax.xaxis.set_ticks(np.linspace(-np.pi, np.pi, 5))
        # ax.yaxis.set_ticks(np.linspace(-np.pi, np.pi, 5))
        # ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        # ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        # ax.tick_params(labelsize=40)
        # ax.set_xlabel(r"$\theta$", fontsize=40)
        # ax.set_ylabel(r"$\varphi$", fontsize=40, rotation=0)
        # Ax2
        # ax2 = ax.twinx()
        #
        # ax2.set_ylabel(r"$\wedge$", fontsize=100, rotation=0)
        # ax2.yaxis.set_label_coords(0.99, 0.57)
        # ax2.set_yticks([])
        #
        # # Ax
        # ax.set_xlabel(r"$\gg$", fontsize=100, rotation=0, position=(0.5, 20))
        # ax.xaxis.set_label_coords(0.5, 0.081)
        # ax.set_ylabel(r"$\wedge$", fontsize=100, rotation=0)
        # ax.yaxis.set_label_coords(-0.01, 0.4)
        # ax.set_title(r"$\gg$", fontsize=100, rotation=0, position=(0.5, 0.935))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
        #ax.grid()
        return ax


    def plot_image_reconstruction(self, batch_size, filename, samples):
        # Fix this
        theta =  np.pi * np.linspace(-1, 1, samples)
        phi = np.pi * np.linspace(-1, 1, samples)
        combinations = []
        for i in itertools.product(theta, phi):
            combinations.append(i)
        combinations = np.array(combinations)
        coordinates = np.zeros((len(combinations), 4))

        coordinates[:, 0] = np.cos(combinations[:, 0])
        coordinates[:, 1] = np.sin(combinations[:, 0])
        coordinates[:, 2] = np.cos(combinations[:, 1])
        coordinates[:, 3] = np.sin(combinations[:, 1])
        decoded = self.decode(coordinates, batch_size)
        # Reshape reconstructions
        images_decoded = decoded.reshape(len(combinations), int(np.sqrt(self.image_size)), int(np.sqrt(self.image_size)))
        # Plot the reconstructed ciphers
        fig = plt.figure(figsize=(5, 5))

        ax0 = fig.add_subplot(111)
        ax0.spines['top'].set_color('none')
        ax0.spines['bottom'].set_color('none')
        ax0.spines['left'].set_color('none')
        ax0.spines['right'].set_color('none')
        # # Ax2
        # ax2 = ax0.twinx()
        #
        # ax2.set_ylabel(r"$\wedge$", fontsize=100, rotation=0)
        # ax2.yaxis.set_label_coords(1.05, 0.57)
        # ax2.set_yticks([])
        #
        # # Ax
        # ax0.set_xlabel(r"$\gg$", fontsize=100, rotation=0, position=(0.5, 20))
        # ax0.xaxis.set_label_coords(0.5, 0.009)
        # ax0.set_ylabel(r"$\wedge$", fontsize=100, rotation=0)
        # ax0.yaxis.set_label_coords(-0.08, 0.4)
        # ax0.set_title(r"$\gg$", fontsize=100, rotation=0, position=(0.5, 1.0))
        ax0.set_xticks([])
        ax0.set_yticks([])

        for i in range(samples):
            for j in range(samples):
                ax = fig.add_subplot(samples, samples, j * samples + i + 1)
                ax.imshow(images_decoded[i * samples + j], cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])


        plt.savefig(filename, bbox_inches = "tight")