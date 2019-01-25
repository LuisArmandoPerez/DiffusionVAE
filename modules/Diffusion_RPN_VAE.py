'''
Created on Dec 6, 2018

@author: jportegi1
'''
from modules import DiffusionVAE
from modules.Diffusion_Sphere_VAE import volume_sphere
import numpy as np
import tensorflow as tf
import keras.backend as K
# Plotting libraries
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

class DiffusionRPNVAE(DiffusionVAE):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        self.manifold = "rpn"
        self.params = params
        latent_dim = self.params.d+1

        self.d = self.params.d
        self.S = lambda x: self.d*(self.d-1)
        self.volume = volume_sphere(self.params.d) / 2
        self.log_prior = np.log(1/self.volume)
        super(DiffusionRPNVAE,self).__init__(latent_dim, params, symmetrize=True)

    
    def kl_tensor(self, logt, y):
        d = self.params.d
        scalar_curv = d*(d-1)
        volume = self.volume
        loss = -d * logt / 2.0 - d * np.log(2.0 * np.pi) /2.0 - d / 2.0 + np.log(volume) \
               + (d + 4) * scalar_curv * K.exp(logt) / 24
        return loss
    
    def projection(self, z):
        """
        This function takes an input latent variable (tensor) in R^3 and projects it into the chosen
        manifold
        :param z: Input latent variable in R^3
        :return:
        """
        z_proj = tf.nn.l2_normalize(z,dim =-1)
        return z_proj



    def plot_latent_space(self, data, batch_size, filename):
        x_test, y_test = data
        root_dir = os.path.split(filename)[0]
        os.makedirs(root_dir, exist_ok=True)
        z_mean, _, _ = self.encoder.predict(x_test,
                                            batch_size=batch_size)


        # Stereographic projection

        if self.params.d == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = Axes3D(fig)
            stereo_proj = stereographic_projection(z_mean)
            ax.scatter(stereo_proj[:, 0], stereo_proj[:, 1], stereo_proj[:, 2], c=y_test)
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

        if self.params.d == 2:
            stereo_proj = stereographic_projection(z_mean)
            ax = plt.scatter(stereo_proj[:, 0], stereo_proj[:, 1], c=y_test)
            samples = 20

            #ax.set_xlim([-1, 1])
            #ax.set_ylim([-1, 1])

            #ax.set_aspect("equal")

        plt.savefig(filename)

    def plot_latent_space_ax(self, data, batch_size, ax):
        x_test, y_test = data
        z_mean, _, _ = self.encoder.predict(x_test,
                                            batch_size=batch_size)

        # Stereographic projection

        if self.params.d == 3:
            stereo_proj = stereographic_projection(z_mean)
            ax.scatter(stereo_proj[:, 0], stereo_proj[:, 1], stereo_proj[:, 2], c=y_test, s = 5)
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
           #00
            #  ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r', alpha=0.1, linewidth=0, shade=True)
            ax.set_aspect("equal")

        if self.params.d == 2:
            stereo_proj = stereographic_projection(z_mean)
            ax.scatter(stereo_proj[:, 0], stereo_proj[:, 1], c=y_test, s = 5)
            ax.set_ylim([-1.02, 1.02])
            ax.set_xlim([-1.02, 1.02])
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            circle = plt.Circle((0, 0), 1.01, color='k', fill=False)
            ax.add_artist(circle)
            ax.set_aspect("equal")
        return ax

    def plot_image_reconstruction(self, batch_size, filename, samples):
        print("Not implemented")


#         theta = 2 * np.pi * np.linspace(0, 1, samples)
#         phi = np.pi * np.linspace(0, 1, samples)
#         combinations = []
#         for i in itertools.product(theta, phi):
#             combinations.append(i)
#         combinations = np.array(combinations)
#         coordinates = np.zeros((len(combinations), 3))
#         coordinates[:, 0] = np.cos(combinations[:, 0]) * np.sin(combinations[:, 1])
#         coordinates[:, 1] = np.sin(combinations[:, 0]) * np.sin(combinations[:, 1])
#         coordinates[:, 2] = np.cos(combinations[:, 1])
#         decoded = self.decode(coordinates, batch_size)
#         # Reshape reconstructions
#         images_decoded = decoded.reshape(len(combinations), int(np.sqrt(self.image_size)), int(np.sqrt(self.image_size)))
#         # Plot the reconstructed ciphers
#         fig = plt.figure(figsize=(10, 10))
#         for i in range(len(images_decoded)):
#             plt.subplot(samples, samples, i + 1)
#             plt.imshow(images_decoded[i])
#             plt.xticks([])
#             plt.yticks([])
#         plt.savefig(filename)


def stereographic_projection(z_values):
    z_upper = np.reshape(-np.sign(z_values[:, -1]), (-1,1)) * z_values
    z_0 = z_upper[:, -1]
    stereo_proj = np.copy(z_upper[:, 0:-1]) / (1 - z_0[:, np.newaxis])
    return stereo_proj

if __name__ == "__main__":
    batch_size = 5
    z = np.random.normal(size=(batch_size, 4))
    z = z/np.reshape(np.linalg.norm(z, axis=-1),(-1,1))
    print(stereographic_projection(z))

