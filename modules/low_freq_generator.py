'''
Module with a class to create a generator for low-frequency images.
'''

import math
import numpy as np
import time
import matplotlib as mpl 
import matplotlib.pyplot as plt

class LowFreqGenerator(object):
    '''
    A class to create a generator of low-frequency images.
    '''

    def __init__(self, batch_size = 128,  N=4, im_size=64, discounting = 1, constant_components = False):
        '''
        Constructor
        '''
        
        self.N = N
        self.im_size = im_size
        self.batch_size = batch_size
        
        # array with discretized Fourier eigenfunctions
        self._base = np.zeros(shape=(im_size, im_size, 2*N+1, 2*N+1), dtype = np.complex64)

        # Create Fourier coefficients              
        if constant_components:
            self._fourier_components = get_constant_fourier_components(N)
        else:
            self._fourier_components = get_normal_fourier_components(N, discounting)

        # array with discretized Fourier eigenfunctions premultiplied with Fourier coefficients
        self._base_multiplied = np.zeros(shape=(im_size, im_size, 2*N+1, 2*N+1), dtype = np.complex64)
        
        self._set_base()

    def set_fourier_components(self, fourier_components):
        assert fourier_components.shape[0] == fourier_components.shape[1], "Not square matrix for Fourier components"
        assert fourier_components.shape[0] == 2*self.N+1, "Not adequate number of components"
        self._fourier_components = fourier_components
        self._set_base()

    def _set_base(self):
        """ Precompute the base arrays, with discretized Fourier eigenfunctions.
        """
        im_size = self.im_size
        N = self.N
        self._base = np.zeros(shape=(2*N+1, 2*N+1, im_size, im_size), dtype = np.complex64)
        
        _fourier_components_reshaped = np.reshape(self._fourier_components, newshape=(2*N+1, 2*N+1, 1, 1))
        
        for m in range(-N,N+1):
            for n in range(-N, N+1):
                for k in range(im_size):
                    for l in range(im_size):
                        self._base[m,n,k,l] = np.exp( 2 * m * 1j * np.pi * (k/im_size) \
                                                      + 2 * n * 1j * np.pi * (l/im_size) )
                            
        self._base_multiplied = self._base * _fourier_components_reshaped
        
    
    def get_shift_multipliers(self):
        """Get 3darray with batch of matrices with Fourier coefficients for translation.
        """
               
        N = self.N
        
        # Create random shifts 
        shifts = np.random.uniform(size=(self.batch_size, 2))
        
        # 
        shift_multipliers = np.ones(shape=(self.batch_size, 2*N+1, 2*N+1), dtype=np.complex64)
        
        # This part of the code is probably quite slow
        for batch in range(self.batch_size):
            for m in range(-N, N+1):
                for n in range(-N, N+1):
                    shift_x = shifts[batch,0]
                    shift_y = shifts[batch,1]
                    
                    shift_multipliers[batch, m, n] = \
                        np.exp( - 2 * m * np.pi * (1j) * shift_x - 2 * n * np.pi * (1j) * shift_y )
        
        return shift_multipliers, shifts

    def get_shift_multipliers2(self):
        """Get 3darray with batch of matrices with Fourier coefficients for translation.
        """

        N = self.N

        # Create random shifts
        shifts = np.random.uniform(size=(self.batch_size, 2))

        #
        shift_multipliers = np.ones(shape=(self.batch_size, 2 * N + 1, 2 * N + 1), dtype=np.complex64)
        meshes_components = np.meshgrid(range(-N,N+1), range(-N,N+1))
        mesh_x = meshes_components[0]
        mesh_y = meshes_components[1]
        # This part of the code is probably quite slow
        for batch in range(self.batch_size):
            shift_x = shifts[batch, 0]
            shift_y = shifts[batch, 1]
            shift_multipliers[batch, :, :] = \
                np.exp(- 2 * mesh_x * np.pi * (1j) * shift_x - 2 * mesh_y * np.pi * (1j) * shift_y)

        return shift_multipliers, shifts


    def generate(self):
        while True:
            shift_multipliers, _ = self.get_shift_multipliers2()
            
            result = np.tensordot(shift_multipliers, self._base_multiplied, axes=((1, 2), (0, 1)))
            reshaped = result.real.reshape([-1,self.im_size**2])
            normalized = reshaped/np.amax(np.abs(reshaped))
            yield (normalized, normalized)

    def generate_shifts(self):
        while True:
            shift_multipliers, shifts = self.get_shift_multipliers()

            result = np.tensordot(shift_multipliers, self._base_multiplied, axes=((1, 2), (0, 1)))
            reshaped = result.real.reshape([-1, self.im_size ** 2])
            normalized = reshaped / np.amax(np.abs(reshaped))
            yield (normalized, shifts)

def get_normal_fourier_components(N, discounting = 1):
    _fourier_components = np.random.normal(size=(2*N+1, 2*N+1)) \
                                    + 1j * np.random.normal(size=(2*N+1, 2*N+1))
        
    _fourier_components = _fourier_components.astype(np.complex64)
    
    discount_matrix = np.zeros(shape=(2*N+1, 2*N+1), dtype=np.complex64)
    for m in range(-N, N+1):
        for n in range(-N, N+1):
            discount_matrix[m,n] = discounting**(abs(m) + abs(n))
            
    return _fourier_components * discount_matrix

def get_constant_fourier_components(N):
    _fourier_components = np.zeros(shape=(2*N+1, 2*N+1),dtype=np.complex64)
    
    _fourier_components[0,1] = 1
    _fourier_components[1,0] = 1
    
    return _fourier_components

if __name__ == '__main__':
    low_freq_gen = LowFreqGenerator(N=2)
    batch_size = 128
    gen = low_freq_gen.generate()
    
    t0 = time.perf_counter()
    result = next(gen)[0].reshape([-1, low_freq_gen.im_size, low_freq_gen.im_size])
    print("time needed to generate", str(low_freq_gen.batch_size), "images: ", time.perf_counter() - t0)
    
    for n in range(10):
        plt.imshow(result[n,:,:])
        plt.show()
    
