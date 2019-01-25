import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def labels_to_circular_colors(labels):
    angular_labels = 2*np.pi*labels/np.amax(labels)
    colors = np.zeros((len(labels),3))
    colors[:,0] = 0.5+np.cos(angular_labels)/2
    colors[:,1] = 0.5+np.sin(angular_labels)/2
    return colors

def plot_datapoint(image, filename):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)


def HSV_embedding(theta, phi):
    result = np.zeros(theta.shape + (3,))

    steps = 10

    theta_rounded = np.round(steps * theta / (2 * np.pi)) / steps
    phi_rounded = 2 * np.pi * np.round(steps * phi / (2 * np.pi)) / steps
    result[..., 0] = 0.65 + 0.1 * np.cos(2 * np.pi * theta_rounded)
    result[..., 1] = 0.7 + 0.2 * np.sin(phi_rounded) + 0.1 * np.sin(2 * np.pi * theta_rounded)
    result[..., 2] = 0.6 + 0.2 * np.cos(phi_rounded) + 0.2 * np.sin(2 * np.pi * theta_rounded)

    return mpl.colors.hsv_to_rgb(result.reshape((-1, 3)))


def HSV_embedding2(theta, phi):
    result = np.zeros(theta.shape + (3,))

    steps = 12

    theta_rounded = np.round(steps * theta / (2 * np.pi)) / steps
    phi_rounded = 2 * np.pi * np.round(steps * phi / (2 * np.pi)) / steps
    result[..., 0] = theta_rounded
    result[..., 1] = 0.7 + 0.3 * np.sin(phi_rounded)
    result[..., 2] = 0.7 + 0.3 * np.cos(phi_rounded)

    return mpl.colors.hsv_to_rgb(result.reshape((-1, 3)))

def yiq_to_rgb(yiq):
    conv_matrix = np.array([[1., 0.956, 0.619],
                            [1., -0.272, 0.647],
                            [1.,-1.106,1.703]])
    return np.tensordot(yiq, conv_matrix, axes=((-1,),(-1)))


def YIQ_embedding(theta, phi):
    result = np.zeros(theta.shape + (3,))

    steps = 10

    rounding = True
    if rounding:
        theta_rounded = 2 * np.pi * np.round(steps * theta / (2 * np.pi)) / steps
        phi_rounded = 2 * np.pi * np.round(steps * phi / (2 * np.pi)) / steps

        theta = theta_rounded
        phi = phi_rounded

    result[..., 0] = 0.5 + 0.1 * np.cos(theta)
    result[..., 1] = 0.15 * np.cos(phi)
    result[..., 2] = 0.15 * np.sin(phi)

    return yiq_to_rgb(result.reshape((-1, 3)))


def YIQ_embedding_2(theta, phi):
    result = np.zeros(theta.shape + (3,))

    steps = 12

    rounding = True
    if rounding:
        theta_rounded = 2 * np.pi * np.round(steps * theta / (2 * np.pi)) / steps
        phi_rounded = 2 * np.pi * np.round(steps * phi / (2 * np.pi)) / steps

        theta = theta_rounded
        phi = phi_rounded

    result[..., 0] = 0.5 + 0.14 * np.cos((theta+phi) * steps / 2)- 0.2*np.sin(phi)
    result[..., 1] = 0.25 * np.cos(phi)
    result[..., 2] = 0.25 * np.sin(phi)

    return yiq_to_rgb(result.reshape((-1, 3)))