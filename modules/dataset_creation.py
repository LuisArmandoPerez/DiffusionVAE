import numpy as np
import itertools


def flatten_normalize_images(images):
    image_size = images.shape[1] ** 2
    x_train = np.reshape(images, [-1, image_size])
    x_train = x_train.astype('float32') / np.amax(x_train)
    return x_train


def binarize(data, seed):
    assert np.amax(data) <= 1.0 and np.amin(data) >= 0.0, "Values not normalized"
    np.random.seed(seed)
    binarized = (np.random.uniform(0.0, 1.0, data.shape) < data).astype(int)
    return binarized


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def circular_shift_image(image: np.ndarray, w_pix_shift: int, h_pix_shift: int):
    """
        Takes a numpy array and shifts it it periodically along the
        width and the height
        :param image: input image, must be at least a 2D numpy array
        :param w_pix_shift: number of pixels that the image is shifted in the width
        :param h_pix_shift: number of pixels that the image is shifted in the height
        :return: numpy array of the shifted image
        """
    # Shift the image along the width
    shifted_image = np.roll(image, w_pix_shift, axis=1)
    # Shift the image along the height
    shifted_image = np.roll(shifted_image, h_pix_shift, axis=0)
    return shifted_image


def combinations_circular_shift(image: np.ndarray):
    (height, width) = image.shape
    shifted_images = np.zeros((height * width, height, width))
    shifts = np.zeros((height * width, 2))
    for i in range(height):
        for j in range(width):
            shifted_images[i * width + j] = circular_shift_image(image, i, j)
            shifts[i * width + j, 0] = i
            shifts[i * width + j, 1] = j
    return shifts, shifted_images


def sinsuoid_image_random(num_samples, n_T, omega_values):
    phases = np.random.uniform(0, 1, (num_samples, 2)) * 2 * np.pi
    space_linspace = np.linspace(0, 1, n_T)
    # Create all possible combinations of phi_1, phi_2
    sinusoid_images = np.zeros((n_T, n_T, len(phases)))

    # Create spatial mesh
    spatial_mesh = np.meshgrid(space_linspace, space_linspace)

    # Generate signals for each combination
    for num_mesh, mesh_dimension in enumerate(spatial_mesh):
        # Omega*dimension
        mesh_expanded_dim = omega_values[num_mesh] * mesh_dimension[:, :, np.newaxis]
        repeated_volume = np.repeat(mesh_expanded_dim, repeats=len(phases), axis=2)
        sinusoid_images += np.sin(np.add(repeated_volume, phases[:, num_mesh]))
    sinusoid_images = np.swapaxes(sinusoid_images, 2, 0)
    return phases, sinusoid_images


def uniform_component_sineimage(n_T, num_components):
    num_samples = 1000
    space_linspace = np.linspace(0, 1, n_T)
    phases = np.random.uniform(0, 1, (num_samples, 2))
    # Omega Volume
    components = np.array(range(-num_components, num_components + 1, 1))
    omega_combinations = []
    for i, j in itertools.product(components, components):
        omega_combinations.append((i, j))
    spatial_meshes = np.meshgrid(space_linspace, space_linspace)
    omega_combinations = np.array(omega_combinations)

    combinations_volume = np.ones((n_T, n_T, len(omega_combinations), num_samples)) + 0j

    for i in range(2):
        omega_volumex = omega_combinations[:, i, np.newaxis, np.newaxis, np.newaxis]
        omega_volumex = np.repeat(omega_volumex, repeats=n_T, axis=1)
        omega_volumex = np.repeat(omega_volumex, repeats=n_T, axis=2)
        omega_volumex = np.repeat(omega_volumex, repeats=num_samples, axis=3)
        omega_volumex = np.swapaxes(omega_volumex, 0, 2)

        spatial_mesh = spatial_meshes[i]
        mesh_expanded_dim = spatial_mesh[:, :, np.newaxis, np.newaxis]
        mesh_expanded_dim = np.repeat(mesh_expanded_dim, repeats=len(omega_combinations), axis=2)
        mesh_expanded_dim = mesh_expanded_dim - 0.5
        mesh_expanded_dim = np.repeat(mesh_expanded_dim, repeats=num_samples, axis=3)

        phases_volumex = phases[:, i, np.newaxis, np.newaxis, np.newaxis]
        phases_volumex = np.repeat(phases_volumex, repeats=n_T, axis=1)
        phases_volumex = np.repeat(phases_volumex, repeats=n_T, axis=3)
        phases_volumex = np.repeat(phases_volumex, repeats=len(omega_combinations), axis=2)
        phases_volumex = np.swapaxes(phases_volumex, 0, 3)

        combinations_volume *= np.exp(2 * omega_volumex * np.pi * (mesh_expanded_dim + phases_volumex) * 1j)
    combinations = np.sum(combinations_volume, axis=2).real
    combinations = np.swapaxes(combinations, 2, 0)
    combinations = combinations / np.amax(combinations)
    return phases, combinations


def sinusoid_image_phase_combination(num_samples1, num_samples2, n_T, omega_values):
    """
    This function produces an array where each row corresponds to a sinusoidal signal with a given phase and
    angular frequency omega. The columns represent the time sampling from the interval [0,1].
    :param phases: Vector with the phases to be used
    :param n_T: Number of elements in the partition of the interval [0,1]
    :param omega: Angular frequency
    :return: np.array with shape (len(phases),n_T)
    """

    phases1 = np.linspace(0, 1, num_samples1) * 2 * np.pi
    phases2 = np.linspace(0, 1, num_samples2) * 2 * np.pi
    # Sampling from phase and space
    space_linspace = np.linspace(0, 1, n_T)
    # Create all possible combinations of phi_1, phi_2
    phase_combinations = np.array(list(itertools.product(phases1, phases2)))
    sinusoid_images = np.zeros((n_T, n_T, len(phase_combinations)))

    # Create spatial mesh
    spatial_mesh = np.meshgrid(space_linspace, space_linspace)

    # Generate signals for each combination
    for num_mesh, mesh_dimension in enumerate(spatial_mesh):
        # Omega*dimension
        mesh_expanded_dim = omega_values[num_mesh] * mesh_dimension[:, :, np.newaxis]
        repeated_volume = np.repeat(mesh_expanded_dim, repeats=len(phase_combinations), axis=2)
        sinusoid_images += np.sin(np.add(repeated_volume, phase_combinations[:, num_mesh]))
    sinusoid_images = np.swapaxes(sinusoid_images, 2, 0)
    return phase_combinations, sinusoid_images


def random_so3_matrices(num_samples):
    random_matrices = np.random.normal(0.0, 1.0, (num_samples, 3, 3))
    u, s, vh = np.linalg.svd(random_matrices)
    orthogonal_matrices = np.matmul(u, vh)
    so3_matrices = np.copy(orthogonal_matrices)
    angles = np.zeros((num_samples, 3, 3))
    for num_matrix, matrix in enumerate(orthogonal_matrices):
        so3_matrices[num_matrix] = np.linalg.det(matrix) * matrix
        angles[num_matrix, :, :] = rotationMatrixToEulerAngles(so3_matrices[num_matrix])
    return so3_matrices, angles


def random_o3_matrices(num_samples):
    random_matrices = np.random.normal(0.0, 1.0, (num_samples, 3, 3))
    u, s, vh = np.linalg.svd(random_matrices)
    orthogonal_matrices = np.matmul(u, vh)
    angles = np.zeros((num_samples, 3, 3))
    for num_matrix, matrix in enumerate(orthogonal_matrices):
        angles[num_matrix, :, :] = rotationMatrixToEulerAngles(np.linalg.det(matrix) * matrix)
    return orthogonal_matrices, angles


def flatten_matrix(matrices):
    flat_matrices = np.reshape(matrices, (-1, 9))
    return flat_matrices
