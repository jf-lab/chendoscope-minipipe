import numpy as np
from scipy.ndimage.measurements import center_of_mass

def get_centroids(A, shape=[324, 243]):

    centroids = np.zeros((A.shape[1], 2))

    for nn in range(A.shape[1]):
        roi = A[:, nn].reshape(shape[0], shape[1])
        centroids[nn, :] = center_of_mass(roi)

    return centroids
