import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import cdist

def match_neurons(data_1, data_2, match=None, match_range=2, frame_siz=[324, 243]):

    A_1 = data_1['A']
    C_1 = data_1['C']
    A_2 = data_2['A']
    C_2 = data_2['C']

    if not match:
        match = np.zeros((A_1.shape[1], 2))

    else:
        match = np.concatenate((match, np.zeros((match.shape[0], 1))), axis=1)

    A_1_centroids = get_centroids(A_1, frame_siz)
    A_2_centroids = get_centroids(A_2, frame_siz)

    for nn in range(A_1_centroids.shape[0]):
        distance = cdist(A_1_centroids[nn,:], A_2_centroids)
        candidates = distance < match_range



def get_centroids(A, frame_siz):

    centroids = np.zeros((A.shape[1], 2))

    for nn in range(A.shape[1]):
        roi = A[:, nn].reshape(frame_siz[0], frame_siz[1])
        centroids[nn, :] = center_of_mass(roi)

    return centroids
