import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import cdist

def stitch_sessions():

    return []

def match_neurons(traces_1, traces_2, A_1, A_2, frame_siz=[324, 243], proximity=2):

    traces_match = np.array([]).reshape(0, traces_1.shape[1] + traces_2.shape[1])
    A_match = np.array([]).reshape(A_1.shape[0], 0)

    A_1_centroids = get_centroids(A_1, frame_siz)
    A_2_centroids = get_centroids(A_2, frame_siz)

    distance = cdist(A_1_centroids, A_2_centroids)
    matched_1 = np.zeros(A_1.shape[1])
    matched_2 = np.zeros(A_2.shape[1]) #matrix used for marking if trace has been matched
    match_range = np.sqrt(np.count_nonzero(A_1, axis=0))/proximity
    print(match_range.shape)

    # Concatenate matched neurons
    for nn1 in range(A_1_centroids.shape[0]):
        for nn2 in range(A_2_centroids.shape[0]):

            if distance[nn1, nn2] < match_range[nn1]:

                if matched_2[nn2] == 1:
                    continue

                trace_cat = np.concatenate((traces_1[nn1, :][None, :], traces_2[nn2, :][None, :]), axis=1)
                traces_match = np.concatenate((traces_match, trace_cat), axis=0)
                A_match = np.concatenate((A_match, A_2[:, nn2][:, None]), axis=1)
                matched_1[nn1] = 1
                matched_2[nn2] = 1

    # Add unmatched neurons from array 1 to bottom of matched arrays
    for nn1 in range(A_1_centroids.shape[0]):
        if matched_1[nn1] == 0:
            trace_cat = np.concatenate((traces_1[nn1, :][None, :], np.zeros((1, traces_2.shape[1]))), axis=1)
            traces_match = np.concatenate((traces_match, trace_cat), axis=0)
            A_match = np.concatenate((A_match, A_1[:, nn1][:, None]), axis=1)

    # Add unmatched neurons from array 2 to bottom of matched arrays
    for nn2 in range(A_2_centroids.shape[0]):
        if matched_2[nn2] == 0:
            trace_cat = np.concatenate(((np.zeros((1, traces_1.shape[1])), traces_2[nn2, :][None, :])), axis=1)
            traces_match = np.concatenate((traces_match, trace_cat), axis=0)
            A_match = np.concatenate((A_match, A_2[:, nn2][:, None]), axis=1)

    return traces_match, A_match


def get_centroids(A, frame_siz=[324, 243]):

    centroids = np.zeros((A.shape[1], 2))

    for nn in range(A.shape[1]):
        roi = A[:, nn].reshape(frame_siz[0], frame_siz[1])
        centroids[nn, :] = center_of_mass(roi)

    return centroids
