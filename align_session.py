import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.sparse import issparse
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import cdist
import argparse

def cat_session(data_1, data_2, frame_siz=[324, 243], proximity=2):

    if 'keep' in data_1 and 'keep' in data_2:
        data_cat = {}
        keep_1, keep_2 = data_1['keep'][0], data_2['keep'][0]
        data_cat['C'], data_cat['A'] = match_neurons(data_1['C'][keep_1, :], data_2['C'][keep_2, :], data_1['A'][:, keep_1], data_2['A'][:, keep_2], frame_siz, proximity)
        data_cat['C_raw'], _ = match_neurons(data_1['C_raw'][keep_1, :], data_2['C_raw'][keep_2, :], data_1['A'][:, keep_1], data_2['A'][:, keep_2], frame_siz, proximity)
        data_cat['S'], _ = match_neurons(data_1['S'][keep_1, :], data_2['S'][keep_2, :], data_1['A'][:, keep_1], data_2['A'][:, keep_2], frame_siz, proximity)

    else:
        data_cat = {}
        data_cat['C'], data_cat['A'] = match_neurons(data_1['C'], data_2['C'], data_1['A'], data_2['A'], frame_siz, proximity)
        data_cat['C_raw'], _ = match_neurons(data_1['C_raw'], data_2['C_raw'], data_1['A'], data_2['A'], frame_siz, proximity)
        data_cat['S'], _ = match_neurons(data_1['S'], data_2['S'], data_1['A'], data_2['A'], frame_siz, proximity)

    return data_cat

def match_neurons(traces_1, traces_2, A_1, A_2, frame_siz, proximity):

    if issparse(A_1):
        A_1 = np.array(A_1.todense())

    if issparse(A_2):
        A_2 = np.array(A_2.todense())

    traces_match = np.array([]).reshape(0, traces_1.shape[1] + traces_2.shape[1])
    A_match = np.array([]).reshape(A_1.shape[0], 0)

    A_1_centroids = get_centroids(A_1, frame_siz)
    A_2_centroids = get_centroids(A_2, frame_siz)

    distance = cdist(A_1_centroids, A_2_centroids)
    matched_1 = np.zeros(A_1.shape[1])
    matched_2 = np.zeros(A_2.shape[1]) #matrix used for marking if trace has been matched
    match_range = np.sqrt(np.count_nonzero(A_1, axis=0))/proximity

    # Concatenate matched neurons
    for nn1 in range(A_1_centroids.shape[0]):
        for nn2 in range(A_2_centroids.shape[0]):

            if distance[nn1, nn2] < match_range[nn1]:

                if matched_1[nn1] == 1 or matched_2[nn2] == 1:
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


def get_args():
    parser = argparse.ArgumentParser(description='Stitching Multiple Sessions')
    parser.add_argument('input', help='ordered file names', nargs='+')
    parser.add_argument('--output', help='output .mat file name', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    data_1 = loadmat(args.input[0])
    data_2 = loadmat(args.input[1])
    data_cat = cat_session(data_1, data_2) # TODO add more optional arguments for command line later
    data_cat['spatial_ds_factor'] = data_1['spatial_ds_factor']
    if len(args.input) > 2:
        for file in range(2, len(args.input)):
            data = loadmat(args.input[file])
            data_cat = cat_session(data_cat, data)

    data_cat['spatial_ds_factor'] = data_1['spatial_ds_factor']
    savemat(args.output + '.mat', data_cat)
