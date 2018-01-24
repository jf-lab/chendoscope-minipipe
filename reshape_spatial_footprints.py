import numpy as np
import argparse
from scipy.io import loadmat, savemat


def get_args():
    parser = argparse.ArgumentParser(description='Reshaping spatial footprints fir registration')
    parser.add_argument('input', help='files', nargs='+')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    for filename in args.input:
        data = loadmat(filename)

        footprints = data['A']
        ds = data['ssub']
        dims = [int(648/ds), int(486/ds)]
        footprints_reshape = np.zeros((footprints.shape[1], dims[0], dims[1]))
        for ss in range(footprints.shape[1]):
            footprints_reshape[ss, :, :] = footprints[:, ss].reshape(int(648/ds), int(486/ds))
    A= {}
    A['A'] = footprints_reshape
    filename_new = filename.replace('.mat', '_spatial.mat')
    savemat(filename_new, A, do_compression=True)
