import numpy as np
import argparse
from scipy.io import loadmat, savemat
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Removing unkept neurons prior to registration')
    parser.add_argument('input', help='files', nargs='+')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    for fname in tqdm(args.input):
        sname = fname.replace('.mat', '_keep.mat')
        data = loadmat(fname)
        keep = data['keep'].ravel()
        data['A'] = data['A'][:, keep]
        data['C_raw'] = data['C_raw'][keep, :]
        data['C'] = data['C'][keep, :]
        data['S'] = data['S'][keep, :]

        savemat(sname, data, do_compression=True)
