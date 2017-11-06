import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pims
from skimage.io import imsave
from skimage.transform import SimilarityTransform, warp, rotate, downscale_local_mean
import argparse


def get_params(target, img, sigma_t, sigma_r, max_iteration):

    [tx, ty, rot] = [0, 0, 0]
    diff = transform_diff(target, img, tx, ty, rot)
    params_record = np.zeros((max_iteration, 4))

    for it in tqdm(range(max_iteration), desc='Optimizing Transformation'):
        tx_p, ty_p, rot_p = propose_params(tx, ty, rot, sigma_t, sigma_r)
        diff_p = transform_diff(target, img, tx_p, ty_p, rot_p)

        if diff_p < diff:
            tx, ty, rot, diff = tx_p, ty_p, rot_p, diff_p

        elif diff_p/diff < np.random.rand():
            tx, ty, rot, diff = tx_p, ty_p, rot_p, diff_p

        params_record[it, :] = [tx_p, ty_p, rot_p, diff_p]

    params_sort = params_record[params_record[:, 3].argsort()]
    params_sort = np.mean(params_sort[:int(max_iteration/10), :], axis=0)

    tx, ty, rot = params_sort[0], params_sort[1], params_sort[2]

    img_trans = transform(img, tx, ty, rot)

    return img_trans, tx, ty, rot, params_record


def mean_sq_diff(target, img):

    return ((target-img)**2)[img > 0].mean()


def corr(target, img):

    coeff = np.corrcoef(target[img > 0].ravel(), img[img > 0].ravel())

    return -coeff[0, 1]


def transform(img, tx, ty, rot):

    img_rot = rotate(img, rot, preserve_range=True)
    transf = SimilarityTransform(translation=[tx, ty])

    return warp(img_rot, transf, preserve_range=True)


def transform_diff(target, img, tx, ty, rot):

    img_transf = transform(img, tx, ty, rot)

    return corr(target, img_transf)


def propose_params(tx, ty, rot, sigma_t, sigma_r):

    tx_prop = np.random.normal(tx, sigma_t)
    ty_prop = np.random.normal(ty, sigma_t)
    rot_prop = np.random.normal(rot, sigma_r)

    return tx_prop, ty_prop, rot_prop

def get_args():
    parser = argparse.ArgumentParser(description='Testing MCMC alignment')
    parser.add_argument('input', help='ordered file names', nargs='+')
    parser.add_argument('-i', '--iterations', type=int)
    parser.set_defaults(iterations=1000)
    parser.add_argument('-t', '--sigma_t', type=float)
    parser.set_defaults(sigma_t=0.1)
    parser.add_argument('-r', '--sigma_r', type=float)
    parser.set_defaults(sigma_r=0.1)
    return parser.parse_args()

if __name__ == '__main__':
    args=get_args()
    print(args.input)
    vid1 = pims.Video(args.input[0])
    slice1 = np.array(vid1[:100])
    img1 = np.round(np.mean(slice1, axis=0))
    vid2 = pims.Video(args.input[1])
    slice2 = np.array(vid2[:100])
    img2 = np.round(np.mean(slice2, axis=0))

    img_trans, tx, ty, rot, _ = get_params(img1, img2, args.sigma_t, args.sigma_r, args.iterations)

    fig, axarr = plt.subplots(1,3, sharex=True, sharey=True)
    axarr[0].matshow(img2)
    axarr[0].set_title('Image')
    axarr[1].matshow(img_trans)
    axarr[1].set_title('Transformed')
    axarr[2].matshow(img1)
    axarr[2].set_title('Target')
    plt.show()
    yesno = input('Apply transformation? [y/n]')

    # TODO add chunking for large videos
    if yesno == 'y':
        vid2_transf = np.zeros((len(vid2), img2.shape[0], img2.shape[1]))
        for frame in tqdm(range(len(vid2)), desc='Applying transformation'):
            img = vid2[frame]
            img_transf = transform(img, tx, ty, rot)
            vid2_transf[frame, :, :] = img_transf

        fname = args.input[1].replace('.tiff', '_transf.tiff')
        imsave(fname, vid2_transf)
