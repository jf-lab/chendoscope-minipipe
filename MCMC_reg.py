import numpy as np
from skimage.transform import SimilarityTransform, warp, rotate, downscale_local_mean


def mean_sq_diff(target, img):

    return ((target-img)**2)[img > 0].mean()


def corr(target, img):

    coeff = np.corrcoef(target[img > 0].ravel(), img[img > 0].ravel())

    return -coeff[0, 1]


def transform(img, tx, ty, rot):

    img_rot = rotate(img, rot, preserve_range=True)
    transf = SimilarityTransform(translation=[tx, ty])

    return warp(img_rot, transf)


def transform_diff(target, img, tx, ty, rot):

    img_transf = transform(img, tx, ty, rot)

    return corr(target, img_transf)


def propose_params(tx, ty, rot, sigma_t, sigma_r):

    tx_prop = np.random.normal(tx, sigma_t)
    ty_prop = np.random.normal(ty, sigma_t)
    rot_prop = np.random.normal(rot, sigma_r)

    return tx_prop, ty_prop, rot_prop

def get_params(target, img, sigma_t, sigma_r, max_iteration):

    [tx, ty, rot] = [0, 0, 0]
    diff = transform_diff(target, img, tx, ty, rot)
    params_record = np.zeros((max_iteration, 4))

    for it in range(max_iteration):

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
