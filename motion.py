'''
Author: Andrew Mocle
Date: July 13, 2017

Motion correction within a single video file.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import scale
from skimage.transform import SimilarityTransform, warp


def align_video(video, reference, thresh=1.8, cutoff=0.05):
    '''
    Motion correct video to target_frame of video.
    Input:
        - video: numpy array of video dim(frame, x, y)
        - threshold: float, for selecting background to fit, default=1.8
        - cutoff: float, spatial frequency of cells to remove, default=0.05
        - reference: numpy array of dim(x,y) of reference image
    Output:
        numpy array of registered video
    '''
    video_spatial = spatial_lp_filter(video, 3, cutoff)
    reference_spatial = reference_lp_filter(reference, 3, cutoff)

    video_reg = np.zeros_like(video)

    for frame in range(video.shape[0]):
        tx, ty = align_frame(reference_spatial, video_spatial[frame, :, :], thresh)
        video_reg[frame, :, :] = translate(video[frame, :, :], -tx, -ty)

    return video_reg


def align_frame(target, img, thresh):
    '''
    Align img frame to target frame.
    Input:
        - target: int, reference frame to register img frame to
        - img: int, frame to register to target
        - thresh: float, threshold for selecting background to filtfilt
    Output:
        - tx: float, translated correction value along x-axis
        - ty: float, translated correction value along y-axis
    '''
    target_x = scale(np.mean(target, axis=0))
    target_y = scale(np.mean(target, axis=1))
    img_x = scale(np.mean(img, axis=0))
    img_y = scale(np.mean(img, axis=1))

    tx = np.mean((np.where(target_x > thresh))) - np.mean((np.where(img_x > thresh)))
    ty = np.mean((np.where(target_y > thresh))) - np.mean((np.where(img_y > thresh)))

    return tx, ty


def translate(img, tx, ty):
    '''
    Transform an img frame based on tx and ty translation values.
    Input:
        - img: numpy array, frame to transform
        - tx: float, translated correction value along x-axis
        - ty: float, translated correction value along y-axis
    Output:
        - numpy array of transformed frame
    '''
    transf = SimilarityTransform(translation=[tx, ty])

    return warp(img, transf, preserve_range=True)


def spatial_lp_filter(video, order, Wn):
    '''
    Filter the active neurons from image leaving only background.
    Input:
        - video: numpy array, dim(frame, x, y)
        - order: int, how sharp cutoff is for low-pass Filter
        - Wn: float between 0 and 1, angular frequency
    Output:
        - video_filtered: numpy array, dim(frame, x, y) of filtered video
    '''
    b, a = signal.butter(order, Wn, btype='low')
    video_flt = signal.filtfilt(b, a, video, axis=1)
    video_flt = signal.filtfilt(b, a, video_flt, axis=2)
    video_filtered = video_flt/np.amax(np.abs(video_flt))

    return video_filtered


def reference_lp_filter(video, order, Wn):

    b, a = signal.butter(order, Wn, btype='low')
    video_flt = signal.filtfilt(b, a, video, axis=0)
    video_flt = signal.filtfilt(b, a, video_flt, axis=1)
    video_filtered = video_flt/np.amax(np.abs(video_flt))

    return video_filtered
