import numpy as np
import pims
import math
from skimage import img_as_uint
from motion import align_video
import skimage.io


def process_chunks(file_path, chunk_size=1000, ds_factor=4, correct_motion = True, thresh=1.8, cutoff=0.05, target_frame=0):

    video = pims.Video(file_path)
    reference = np.round(np.mean(np.array(video[0:ds_factor])[:,:,:,0], axis=0))
    save_name = file_path.replace('.mkv', '_proc')

    for chunk in range(math.ceil(len(video)/chunk_size)):

        print('Processing chunk {} of {}'.format(chunk+1, math.ceil(len(video)/chunk_size)))

        start = chunk*chunk_size

        if start + chunk_size < len(video):
            stop = start + chunk_size

        else:
            stop = start + len(video) - chunk_size

        video_chunk = video[start:stop]
        video_chunk_ds = downsample(video_chunk, ds_factor)

        if correct_motion:
            video_chunk_reg = align_video(video_chunk_ds, reference, thresh, cutoff)
            skimage.io.imsave(save_name + '_{}.tiff'.format(chunk), img_as_uint(video_chunk_reg/2**16))

        else:
            skimage.io.imsave(save_name + '_{}.tiff'.format(chunk), img_as_uint(video_chunk_ds/2**16))


def downsample(vid, ds_factor):

    dims = vid[0].shape
    vid_ds = np.zeros((int(len(vid)/ds_factor), dims[0], dims[1]))

    frame_ds = 0
    for frame in range(0, len(vid), ds_factor):
        if frame + ds_factor <= len(vid):
            stack = np.array(vid[frame:frame+ds_factor])[:,:,:,0]
            vid_ds[frame_ds, :, :] = np.round(np.mean(stack, axis=0))
            frame_ds += 1

        else:
            continue

    return vid_ds
