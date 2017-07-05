import numpy as np
import pims
from tqdm import tqdm
import skimage
import skimage.io

def mkv2tiff(file_path, save_name, ds_factor=4):

    print('Accessing Video')
    vid = pims.Video(file_path)

    dims = vid[0].shape
    vid_ds = np.zeros((int(len(vid)/ds_factor), dims[0], dims[1]))

    frame_ds = 0
    for frame in tqdm(range(0, len(vid), ds_factor), desc='Downsampling'):
        if frame + ds_factor < len(vid):
            slice = np.array(vid[frame:frame+ds_factor])[:,:,:,0]
            vid_ds[frame_ds, :, :] = np.round(np.mean(slice, axis=0))
            frame_ds += 1

        else:
            continue

    skimage.io.imsave(save_name + '.tiff', skimage.img_as_uint(vid_ds/vid_ds.max()))
