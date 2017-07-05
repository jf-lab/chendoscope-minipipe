'''
Author: Andrew Mocle
Date: July 2017

File to convert and downsample .mkv files to .tiff

Command Line Usage:

cd /cygdrive/f/jflab-minipipe/
python mkv2tiff.py /directory/of/videos/*.mkv -d 4
'''

import numpy as np
import pims
from tqdm import tqdm
import skimage
import skimage.io
import argparse as arg

def mkv2tiff(file_path, ds_factor=4):
    '''
    Input:
        - file_path: path to mkv file
        - ds_factor: factor to downsample by, default = 4, choose 1 for no
        downsampling
    Output:
        - NA, saves new downsampled and converted file to same path as original
        with '_ds' appended to end of filename.
    '''
    print('Accessing Video')
    vid = pims.Video(file_path)
    save_name = file_path.replace('.mkv', '_ds')

    dims = vid[0].shape
    vid_ds = np.zeros((int(len(vid)/ds_factor), dims[0], dims[1]))

    frame_ds = 0
    for frame in range(0, len(vid), ds_factor):
        if frame + ds_factor < len(vid):
            slice = np.array(vid[frame:frame+ds_factor])[:,:,:,0]
            vid_ds[frame_ds, :, :] = np.round(np.mean(slice, axis=0))
            frame_ds += 1
            if frame % 500 == 0:
                print('frame {} of {}'.format(frame, len(vid)))
        else:
            continue

    skimage.io.imsave(save_name + '.tiff', skimage.img_as_uint(vid_ds/vid_ds.max()))

def main():
    parser = arg.ArgumentParser(description='Convert and downsample .mkv files to .tiff')
    parser.add_argument('input', help='files', nargs='+')
    parser.add_argument('-d', '--downsample', help='downsample factor, default is 4', type=int, default=4)
    args = parser.parse_args()
    files = args.input
    for f in files:
        mkv2tiff(f, args.downsample)

if __name__ == '__main__':
    main()
