'''
Authors: Andrew Mocle and Lina Tran
Date: July 10, 2017

Pre-cnmf-e processing of videos in chunks:
- Downsampling
- Motion Correction

'''
import pims
import av
import numpy as np
import math
from tqdm import tqdm
from skimage import img_as_uint
from motion import align_video
import skimage.io
import skimage.filters
from skimage.morphology import square


def process_chunk(filename, start, stop, reference, save_name, ds_factor=4, correct_motion=True, thresh=1.8, cutoff=0.05, clean_pixels=False, pixel_thresh=1.1, format='tiff'):
    '''
    Process one chunk of a video read in from pims and save as .tiff

    Input:
        - filename: video path
        - start: start frame
        - stop: stop frame
        - reference: reference frame
        - ds_factor: int, downsample factor, default=4
        - correct_motion: bool, correct motion, default=True
        - thresh: flt, threshold for motion correction, default=1.0
        - cutoff: flt, cutoff for motion correction, default=0.05
    Output:
        - None, saves processed chunk as .tiff or .avi
    '''
    chunk = stop/(stop-start)
    video = pims.ImageIOReader(filename)
    frame_rate = video.frame_rate
    video_chunk = video[start:stop]
    print("Processing frames {} to {} of {}".format(start, stop, len(video)))

    video_chunk_ds = downsample(video_chunk, ds_factor)
    #in order to have 01, 02 for file sorting and concatenation of chunks
    if chunk < 10:
        chunk = '0' + str(chunk)

    if clean_pixels:
        remove_dead_pixels(video_chunk_ds, pixel_thresh)

    if correct_motion:
        video_chunk_ds = align_video(video_chunk_ds, reference, thresh, cutoff)
    
    if format == 'tiff':
        skimage.io.imsave(save_name + '_temp_{}.tiff'.format(chunk), img_as_uint(video_chunk_ds/2**16))
    elif format == 'avi':
        save_to_avi(video_chunk_ds, fps = frame_rate / ds_factor, filename = save_name + '_temp_{}.avi'.format(chunk))


def downsample(vid, ds_factor):
    '''
    Downsample video by ds_factor.

    Input:
        - vid: numpy array, video
        - ds_factor: int, downsample factor
    Output:
        - vid_ds: numpy array, downsampled video
    '''
    dims = vid[0].shape
    vid_ds = np.zeros((int(len(vid)/ds_factor), dims[0], dims[1]))

    frame_ds = 0
    for frame in tqdm(range(0, len(vid), ds_factor), desc='Downsampling'):
        if frame + ds_factor < len(vid):
            stack = np.array(vid[frame:frame+ds_factor])[:,:,:,0]
            vid_ds[frame_ds, :, :] = np.round(np.mean(stack, axis=0))
            frame_ds += 1

        else:
            continue

    return vid_ds


def remove_dead_pixels(vid, thresh=1.1):
    for frame in tqdm(range(vid.shape[0]), desc='Removing Dead Pixels'):
        med = skimage.filters.median(vid[frame, :, :], square(10)).ravel()
        img = vid[frame, :, :].ravel()
        img[img>thresh*med] = med[img>thresh*med]
        vid[frame, :, :] = img.reshape(vid.shape[1], vid.shape[2])

def save_to_avi(vid, fps, filename):
    
    total_frames, height, width = vid.shape
    container = av.open(filename, 'w')
    stream = container.add_stream('rawvideo', rate=fps) # example vid from miniscopy uses palettize codec
    stream.height = height
    stream.width = width
    stream.pix_fmt = 'bgr24'
    
    for frame in vid:
        # Convert frame to RGB uint8 values
        frame = frame.astype('uint8')
        frame = np.repeat(np.reshape(frame, newshape=(frame.shape[0], frame.shape[1], 1)), repeats=3, axis=2)
        
        # Encode frame into stream
        frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        for packet in stream.encode(frame):
            container.mux(packet)
    
    # Flush Stream
    for packet in stream.encode():
        container.mux(packet)

    # Close file
    container.close()
