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


def process_chunk(filename, start, stop, reference, save_name, xlims = None, ylims = None, fps= 20, ds_factor=4, correct_motion=True, thresh=1.8, cutoff=0.05, clean_pixels=False, pixel_thresh=1.1, format='tiff'):
    '''
    Process one chunk of a video read in from pims and save as .tiff

    Input:
        - filename: video path
        - start: start frame
        - stop: stop frame
        - reference: reference frame
        - xlims: tuple of 2 ints, crop limits on x-axis
        - ylims: tuple of 2 ints, crop limits on y-axis
        - fps: int, output frames per second
        - ds_factor: int, downsample factor, default=4
        - correct_motion: bool, correct motion, default=True
        - thresh: flt, threshold for motion correction, default=1.0
        - cutoff: flt, cutoff for motion correction, default=0.05
        - format: string, 'tiff' or 'avi'
    Output:
        - None, saves processed chunk as .tiff or .avi
    '''
    chunk = stop/(stop-start)
    video = pims.ImageIOReader(filename)
    frame_rate = fps  # video.frame_rate
    video_chunk = video[start:stop]
    print("Processing frames {} to {} of {}".format(start, stop, len(video)))

    video_chunk_ds = downsample(video_chunk, ds_factor, xlims, ylims)
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

def downsample(vid, ds_factor, xlims=None, ylims=None):
    '''
    Downsample video by ds_factor.
    
    If xlims and ylims are not None, crop video to these limits also

    Input:
        - vid: numpy array, video
        - ds_factor: int, downsample factor
        - xlims (optional): tuple of ints, x-index of crop limits
        - ylims (optional): tuple of ints: y-index of crop limits
        
    Output:
        - vid_ds: numpy array, downsampled video
    '''
    dims = vid[0].shape
    
    if xlims is not None:
        xs, xe = xlims
    else:
        xs = 0
        xe = dims[1] - 1
    
    if ylims is not None:
        ys, ye = ylims
    else:
        ys = 0
        ye = dims[0] - 1
        
    
    dims = vid[0].shape
    vid_ds = np.zeros((int(len(vid)/ds_factor), ye-ys, xe-xs))

    frame_ds = 0
    for frame in tqdm(range(0, len(vid), ds_factor), desc='Downsampling'):
        if frame + ds_factor <= len(vid):
            stack = np.array(vid[frame:frame+ds_factor])[:,ys:ye,xs:xe,0]
            vid_ds[frame_ds, :, :] = np.round(np.mean(stack, axis=0))
            frame_ds += 1

        else:
            continue

    return vid_ds

def get_crop_lims(vid, crop_thresh=40):
    '''
    Find x,y limits where the mean fluorescence is always above a defined threshold value
    
    Input:
        - vid: numpy array, video
        - crop_thresh: int, fluorescence threshold to find x,y limits to crop to
    Output:
        - xlims: tuple of 2 ints, x-axis pixels to crop to
        - ylims: tuple of 2 ints, y-axis pixels to crop to
    '''
    dims = vid[0].shape
    xs = np.inf
    xe = 0
    ys = np.inf
    ye = 0

    y = np.arange(dims[0])
    x = np.arange(dims[1])
    
    for frame in vid:
        frame = np.array(frame)[:,:,0]

        xf = frame.mean(axis=0)
        yf = frame.mean(axis=1)

        x_thresh = x[xf>=crop_thresh]
        y_thresh = y[yf>=crop_thresh]

        if x_thresh[0] < xs:
            xs = x_thresh[0]

        if x_thresh[-1] > xe:
            xe = x_thresh[-1]

        if y_thresh[0] < ys:
            ys = y_thresh[0]

        if y_thresh[-1] > ye:
            ye = y_thresh[-1]
            
        return (xs, xe), (ys, ye)


def remove_dead_pixels(vid, thresh=1.1):
    for frame in tqdm(range(vid.shape[0]), desc='Removing Dead Pixels'):
        med = skimage.filters.median(vid[frame, :, :], square(10)).ravel()
        img = vid[frame, :, :].ravel()
        img[img>thresh*med] = med[img>thresh*med]
        vid[frame, :, :] = img.reshape(vid.shape[1], vid.shape[2])

def save_to_avi(vid, fps, filename):
    
    total_frames, height, width = vid.shape
    container = av.open(filename, 'w')
    stream = container.add_stream('rawvideo', rate=fps)
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
