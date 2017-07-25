'''
Author: Lina Tran
Date: July 14, 2017

Pipeline using motion correction and pre_cnmfe to motion correct, downsample,
and turn files into .tiffs

Requirements:

tiffcp
    install tiffcp using:
    $ apt-get install libtiff-tools

Command Line Usage:

$ python minipipe.py file1.mkv file2.mkv file3.mkv -d 4 -c 5000 --correct_motion
    -t 1.8 --target_frame 0 --cores 2

-d: downsample factor
-c chunk_size
--correct_motion if you want to motion correct_motion
-t if you want to indicate threshold
-target_frame if you want to choose a frame other than the first to reference
--cores number of threads to run in parallel
'''


from pre_cnmfe import process_chunk
import argparse
import pims
import numpy as np
from os import system, path
from joblib import Parallel, delayed


def get_args():
    parser = argparse.ArgumentParser(description='Convert and downsample .mkv files to .tiff')
    parser.add_argument('input', help='files', nargs='+')
    parser.add_argument('-d', '--downsample', help='downsample factor, default is 4', type=int, default=4)
    parser.add_argument('-c', '--chunk_size', help='chunk_size of frames, default is 2000', type=int, default=2000)
    parser.add_argument('--motion_corr', dest='correct_motion', help='motion correct the given video', action='store_true')
    parser.add_argument('--no_motion_corr', dest='correct_motion', help='motion correct the given video', action='store_false')
    parser.set_defaults(correct_motion=True)
    parser.add_argument('-t', '--threshold', help='threshold for moco, default is 1.0', type=float, default=1.0)
    parser.add_argument('--target_frame', help='target frame to reference, default is 0', type=int, default=0)
    parser.add_argument('--bigtiff', dest='bigtiff', help='use bigtiff file format for large (>4Gb) .tiffs', action='store_true')
    parser.set_defaults(bigtiff=False)
    parser.add_argument('--cores', help='cores to use, default is 1', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    for filename in args.input:
        print("Processing {}".format(filename))
        directory = path.dirname(filename)
        vid = pims.Video(filename)
        reference = np.round(np.mean(np.array(vid[args.target_frame:args.downsample])[:,:,:,0], axis=0))
        save_name = filename.replace('.mkv', '_proc')
<<<<<<< HEAD
        process_chunks(filename, args.chunk_size, args.downsample, args.correct_motion, args.threshold, 0.05, args.target_frame)
        if args.bigtiff:
            system("tiffcp -8 {}/*_temp_* {}.tiff".format(directory, save_name))
        else:
            system("tiffcp {}/*_temp_* {}.tiff".format(directory, save_name))
        system("rm {}/*_temp_*".format(directory))
=======
>>>>>>> fffe865933ae188d334bbc413a5a65a6212ffd15

        starts = np.arange(0,len(vid),args.chunk_size)
        stops = starts+args.chunk_size
        frames = list(zip(starts, stops))

        Parallel(n_jobs=args.cores)(delayed(process_chunk)(filename=filename, start=start, stop=stop, reference=reference, save_name=save_name, ds_factor=args.downsample, correct_motion=args.correct_motion, thresh=args.threshold) for start, stop in frames)
        system("tiffcp {}/*_temp_* {}.tiff".format(directory, save_name))
        system("rm {}/*_temp_*".format(directory))
