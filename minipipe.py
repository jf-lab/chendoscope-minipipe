'''
Author: Lina Tran
Date: July 14, 2017

Pipeline using motion correction and pre_cnmfe to motion correct, downsample,
and turn files into .tiffs

Requirements:

mkvmerge
    install mkvmerge using:
    $ sudo apt-get install mkvtoolnix mkvtoolnix-gui

tiffcp
    install tiffcp using:
    $ sudo apt-get install libtiff-tools

avimerge
    install avimerge via transcode:
    $ sudo apt-get install transcode

dependencies for av
    $ sudo apt-get install libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libavresample-dev libavfilter-dev


Command Line Usage:

$ python minipipe.py file1.mkv file2.mkv file3.mkv -d 4 -c 5000 --correct_motion
    -t 1.8 --target_frame 0 --cores 2 --bigtiff --merge -o output.mkv

Flags:
-d/--downsample: downsample factor, default is 4 (typically 20fps -> 5fps)
-c/--chunk_size: chunk_size, default is 2000
--motion_corr: if you want to motion correct_motion, default is False
-t/--threshold: if you want to indicate threshold, default is 1.0
-target_frame: if you want to choose a frame other than the first to reference
--cores: number of threads to run in parallel, default is 4
--bigtiff: If .mkv(s) amount to > 12Gb, must use this mode or memory error will occur
--merge: merge all the files instead of individually processing them
-o/--output: If --merge, then the name for the merged .tiff file
'''


from pre_cnmfe import process_chunk, get_crop_lims
import argparse
import pims
import numpy as np
from os import system, path
from joblib import Parallel, delayed
import h5py as hd
from glob import glob


def get_args():
    parser = argparse.ArgumentParser(description='Convert and downsample .mkv files to .tiff')
    parser.add_argument('input', help='files', nargs='+')
    parser.add_argument('--fps', dest = 'fps', help='Frames per second', default=20)
    parser.add_argument('-d', '--downsample', help='downsample factor, default is 4', type=int, default=4)
    parser.add_argument('-c', '--chunk_size', help='chunk_size of frames, default is 2000', type=int, default=2000)
    parser.add_argument('--motion_corr', dest='correct_motion', help='motion correct the given video', action='store_true')
    parser.add_argument('--no_motion_corr', dest='correct_motion', help='motion correct the given video', action='store_false')
    parser.set_defaults(correct_motion=False)
    parser.add_argument('-t', '--threshold', help='threshold for moco, default is 1.0', type=float, default=1.0)
    parser.add_argument('--target_frame', help='target frame to reference, default is 0', type=int, default=0)
    parser.add_argument('--bigtiff', dest='bigtiff', help='use bigtiff file format for large (>4Gb) .tiffs', action='store_true')
    parser.set_defaults(bigtiff=True)
    parser.add_argument('--merge', dest='merge', help='merge input files instead of serially processing', action='store_true')
    parser.set_defaults(merge=False)
    parser.add_argument('-o', '--output', help='if --merge, name of merged file', type=str)
    parser.add_argument('-f', '--format', help='format of output file : tiff, avi or hdf5', type=str, default='tiff')
    parser.add_argument('--cores', help='cores to use, default is 1', type=int, default=4)
    parser.add_argument('--crop', dest='crop', action='store_true')
    parser.add_argument('-ct', '--crop_thresh', dest='crop_thresh', type=int, default=40)
    parser.set_defaults(crop=False)
    parser.add_argument('--remove_dead_pixels', action='store_true')
    parser.set_defaults(remove_dead_pixels=False)
    parser.add_argument('-p', '--pixel_thresh', type=float, default=1.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.merge:
        assert args.output != None, "Please provide an output filename if using --merge"
        directory = path.dirname(args.input[0])
        args.output = directory + '/' + args.output
        files = ' + '.join(args.input)

        print(files)
        system("mkvmerge -o {} {}".format(args.output, files))
        filename = args.output

        print("Processing {}".format(filename))
        directory = path.dirname(filename)
        vid = pims.ImageIOReader(filename)
        reference = np.round(np.mean(np.array(vid[args.target_frame:args.downsample])[:,:,:,0], axis=0))
        save_name = filename.replace('.mkv', '_proc')

        if len(vid) % args.chunk_size < 10:
            args.chunk_size += 5

        starts = np.arange(0,len(vid),args.chunk_size)
        stops = starts+args.chunk_size
        frames = list(zip(starts, stops))

        Parallel(n_jobs=args.cores)(delayed(process_chunk)(filename=filename, start=start, stop=stop, reference=reference, save_name=save_name, format=args.format, ds_factor=args.downsample, correct_motion=args.correct_motion, thresh=args.threshold) for start, stop in frames)
        if args.format == 'tiff':
            if args.bigtiff:
                system("tiffcp -8 {}/*_temp_* {}.tiff".format(directory, save_name))
            else:
                system("tiffcp {}/*_temp_* {}.tiff".format(directory, save_name))
            system("rm {}/*_temp_*".format(directory))

        elif args.format == 'avi':
            system("avimerge -o {}.avi -i {}/*_temp_*.avi".format(save_name, directory))
            system("rm {}/*_temp_*".format(directory))

    else:
        for filename in args.input:
            print("Processing {}".format(filename))
            directory = path.dirname(filename)
            vid = pims.ImageIOReader(filename)
            reference = np.round(np.mean(np.array(vid[args.target_frame:args.downsample])[:,:,:,0], axis=0))
            save_name = filename.replace('.mkv', '_proc')

            if args.crop:
                xlims, ylims = get_crop_lims(vid, crop_thresh=args.crop_thresh)
            else:
                xlims = None
                ylims = None

            if len(vid) % args.chunk_size < 10:
                args.chunk_size += 5

            starts = np.arange(0,len(vid),args.chunk_size)
            stops = starts+args.chunk_size
            frames = list(zip(starts, stops))
            print(args.cores)

            Parallel(n_jobs=args.cores)(delayed(process_chunk)(filename=filename, start=start, stop=stop, xlims=xlims, ylims=ylims, fps=args.fps, reference=reference, save_name=save_name, format=args.format, ds_factor=args.downsample, correct_motion=args.correct_motion, thresh=args.threshold, clean_pixels=args.remove_dead_pixels, pixel_thresh=args.pixel_thresh) for start, stop in frames)

            if args.format == 'tiff':
                if args.bigtiff:
                    system("tiffcp -8 {}/*_temp_*.tiff {}.tiff".format(directory, save_name))
                else:
                    system("tiffcp {}/*_temp_*.tiff {}.tiff".format(directory, save_name))

            elif args.format == 'avi':
                system("avimerge -o {}.avi -i {}/*_temp_*.avi".format(save_name, directory))

            elif args.format == 'hdf5':
                if args.crop:
                    tdim = len(vid)/args.downsample
                    xdim = xlims[1]- xlims[0]
                    ydim = ylims[1]- ylims[0]
                else:
                    tdim, xdim, ydim = len(vid)/args.downsample, vid[0].shape[0], vid[0].shape[1]
                files = glob(directory + "/*_temp_*")
                filename_new = save_name + '.hdf5'
                if path.exists(filename_new):
                    system('rm %s'%filename_new)
                full_mov = hd.File(filename_new)

                movie = full_mov.create_dataset('original', shape=(tdim, xdim*ydim), chunks=True)
                full_mov.attrs['folder'] = directory
                full_mov.attrs['filename'] = path.basename(save_name)

                full_mov['original'].attrs['duration'] = int(tdim)
                full_mov['original'].attrs['dims'] = (ydim, xdim) # 2D np.arrays are (row X cols) --> (ydim X xdim) 

                # account for new vid sizes given downsampling
                start = 0
                for ix, f in enumerate(files):
                    chunk = hd.File(f)
                    chunk_mov = chunk['original']
                    stop = start + len(chunk_mov)

                    full_mov['original'][start:stop] = chunk_mov[:]


                    #remove from memory, read from file again
                    full_mov.flush()
                    chunk.close()
                    start = stop


                full_mov.close()

            system("rm {}/*_temp_*".format(directory))
