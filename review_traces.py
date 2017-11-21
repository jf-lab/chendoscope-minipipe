'''
Author: Andrew Mocle
Date July 27, 2017

Review CNMF-E traces to keep or not.

Command Line Usage:
python review_traces.py traces.mat

Go through each of the traces and choose keep or exclude.
It will save to the .mat file a new dictionary entry for the kept traces.
'''

import numpy as np
import argparse
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from scipy.sparse import issparse


def get_args():
    parser = argparse.ArgumentParser(description='Review CNMF-E traces')
    parser.add_argument('input', help='.mat output files to review')
    parser.add_argument('-f', '--fs', help='Sampling rate for CNMFE input')
    parser.set_defaults(fs=5)
    parser.add_argument('-c', '--checkpoint', help='Whether to pick up from checkpoint', action='store_true')
    return parser.parse_args()


def review_traces(data, fs, checkpoint_ind, keep):

    ind_c = checkpoint_ind
    traces_to_keep = keep


    check_ind = 0
    ds = data['ssub']

    fig = plt.figure()
    grid = gridspec.GridSpec(4,6)


    # spatial footprint for each neuron
    ax = fig.add_subplot(grid[:, :2])
    ax.matshow(data['A'][:,0].reshape(int(648/ds), int(486/ds)))

    # raw and fitted traces
    t = np.arange(0, data['C'].shape[1])/fs[0,:]
    ax1 = fig.add_subplot(grid[:2, 2:] )
    trace_Craw, = ax1.plot(t, data['C_raw'][ind_c, :])
    ax2 = fig.add_subplot(grid[2:,2:], sharex=ax1)
    trace_C, = ax2.plot(t, data['C'][ind_c, :])

    plt.setp(ax1, title='Raw Trace')
    plt.setp(ax2, title='Fitted Trace')
    grid.update(top=0.4, left=0.5, hspace=0.4, wspace=0.2)
    grid.tight_layout(fig)
    fig.suptitle("Cell {} of {}".format(ind_c, data['C'].shape[0]), x=0.1)

    class Index(object):
        ind = ind_c

        def keep(self, event):
            traces_to_keep.append(self.ind)
            self.ind += 1
            if self.ind >= data['C'].shape[0]:
                plt.close()
            else:
                ax.matshow(data['A'][:,self.ind].reshape(int(648/ds), int(486/ds)))
                ydata = data['C'][self.ind, :]
                trace_C.set_ydata(ydata)
                ydata = data['C_raw'][self.ind, :]
                trace_Craw.set_ydata(ydata)
                fig.suptitle("Cell {} of {}".format(self.ind, data['C'].shape[0]), x=0.1)
                plt.draw()


        def exclude(self, event):
            self.ind += 1
            if self.ind >= data['C'].shape[0]:
                plt.close()
            else:
                ax.matshow(data['A'][:,self.ind].reshape(int(648/ds), int(486/ds)))
                ydata = data['C'][self.ind, :]
                trace_C.set_ydata(ydata)
                ydata = data['C_raw'][self.ind, :]
                trace_Craw.set_ydata(ydata)
                fig.suptitle("Cell {} of {}".format(self.ind, data['C'].shape[0]), x=0.1)
                plt.draw()

        def undo(self, event):
            self.ind -= 1
            del traces_to_keep[-1]
            ax.matshow(data['A'][:,self.ind].reshape(int(648/ds), int(486/ds)))
            ydata = data['C'][self.ind, :]
            trace_C.set_ydata(ydata)
            ydata = data['C_raw'][self.ind, :]
            trace_Craw.set_ydata(ydata)
            fig.suptitle("Cell {} of {}".format(self.ind, data['C'].shape[0]), x=0.1)
            plt.draw()

        def check(self, event):
            check_ind = self.ind # TODO How come checkpoint_ind only saves as 0?
            data['checkpoint_ind'] = check_ind
            print(check_ind)
            plt.close()


    callback = Index()

    # press keep or exclude buttons for each trace
    axkeep = plt.axes([0.05, 0.05, 0.1, 0.075])
    axexclude = plt.axes([0.17, 0.05, 0.1, 0.075])
    axundo = plt.axes([0.29, 0.05, 0.1, 0.075])
    axcheck = plt.axes([0.41, 0.05, 0.1, 0.075])

    bkeep = Button(axkeep, 'Keep')
    bkeep.on_clicked(callback.keep)

    bexclude = Button(axexclude, 'Exclude')
    bexclude.on_clicked(callback.exclude)

    bundo = Button(axundo, 'Undo')
    bundo.on_clicked(callback.undo)

    bcheck = Button(axcheck, 'Checkpoint')
    bcheck.on_clicked(callback.check)


    # alternatively, press k to keep, j to exclude
    def key_press(event):
        if event.key == 'k':
            callback.keep(event)
        if event.key == 'j':
            callback.exclude(event)
        if event.key == 'u':
            callback.undo(event)

    plt.gcf().canvas.mpl_connect('key_press_event', key_press)

    plt.show()

    data['keep'] = traces_to_keep

if __name__ == '__main__':
    args = get_args()
    data = sio.loadmat(args.input)
    fs = args.fs/data['tsub']

    if issparse(data['A']):
        data['A'] = np.array(data['A'].todense())

    print(args.checkpoint)
    if args.checkpoint:
        checkpoint_ind=data['checkpoint_ind'][0][0]
        keep = data['keep'][0].tolist()
    if not args.checkpoint:
        checkpoint_ind=0
        keep = []

    review_traces(data, fs, checkpoint_ind, keep)
    sio.savemat(args.input, data, do_compression=True)
