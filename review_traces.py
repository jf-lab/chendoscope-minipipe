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
    return parser.parse_args()


def review_traces(data):

    traces_to_keep = []
    ds = data['ssub']

    fig = plt.figure()
    grid = gridspec.GridSpec(4,6)


    # spatial footprint for each neuron
    ax = fig.add_subplot(grid[:, :2])
    ax.matshow(data['A'][:,0].reshape(int(648/ds), int(486/ds)))

    # raw and fitted traces
    t = np.arange(0, data['C'].shape[1])
    ax1 = fig.add_subplot(grid[:2, 2:] )
    trace_Craw, = ax1.plot(t, data['C_raw'][0, :])
    ax2 = fig.add_subplot(grid[2:,2:], sharex=ax1)
    trace_C, = ax2.plot(t, data['C'][0, :])

    plt.setp(ax1, title='Raw Trace')
    plt.setp(ax2, title='Fitted Trace')
    grid.update(top=0.4, left=0.5, hspace=0.4, wspace=0.2)
    grid.tight_layout(fig)
    fig.suptitle("Cell 0 of {}".format(data['C'].shape[0]), x=0.1)

    class Index(object):
        ind = 0

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


    callback = Index()

    # press keep or exclude buttons for each trace
    axkeep = plt.axes([0.05, 0.05, 0.1, 0.075])
    axexclude = plt.axes([0.17, 0.05, 0.1, 0.075])
    bkeep = Button(axkeep, 'Keep')
    bkeep.on_clicked(callback.keep)
    bexclude = Button(axexclude, 'Exclude')
    bexclude.on_clicked(callback.exclude)

    # alternatively, press k to keep, j to exclude
    def key_press(event):
        if event.key == 'k':
            callback.keep(event)
        if event.key == 'j':
            callback.exclude(event)

    plt.gcf().canvas.mpl_connect('key_press_event', key_press)

    plt.show()

    data['keep'] = traces_to_keep

if __name__ == '__main__':
    args = get_args()
    data = sio.loadmat(args.input)

    if issparse(data['A']):
        data['A'] = np.array(data['A'].todense())

    review_traces(data)
    sio.savemat(args.input, data)
