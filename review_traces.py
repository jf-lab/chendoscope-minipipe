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
from matplotlib.widgets import Button


def get_args():
    parser = argparse.ArgumentParser(description='Review CNMF-E traces')
    parser.add_argument('input', help='.mat output files to review')
    return parser.parse_args()


def review_traces(data):

    traces_to_keep = []
    fig, ax = plt.subplots(1,2)
    plt.subplots_adjust(bottom=0.2)
    ds = data['spatial_ds_factor']
    ax[0].matshow(data['A'][:,0].reshape(int(648/ds), int(486/ds)))
    t = np.arange(0, data['C'].shape[1])
    trace_C, = plt.plot(t, data['C'][0, :])
    trace_Craw, = plt.plot(t, data['C_raw'][0, :])

    class Index(object):
        ind = 0

        def keep(self, event):
            traces_to_keep.append(self.ind)
            self.ind += 1
            if self.ind >= data['C'].shape[0]:
                plt.close()
            else:
                ax[0].matshow(data['A'][:,self.ind].reshape(int(648/ds), int(486/ds)))
                ydata = data['C'][self.ind, :]
                trace_C.set_ydata(ydata)
                ydata = data['C_raw'][self.ind, :]
                trace_Craw.set_ydata(ydata)
                plt.draw()

        def exclude(self, event):
            self.ind += 1
            if self.ind >= data['C'].shape[0]:
                plt.close()
            else:
                ax[0].matshow(data['A'][:,self.ind].reshape(int(648/ds), int(486/ds)))
                ydata = data['C'][self.ind, :]
                trace_C.set_ydata(ydata)
                ydata = data['C_raw'][self.ind, :]
                trace_Craw.set_ydata(ydata)
                plt.draw()


    callback = Index()
    axkeep = plt.axes([0.81, 0.05, 0.1, 0.075])
    axexclude = plt.axes([0.7, 0.05, 0.1, 0.075])
    bkeep = Button(axkeep, 'Keep')
    bkeep.on_clicked(callback.keep)
    bexclude = Button(axexclude, 'Exclude')
    bexclude.on_clicked(callback.exclude)
    plt.show()

    data['keep'] = traces_to_keep

if __name__ == '__main__':
    args = get_args()
    data = sio.loadmat(args.input)
    review_traces(data)
    sio.savemat(args.input, data)
