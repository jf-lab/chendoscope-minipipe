import numpy as np
import argparse
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def get_args():
    parser = argparse.ArgumentParser(description='Review CNMF-E traces')
    parser.add_argument('input', help='.mat output files to review')
    return parser.parse_args()


def review_traces(trace_array):

    traces_to_keep = []
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(0, trace_array.shape[1])
    trace, = plt.plot(t, trace_array[0, :])

    class Index(object):
        ind = 0

        def keep(self, event):
            traces_to_keep.append(self.ind)
            self.ind += 1
            if self.ind >= trace_array.shape[0]:
                plt.close()
            else:
                ydata = trace_array[self.ind, :]
                trace.set_ydata(ydata)
                plt.draw()

        def exclude(self, event):
            self.ind += 1
            if self.ind >= trace_array.shape[0]:
                plt.close()
            else:
                ydata = trace_array[self.ind, :]
                trace.set_ydata(ydata)
                plt.draw()


    callback = Index()
    axkeep = plt.axes([0.81, 0.05, 0.1, 0.075])
    axexclude = plt.axes([0.7, 0.05, 0.1, 0.075])
    bkeep = Button(axkeep, 'Keep')
    bkeep.on_clicked(callback.keep)
    bexclude = Button(axexclude, 'Exclude')
    bexclude.on_clicked(callback.exclude)
    plt.show()

    return traces_to_keep

if __name__ == '__main__':
    args = get_args()
    data = sio.loadmat(args.input)
    data['keep'] = review_traces(data['C_raw'])
    sio.savemat(args.input, data)
