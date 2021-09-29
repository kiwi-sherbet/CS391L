#! /usr/bin/python3

import os
import pickle
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from urllib.request import urlretrieve

PATH_SCRIPT = os.path.dirname(os.path.realpath(__file__))
PATH_ROOT   = os.path.dirname(PATH_SCRIPT)
PATH_DATA   = os.path.join(PATH_ROOT, 'data')
PATH_SAVE   = os.path.join(PATH_ROOT, 'save')
PATH_PLOT   = os.path.join(PATH_ROOT, 'plot')

os.makedirs(PATH_DATA, exist_ok=True)
os.makedirs(PATH_SAVE, exist_ok=True)
os.makedirs(PATH_PLOT, exist_ok=True)

SEED_IDX=0
SOUND_NUM = 3
SAMPLE_NUM = 3
ETA = 0.01
ITERATION_NUM = 5000
BATCH_SIZE=16

COLORS = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

def save_data(data):

    with open("{}/batch-{}_eta-{}_mix-{}_sample-{}.pth".format(PATH_SAVE, data["batch size"], data["eta"], data["sound number"], data["sample number"]),"wb") as f:
        pickle.dump(data, f)


def load_data(file_name):

    with open("{}/{}.pth".format(PATH_SAVE, file_name),"rb") as f:
        return pickle.load(f)


def load_sounds():

    url = 'https://www.cs.utexas.edu/~dana/MLClass/'
    files = ['sounds.mat', 'icaTest.mat']

    path = os.path.join(PATH_DATA, 'sounds')
    os.makedirs(path, exist_ok=True)

    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    sound_set = loadmat(os.path.join(path, files[0]))

    return sound_set


def compute_correlation(input_1, input_2):

    input_2_cpy = copy.copy(input_2)

    correlation = []
    indices = []

    for i in range(input_1.shape[0]):
        vals = []
        for j in range(input_2_cpy.shape[0]):
            vals.append(np.corrcoef(input_1[i], input_2_cpy[j])[0,1])

        max_idx = np.argmax(vals)
        min_idx = np.argmin(vals)

        out_idx = np.where(vals[max_idx] > - vals[min_idx], max_idx, min_idx)

        indices.append(out_idx)
        correlation.append(vals[out_idx])
        input_2_cpy[out_idx][:] = 0

    return correlation, indices


class ICA_model():

    def __init__(self, sound_set, batch_size=BATCH_SIZE, iteration_num=ITERATION_NUM, eta=ETA):

        self.sound_set = sound_set["sounds"]

        self.eta = eta
        self.batch_size = batch_size
        self.iteration_num = iteration_num

        self.seed_idx=None
        self.sound_num=None

        self.W_0 = None
        self.W_i = None
        self.U = None
        self.X = None


    def reset(self, batch_size=BATCH_SIZE, iteration_num=ITERATION_NUM, eta=ETA):

        self.eta = eta
        self.batch_size = batch_size
        self.iteration_num = iteration_num

        self.seed_idx=None
        self.sound_num=None

        self.W_0 = None
        self.W_i = None
        self.U = None
        self.X = None


    def seed(self, seed_idx=SEED_IDX, sound_num=SOUND_NUM, sample_num=SAMPLE_NUM):

        self.sound_num = sound_num
        self.sample_num = sample_num

        np.random.seed(seed_idx)
        sample_idx = np.random.choice(self.sound_set.shape[0], size=self.sound_num)
        sample_A = np.random.rand(self.sample_num, self.sound_num) - 0.3*np.random.rand(self.sample_num, self.sound_num)

        self.seed_idx = seed_idx
        self.U = self.sound_set[sample_idx,:]
        self.X = sample_A @ self.U
        self.W_0 = np.random.rand(self.sound_num, self.sample_num)
        self.W_i = copy.copy(self.W_0)


    def iterate(self):

        correlations = []

        for idx in range(self.iteration_num):
            batch_indices = np.random.choice(self.X.shape[1], size=self.batch_size)
            batch_X = self.X[:,batch_indices]
            self._gradient_descent(batch_X)

            if idx % 10 == 0:
                U_res = self.W_i @ self.X
                correlation, _ = compute_correlation(self.U, U_res)
                correlations.append(correlation)

        return correlations


    def _gradient_descent(self, batch_X):

        Y = self.W_i @ batch_X
        Z = 1 / (1 + np.exp(- Y))
        A = (np.ones(Z.shape) - 2 * Z) @ Y.T
        grad = (np.identity(self.W_i.shape[0]) + A / Y.shape[1]) @ self.W_i

        self.W_i += self.eta * grad


    def load(self, file_name):

        results = load_data(file_name)

        if results != None:

            self.eta = results["eta"]
            self.batch_size = results["batch size"]
            self.iteration_num = results["interation number"]

            self.seed_idx = results["seed"]
            self.sound_num = results["sound number"]
            self.sample_num = results["sample number"]

            self.W_0 = results["W init"]
            self.W_i = results["W iter"]
            self.U = results["U"]
            self.X = results["X"]

            print("Loaded data successfully.")

        else:

            print("Couldn't load data successfully.")


    def save(self):

        results = {}

        results["batch size"] = self.batch_size
        results["interation number"] = self.iteration_num
        results["eta"] = self.eta

        results["seed"] = self.seed_idx
        results["sound number"] = self.sound_num
        results["sample number"] = self.sample_num

        results["W init"] = self.W_0
        results["W iter"] = self.W_i
        results["U"] = self.U
        results["X"] = self.X

        save_data(results)

        print("Saved data successfully.")


    def show_original_signals(self):
        
        plot_num = self.U.shape[0]

        plt.figure()
        plt.rcParams.update({'font.size': 10})

        for idx in range(plot_num):

            plt.subplot(plot_num, 1, idx + 1)
            plt.plot(self.U[idx], color=COLORS[idx%len(COLORS)])
            # plt.ylabel(str(idx + 1)+" ", rotation=0)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel("Signal #{}".format(idx+1))

        plt.xlabel("Source signals")
        plt.savefig("{}/original.png".format(PATH_PLOT))

        return


    def show_mixed_singals(self):

        plot_num = self.X.shape[0]

        plt.figure()
        plt.rcParams.update({'font.size': 10})

        for idx in range(plot_num):

            plt.subplot(plot_num, 1, idx + 1)
            plt.plot(self.X[idx], color=COLORS[idx%len(COLORS)])
            plt.xticks([])
            plt.yticks([])
            plt.ylabel("Signal #{}".format(idx+1))
    
        plt.xlabel("Mixed signals")
        plt.savefig("{}/mixed.png".format(PATH_PLOT))


        return
        

    def show_restored_signals(self):

        U_res = self.W_i @ self.X
        _, indices = compute_correlation(self.U, U_res)

        plot_num = U_res.shape[0]

        plt.figure()
        plt.rcParams.update({'font.size': 10})

        for idx in range(plot_num):

            plt.subplot(plot_num, 1, idx + 1)
            plt.plot(U_res[indices[idx]], color=COLORS[idx%len(COLORS)])
            plt.xticks([])
            plt.yticks([])
            plt.ylabel("Signal #{}".format(idx+1))

        plt.xlabel("Restored signals")
        plt.savefig("{}/restored.png".format(PATH_PLOT))

        return


if __name__ == "__main__":

    sound_set = load_sounds()
    model = ICA_model(sound_set)

    batch_test = []
    batch_size = [4, 8, 16, 32, 64, 128, 256]

    for size in batch_size:
        time_init = time.time()
        model.reset(batch_size=size)
        model.seed()
        correlations = model.iterate()
        correlation_ra = np.array(correlations)
        computation_time = 1000*(time.time()-time_init)/ITERATION_NUM
        model.save()
        batch_test.append(correlation_ra)

    plt.figure()
    plt.rcParams.update({'font.size': 6})

    for idx in range(SOUND_NUM):
        plt.subplot(SOUND_NUM, 1, idx+1)

        for subidx in range(len(batch_size)):
            plt.plot(10*np.arange(ITERATION_NUM/10), batch_test[subidx][:,idx], label = "Batch:"+str(batch_size[subidx]))

        plt.xlim([0, ITERATION_NUM])
        plt.ylim([-1.05, 1.05])
        plt.ylabel("Signal #{}\nCorrelation".format(idx+1))

    plt.xlabel("Iteration")
    plt.legend(loc="lower center", bbox_to_anchor=(1.1, -0.3))
    plt.tight_layout()
    plt.savefig("{}/batch_test.png".format(PATH_PLOT), dpi=300)

    plt.figure()
    plt.rcParams.update({'font.size': 6})

    for idx in range(SOUND_NUM):
        plt.subplot(SOUND_NUM, 1, idx+1)

        for subidx in range(len(batch_size)):
            plt.plot(10*np.arange(ITERATION_NUM/10), np.abs(batch_test[subidx][:,idx]), label = "Batch:"+str(batch_size[subidx]))

        plt.xlim([0, ITERATION_NUM])
        plt.ylim([-1.05, 1.05])
        plt.ylabel("Signal #{}\nAbsolute\nCorrelation".format(idx+1))

    plt.xlabel("Iteration")
    plt.legend(loc="lower center", bbox_to_anchor=(1.1, -0.3))
    plt.tight_layout()
    plt.savefig("{}/batch_abs_test.png".format(PATH_PLOT), dpi=300)

    print("Batch tests are over.")

    eta_test = []
    eta_size = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    for size in eta_size:
        time_init = time.time()
        model.reset(eta=size)
        model.seed()
        correlations = model.iterate()
        correlation_ra = np.array(correlations)
        computation_time = 1000*(time.time()-time_init)/ITERATION_NUM
        model.save()
        eta_test.append(correlation_ra)


    plt.figure()
    plt.rcParams.update({'font.size': 6})

    for idx in range(SOUND_NUM):
        plt.subplot(SOUND_NUM, 1, idx+1)

        for subidx in range(len(eta_size)):
            plt.plot(10*np.arange(ITERATION_NUM/10), eta_test[subidx][:,idx], label = "\u03B7="+str(eta_size[subidx]))

        plt.ylim([0, ITERATION_NUM])
        plt.ylim([-1.05, 1.05])
        plt.ylabel("Signal #{}\nCorrelation".format(idx+1))

    plt.xlabel("Iteration")
    plt.legend(loc="lower center", bbox_to_anchor=(1.1, -0.3))
    plt.tight_layout()
    plt.savefig("{}/eta_test.png".format(PATH_PLOT), dpi=300)

    plt.figure()
    plt.rcParams.update({'font.size': 6})

    for idx in range(SOUND_NUM):
        plt.subplot(SOUND_NUM, 1, idx+1)

        for subidx in range(len(eta_size)):
            plt.plot(10*np.arange(ITERATION_NUM/10), np.abs(eta_test[subidx][:,idx]), label = "\u03B7="+str(eta_size[subidx]))

        plt.ylim([0, ITERATION_NUM])
        plt.ylim([-1.05, 1.05])
        plt.ylabel("Signal #{}\nAbsolute\nCorrelation".format(idx+1))

    plt.xlabel("Iteration")
    plt.legend(loc="lower center", bbox_to_anchor=(1.1, -0.3))
    plt.tight_layout()
    plt.savefig("{}/eta_abs_test.png".format(PATH_PLOT), dpi=300)

    print("Eta tests are over.")


    sample_test = []
    sample_size = [2, 3, 4, 5]

    for size in sample_size:
        time_init = time.time()
        model.reset()
        model.seed(sample_num=size)
        correlations = model.iterate()
        correlation_ra = np.array(correlations)
        computation_time = 1000*(time.time()-time_init)/ITERATION_NUM
        model.save()
        sample_test.append(correlation_ra)

    plt.figure()
    plt.rcParams.update({'font.size': 6})

    for idx in range(SOUND_NUM):
        plt.subplot(SOUND_NUM, 1, idx+1)

        for subidx in range(len(sample_size)):
            plt.plot(10*np.arange(ITERATION_NUM/10), sample_test[subidx][:,idx], label = "Sample:"+str(sample_size[subidx]))

        plt.ylim([0, ITERATION_NUM])
        plt.ylim([-1.05, 1.05])
        plt.ylabel("Signal #{}\nCorrelation".format(idx+1))

    plt.xlabel("Iteration")
    plt.legend(loc="lower center", bbox_to_anchor=(1.1, -0.3))
    plt.tight_layout()
    plt.savefig("{}/sample_test.png".format(PATH_PLOT), dpi=300)

    plt.figure()
    plt.rcParams.update({'font.size': 6})

    for idx in range(SOUND_NUM):
        plt.subplot(SOUND_NUM, 1, idx+1)

        for subidx in range(len(sample_size)):
            plt.plot(10*np.arange(ITERATION_NUM/10), np.abs(sample_test[subidx][:,idx]), label = "Sample:"+str(sample_size[subidx]))

        plt.ylim([0, ITERATION_NUM])
        plt.ylim([-1.05, 1.05])
        plt.ylabel("Signal #{}\nAbsolute\nCorrelation".format(idx+1))

    plt.xlabel("Iteration")
    plt.legend(loc="lower center", bbox_to_anchor=(1.1, -0.3))
    plt.tight_layout()
    plt.savefig("{}/sample_abs_test.png".format(PATH_PLOT), dpi=300)

    print("Sample tests are over.")


    model.load("batch-{}_eta-{}_mix-{}_sample-{}".format(BATCH_SIZE, ETA, SOUND_NUM, SAMPLE_NUM))

    model.show_original_signals()
    model.show_mixed_singals()
    model.show_restored_signals()

    plt.figure()
    plt.rcParams.update({'font.size': 10})

    for idx in range(SOUND_NUM):
        plt.plot(10*np.arange(ITERATION_NUM/10), batch_test[2][:,idx], label = "Signal #"+str(idx+1))
        plt.xlim([0, ITERATION_NUM])
        plt.ylim([-1.05, 1.05])
        plt.ylabel("Correlation")

    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}/basic_test.png".format(PATH_PLOT), dpi=300)


    plt.figure()
    plt.rcParams.update({'font.size': 10})

    for idx in range(SOUND_NUM):
        plt.plot(10*np.arange(ITERATION_NUM/10), np.abs(batch_test[2][:,idx]), label = "Signal #"+str(idx+1))
        plt.xlim([0, ITERATION_NUM])
        plt.ylim([-1.05, 1.05])
        plt.ylabel("Absolute Correlation")

    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}/basic_abs_test.png".format(PATH_PLOT), dpi=300)

    print("Plotted saved data: batch-{}, eta-{}, mix-{}, sample-{}".format(BATCH_SIZE, ETA, SOUND_NUM, SAMPLE_NUM))
