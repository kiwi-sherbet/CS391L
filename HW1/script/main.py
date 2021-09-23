#! /usr/bin/python3

import os
import pickle
import gzip
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

PATH_SCRIPT = os.path.dirname(os.path.realpath(__file__))
PATH_ROOT   = os.path.dirname(PATH_SCRIPT)
PATH_DATA   = os.path.join(PATH_ROOT, 'data')
PATH_SAVE   = os.path.join(PATH_ROOT, 'save')
PATH_PLOT   = os.path.join(PATH_ROOT, 'plot')

os.makedirs(PATH_DATA, exist_ok=True)
os.makedirs(PATH_SAVE, exist_ok=True)
os.makedirs(PATH_PLOT, exist_ok=True)

TRAIN_SIZE = 1000
PCA_SIZE = 24
K = 5


def save_data(data):

    with open("{}/train-{}.pth".format(PATH_SAVE, data["sample size"]),"wb") as f:
        pickle.dump(data, f)


def load_data(file_name):

    with open("{}/{}.pth".format(PATH_SAVE, file_name),"rb") as f:
        return pickle.load(f)


def load_MNIST():

    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    path = os.path.join(PATH_DATA, 'mnist')
    os.makedirs(path, exist_ok=True)

    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    def _images(path):
        with gzip.open(path) as f:
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def _labels(path):
        with gzip.open(path) as f:
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        def _onehot(integer_labels):
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype='uint8')
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

        return _onehot(integer_labels)

    train_set = {}
    test_set = {}
    
    train_set["images"] = _images(os.path.join(path, files[0]))
    train_set["labels"] = _labels(os.path.join(path, files[1]))
    test_set["images"] = _images(os.path.join(path, files[2]))
    test_set["labels"] = _labels(os.path.join(path, files[3]))

    return train_set, test_set


def reduce_set(set, size):

    if set["images"].shape[0] > size:

        reduced_set = {}
        reduced_set["images"] = set["images"][:size]
        reduced_set["labels"] = set["labels"][:size]

        return reduced_set

    else:
        return copy.copy(set)



def find_eigen_vecs(inputs):

    nums, pixels = inputs.shape
    means = np.mean(inputs, axis=0)
    vecs = inputs - means

    covs = (vecs.T @ vecs)/nums

    eig_vals, eig_vecs = np.linalg.eig(covs)

    return eig_vals, eig_vecs


def find_KNNs(target_vec, sample_vecs, K):

    dists = np.linalg.norm(target_vec-sample_vecs, axis=1)
    k_args = np.argpartition(dists, K)

    return k_args


class PCA_model():

    def __init__(self, pca_size = PCA_SIZE, k = K):
        self.pca_size = pca_size
        self.k = k
        self.sample_size = 0


    def train(self, train_set):

        self.sample_imgs = train_set["images"]
        self.sample_labels = train_set["labels"]
        self.sample_size = self.sample_imgs.shape[0]

        self.eig_vals, self.eig_vecs = find_eigen_vecs(self.sample_imgs)
        self.sample_proj = self.sample_imgs @ self.eig_vecs

        results = { "sample size":self.sample_size, 
                    "sample proj": self.sample_proj, "sample labels": self.sample_labels, "eigen vectors": self.eig_vecs}

        save_data(results)

        return results


    def load(self, file_name):

        results = load_data(file_name)

        if results != None:
            self.sample_size = results["sample size"]
            self.sample_proj = results["sample proj"]
            self.sample_labels = results["sample labels"]
            self.eig_vecs = results["eigen vectors"]


    def test(self, test_set):

        test_imgs = test_set["images"]
        test_labels = test_set["labels"]
        test_size = test_imgs.shape[0]

        sample_pca_proj = self.sample_proj[:, :self.pca_size]
        test_pca_proj = test_imgs @ self.eig_vecs[:, :self.pca_size]

        num_correct = 0

        for idx in range(test_size):
            k_args = find_KNNs(test_pca_proj[idx], sample_pca_proj, self.k)
            test_digit = np.argmax(test_labels[idx])
            infer_digit = np.argmax(self.sample_labels[k_args])

            if (infer_digit == test_digit):
                num_correct += 1

        accuracy = num_correct/test_size

        print("Success rate: {}%".format(100*accuracy))
    
        return accuracy


    def infer(self, input_img):

        sample_pca_proj = self.sample_proj[:, :self.pca_size]
        input_pca_proj = input_img @ self.eig_vecs[:, :self.pca_size]

        k_args = find_KNNs(input_pca_proj, sample_pca_proj, self.k)
        infer_digit = np.argmax(self.sample_labels[k_args])

        return infer_digit, input_pca_proj


    def show_PCAs(self, num_PCAs=None):

        if num_PCAs == None:
            num_PCAs = self.pca_size

        fig_cols = 6
        fig_rows = int((num_PCAs-1)/fig_cols) + 1

        plt.figure()

        for idx in range(num_PCAs):
            pca_img = self.eig_vecs[:, idx]

            plt.subplot(fig_rows, fig_cols, idx + 1)
            plt.axis('off')
            plt.title(str(idx + 1))
            plt.imshow(pca_img.reshape(28, 28, 1).astype(np.float32))

        plt.savefig("{}/eigen_vectors.png".format(PATH_PLOT))


    def show_eig_vals(self, num_PCAs=None):

        plt.figure()
        plt.plot(self.eig_vals)
        plt.xlabel("Eigen vecrors")
        plt.ylabel("Eigen values")

        plt.savefig("{}/eigen_values.png".format(PATH_PLOT))


    def show_reconst_digits(self, test_set):

        test_imgs = test_set["images"]
        test_labels = test_set["labels"]
        test_size = test_imgs.shape[0]

        fig_cols = 5
        fig_rows = int((test_size-1)/fig_cols) + 1

        plt.figure()

        for idx in range(test_size):
            input_img = test_imgs[idx]
            infer_digit, input_pca_proj = self.infer(input_img)
            reconst_img = self.eig_vecs[:, :self.pca_size] @ input_pca_proj
            
            plt.subplot(fig_rows, fig_cols, idx + 1)
            plt.axis('off')
            plt.title(idx+1)
            plt.text(0, 32, "inferred as {}".format(infer_digit))
            plt.imshow(reconst_img.reshape(28, 28, 1).astype(np.float32))

        plt.savefig("{}/reconst-PCA-{}.png".format(PATH_PLOT,self.pca_size))


if __name__ == "__main__":
    train_set, test_set = load_MNIST()

    # Training data
    model = PCA_model()
    model.train(reduce_set(train_set, 1000))
    model.test(test_set)
    print("Training is over.")

    # Plotting PCAs and corresponding eigen values
    model.show_eig_vals()
    model.show_PCAs()
    print("Trained PCAs are plotted.")

    # Testing the model with different PCA sizes
    pca_test = []
    pca_time = []
    pca_sizes = [5, 10, 20, 50, 100, 500, 784]
    for size in pca_sizes:
        time_init = time.time()
        model.pca_size = size
        accuracy = 100 * model.test(test_set)
        computation_time = (time.time()-time_init)/10
        model.show_reconst_digits(reduce_set(test_set, 15))
        pca_time.append(computation_time)
        pca_test.append(accuracy)
    model.pca_size = PCA_SIZE

    plt.figure()
    plt.plot(pca_sizes, pca_test)
    plt.xlabel("PCA size")
    plt.ylabel("Accuracy [%]")
    plt.savefig("{}/pca_test.png".format(PATH_PLOT))

    plt.figure()
    plt.plot(pca_sizes, pca_time)
    plt.xlabel("PCA size")
    plt.ylabel("Computation time [msec/image]")
    plt.savefig("{}/pca_time.png".format(PATH_PLOT))

    print("PCA tests are over.")

    # Testing the model with different KNN sizes
    knn_test = []
    knn_time = []
    knn_sizes = [1, 3, 5, 10, 25]
    for size in knn_sizes:
        time_init = time.time()
        model.k = size
        accuracy = 100 * model.test(test_set)
        computation_time = (time.time()-time_init)/10
        knn_time.append(computation_time)
        knn_test.append(accuracy)
    model.k = K

    plt.figure()
    plt.plot(knn_sizes, knn_test)
    plt.xlabel("KNN size")
    plt.ylabel("Accuracy [%]")
    plt.savefig("{}/knn_test.png".format(PATH_PLOT))

    plt.figure()
    plt.plot(knn_sizes, knn_time)
    plt.xlabel("KNN size")
    plt.ylabel("Computation time [msec/image]")
    plt.savefig("{}/knn_time.png".format(PATH_PLOT))

    print("KNN tests are over.")

    # Testing the model with different training sample sizes
    sample_test = []
    sample_time = []
    sample_sizes = [300, 500, 1000, 5000, 10000, 30000, 60000]
    for size in sample_sizes:
        time_init = time.time()
        model.train(reduce_set(train_set, size))
        accuracy = 100 * model.test(test_set)
        computation_time = (time.time()-time_init)/10
        sample_time.append(computation_time)
        sample_test.append(accuracy)

    plt.figure()
    plt.plot(sample_sizes, sample_test)
    plt.xlabel("Sample size")
    plt.ylabel("Accuracy [%]")
    plt.savefig("{}/sample_test.png".format(PATH_PLOT))

    plt.figure()
    plt.plot(sample_sizes, sample_time)
    plt.xlabel("Sample size")
    plt.ylabel("Computation time [msec/image]")
    plt.savefig("{}/sample_time.png".format(PATH_PLOT))

    print("Training sample tests are over.")

