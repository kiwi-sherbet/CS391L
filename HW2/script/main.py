#! /usr/bin/python3

# 1. Write, review, and submit a brief report containing your plots and detailing what you did and how well it worked. -- 4 points (Introduction, Method, Results, Summary, etc.)
# 2. Plot the bottom signals,  the mixed signals, and the recovered signals.  -- 4 points




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

TRAIN_SIZE = 1000
PCA_SIZE = 24
K = 5


def save_data(data):

    with open("{}/train-{}.pth".format(PATH_SAVE, data["sample size"]),"wb") as f:
        pickle.dump(data, f)


def load_data(file_name):

    with open("{}/{}.pth".format(PATH_SAVE, file_name),"rb") as f:
        return pickle.load(f)


def load_Sounds():

    url = 'https://www.cs.utexas.edu/~dana/MLClass/'
    files = ['sounds.mat', 'icaTest.mat']

    path = os.path.join(PATH_DATA, 'sounds')
    os.makedirs(path, exist_ok=True)

    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    sound_set = loadmat(os.path.join(path, files[0]))
    test_matrics = loadmat(os.path.join(path, files[1]))

    return sound_set, test_matrics


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
    train_set, test_set = load_Data()

    print("Training sample tests are over.")

    print(train_set)