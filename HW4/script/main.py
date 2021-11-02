#! /usr/bin/python3

import os
import csv
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import random


PATH_SCRIPT = os.path.dirname(os.path.realpath(__file__))
PATH_ROOT   = os.path.dirname(PATH_SCRIPT)
PATH_DATA   = os.path.join(PATH_ROOT, 'data')
PATH_PLOT   = os.path.join(PATH_ROOT, 'plot')

MAX_ITERATION = 2000

CONFIDENCE_STD_95 = 1.96


def load_data(tag, marker):

    dic_data = {}
    x_t = 'frame'
    marker_x = marker+'_x'
    marker_y = marker+'_y'
    marker_z = marker+'_z'
    marker_c = marker+'_c'

    for root, dirs, files in os.walk(os.path.join(PATH_DATA, tag)):

        if len(files) != 0:
            file = files[-1]

            if file.endswith("csv"):
                data = np.array([0 , 0, 0])

                with open(os.path.join(root, file), newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        data = np.vstack([data, np.array([row[x_t], row[marker_x], row[marker_c]],dtype=float)])
                data = np.delete(data,0,0)

                dic_data[file] = data

    return dic_data



def mix_data(dic_data):

    length = 1010
    cnt=0
    data = np.zeros((length,2), dtype=float)

    ra_data = list(dic_data.values())

    for i in range(length):

        if all([data[i,2]<0 for data in ra_data]):
            continue

        done = False
        while not done:

            sample_data = random.choice(ra_data)

            if sample_data[i,2]>0:
                data[cnt,:] = sample_data[i,:-1]
                cnt+=1
                done = True
    return data


os.makedirs(PATH_DATA, exist_ok=True)
os.makedirs(PATH_PLOT, exist_ok=True)
np.random.seed(15)

dic_data = load_data("YY", '14')
data = mix_data(dic_data)

sigma_f_vals = []
sigma_l_vals = []
sigma_n_vals = []

time_horizon=50
time_start=0
time_end = 1010
time_stride = 5
time_points = np.arange(time_start, time_end-time_horizon, time_stride)

sigma_f = -4
sigma_l=-4
sigma_n = -4

eta = 1e-2


## Compute optimal hyperparameters, process local GP ##
plt.figure(0)
local_se_losses = []
local_lh_losses = []

for point in time_points:
    
    start_idx = point
    end_idx = point+time_horizon

    data_cur = data[start_idx:end_idx]

    X = data_cur[:,0].reshape(-1,1)
    Y = data_cur[:,1].reshape(-1,1)
    N = X.shape[0]

    Xii = X * X
    Xii = np.tile(Xii, N)
    XXi_XXj = Xii + Xii.T - 2 * X @ X.T

    for _ in range(MAX_ITERATION):

        kp = np.exp(sigma_f - 0.5*np.exp(sigma_l)*XXi_XXj)
        kl = -0.5 * np.exp(sigma_l) * XXi_XXj

        Q = kp + np.exp(sigma_n) * np.eye(N)
        Qinv = np.linalg.inv(Q)

        dPdf = 0.5 * (Y.T @ Qinv @ kp @ Qinv @ Y - np.trace(Qinv @ kp))
        dPdl = 0.5 * (Y.T @ Qinv @ kp * kl @ Qinv @ Y - np.trace(Qinv @ kp * kl))
        dPdn = 0.5 * np.exp(sigma_n) * (Y.T @ Qinv @ Qinv@ Y - np.trace(Qinv))
        
        dPdf = dPdf[0,0]
        dPdl = dPdl[0,0]
        dPdn = dPdn[0,0]

        sigma_f += eta*dPdf
        sigma_l += eta*dPdl
        sigma_n += eta*dPdn

        if 1 < dPdf**2 +  dPdl**2 + dPdn**2:
            break

    sigma_f_vals.append(sigma_f)
    sigma_l_vals.append(sigma_l)
    sigma_n_vals.append(sigma_n)

    Xstar = np.linspace(X[0,],X[-1,]).reshape(-1, 1)
    N1 = Xstar.size

    Xstar1 = np.tile(X,N1)
    Xstar2 = np.tile(Xstar,N).T

    XXstar1 = Xstar1 * Xstar1
    XXstar2 = Xstar2 * Xstar2
    XXstar = XXstar1 + XXstar2 - 2 * X @ Xstar.T

    Xstartmp = np.linspace(X[0],X[-1]);#XX+0.5;
    Xstartmp = Xstartmp.reshape(-1,1)
    Xstartmp = Xstartmp.transpose()
    XstarCov = np.multiply(matlib.repmat(Xstartmp,N1,1),matlib.repmat(Xstartmp,N1,1))\
         + (np.multiply(matlib.repmat(Xstartmp,N1,1),matlib.repmat(Xstartmp,N1,1))).transpose() \
             -2*Xstartmp.T.dot(Xstartmp)

    KXstarX = np.exp(sigma_f - 0.5*np.exp(sigma_l)*XXstar.T)

    kp = np.exp(sigma_f - 0.5*np.exp(sigma_l)*XXi_XXj) + np.eye(N)*np.exp(sigma_n)
    kpinv = np.linalg.inv(kp)
    mXstar = KXstarX @ kpinv @ Y

    PXstar = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XstarCov) + np.eye(N1)*np.exp(sigma_n) - KXstarX.dot(kpinv).dot(KXstarX.transpose())

    confidence_range = CONFIDENCE_STD_95 * np.sqrt(np.diagonal(PXstar))
    se_loss = np.mean((Y-mXstar)**2)

    term1 = -1 / 2 * Y.T @ kpinv @ Y
    term2 = -1 / 2 * np.log(np.linalg.det(kp))
    term3 = N / 2 * np.log(2 * np.pi)
    lh_loss = np.asscalar(term1 + term2 + term3)/N

    local_se_losses.append(se_loss)
    local_lh_losses.append(lh_loss)

    plt_mean, = plt.plot(Xstar, mXstar, color="k", label='Local GP mean', zorder=2)
    plt_error = plt.fill_between(Xstar.flatten(), mXstar.flatten()-confidence_range, mXstar.flatten()+confidence_range, facecolor='#089FFF', zorder=0, label='95% confidence')
    plt_data = plt.scatter(X.flatten(), Y.flatten(), s= 5, color = "r", label='Input data', zorder=1)

local_se_losses = np.array(local_se_losses)
local_lh_losses = np.array(local_lh_losses)

plt.title('Local GP kernels')
plt.xlabel('Frame')
plt.ylabel('Position [m]')
plt.xlim([time_start, time_end-time_horizon])
plt.legend(handles=[plt_mean, plt_error, plt_data])
plt.savefig("{}/local.png".format(PATH_PLOT))


plt.figure(1)

plt.plot(time_points, sigma_f_vals, color="r", label='$\sigma_f$')
plt.plot(time_points, sigma_l_vals, color="g", label='$\sigma_l$')
plt.plot(time_points, sigma_n_vals, color="b", label='$\sigma_n$')

plt.title('Local hyperparameters')
plt.xlabel('Frame')
plt.ylabel('Hyperparameters')
plt.xlim([time_start, time_end-time_horizon])
plt.legend(loc='upper right')
plt.savefig("{}/hyper.png".format(PATH_PLOT))


sigma_f_means = []
sigma_l_means = []
sigma_n_means = []
len_interval = 50

for point in time_points:

    interval_start = int((int(point/len_interval) * len_interval)/time_stride)
    interval_end = int((int(point/len_interval) * len_interval + len_interval)/time_stride)
    sigma_f_means.append(np.mean(sigma_f_vals[interval_start:interval_end]))
    sigma_l_means.append(np.mean(sigma_l_vals[interval_start:interval_end]))
    sigma_n_means.append(np.mean(sigma_n_vals[interval_start:interval_end]))

plt.figure(5)

plt.plot(time_points, sigma_f_means, color="r", label='$\sigma_f$')
plt.plot(time_points, sigma_l_means, color="g", label='$\sigma_l$')
plt.plot(time_points, sigma_n_means, color="b", label='$\sigma_n$')

plt.title('Local hyperparameter means in intervals')

plt.xticks(np.arange(time_start, time_end, 100))
plt.grid(color='k', linestyle='--', linewidth=1)
plt.xlabel('Frame')
plt.ylabel('Hyperparameters')
plt.xlim([time_start, time_end-time_horizon])
plt.legend(loc='upper right')
plt.savefig("{}/interval.png".format(PATH_PLOT))

## Process global GP ##

plt.figure(2)
means = {}
devs = {}
global_lh_losses = []

print("Optimal hyperparameters for global GP")
print("Sigma f: ", sigma_f)
print("Sigma l: ", sigma_l)
print("Sigma n: ", sigma_n)

for point in time_points:

    means[point] = []
    devs[point] = []

for point in time_points:

    start_idx = point
    end_idx = point+time_horizon

    data_cur = data[start_idx:end_idx]

    X = data_cur[:,0].reshape(-1,1)
    Y = data_cur[:,1].reshape(-1,1)
    N = X.shape[0]

    Xii = X * X
    Xii = np.tile(Xii, N)
    XXi_XXj = Xii + Xii.T - 2 * X @ X.T

    Xstar = np.linspace(X[0,],X[-1,]).reshape(-1, 1)
    N1 = Xstar.size

    Xstar1 = np.tile(X,N1)
    Xstar2 = np.tile(Xstar,N).T

    XXstar1 = Xstar1 * Xstar1
    XXstar2 = Xstar2 * Xstar2
    XXstar = XXstar1 + XXstar2 - 2 * X @ Xstar.T

    Xstartmp = np.linspace(X[0],X[-1])
    Xstartmp = Xstartmp.reshape(-1,1)
    Xstartmp = Xstartmp.transpose()
    XstarCov = np.multiply(matlib.repmat(Xstartmp,N1,1),matlib.repmat(Xstartmp,N1,1))\
         + (np.multiply(matlib.repmat(Xstartmp,N1,1),matlib.repmat(Xstartmp,N1,1))).transpose() \
             -2*Xstartmp.T.dot(Xstartmp)

    KXstarX = np.exp(sigma_f - 0.5*np.exp(sigma_l)*XXstar.T)

    kp = np.exp(sigma_f - 0.5*np.exp(sigma_l)*XXi_XXj) + np.eye(N)*np.exp(sigma_n)
    kpinv = np.linalg.inv(kp)
    mXstar = KXstarX @ kpinv @ Y

    PXstar = np.exp(sigma_f-0.5*np.exp(sigma_l)*XstarCov)\
             + np.eye(N1)*np.exp(sigma_n) - KXstarX.dot(kpinv).dot(KXstarX.transpose())

    term1 = -1 / 2 * Y.T @ kpinv @ Y
    term2 = -1 / 2 * np.log(np.linalg.det(kp))
    term3 = N / 2 * np.log(2 * np.pi)
    lh_loss = np.asscalar(term1 + term2 + term3)/N

    global_lh_losses.append(lh_loss)

    for idx in range(mXstar.shape[0]):

        if start_idx+idx not in means:
            means[start_idx+idx] = []
        if start_idx+idx not in devs:
            devs[start_idx+idx] = []

        means[start_idx+idx].append(mXstar[idx])
        devs[start_idx+idx].append(np.sqrt(PXstar[idx][idx]))

    plt_data = plt.scatter(X.flatten(), Y.flatten(), s= 5, color = "r", label='Input data', zorder=1)


global_se_losses = []
global_means = []
global_confs = []
for point in time_points:

    if len(means[point]) == 0: 
        global_means.append(np.nan)
    else:
        mean = np.mean(means[point])
        global_means.append(mean)
        global_se_losses.append((mean-data[point,1])**2)

    if len(devs[point]) == 0: 
        global_confs.append(np.nan)
    else:
        global_confs.append(CONFIDENCE_STD_95 *np.mean(devs[point]))

global_means = np.array(global_means)
global_confs = np.array(global_confs)
global_se_losses = np.array(global_se_losses)
global_lh_losses = np.mean(global_lh_losses)*np.ones(time_points.shape)

plt_error = plt.fill_between(time_points, global_means-global_confs, global_means+global_confs, facecolor='#089FFF', zorder=0, label='95% confidence')
plt_mean, = plt.plot(time_points, global_means, color="k", label='Global GP mean', zorder=2)

plt.title('Global GP kernel')
plt.xlabel('Frame')
plt.ylabel('Position [m]')
plt.xlim([time_start, time_end-time_horizon])
plt.legend(handles=[plt_mean, plt_error, plt_data])
plt.savefig("{}/global.png".format(PATH_PLOT))

plt.figure(3)
plt.plot(time_points, local_se_losses, color="r", label='Local GP loss')
plt.plot(time_points, global_se_losses, color="b", label='Global GP loss')
plt.title('Loss evaluation (suqare error)')
plt.xlabel('Frame')
plt.ylabel('Square error [$m^2$]')
plt.xlim([time_start, time_end-time_horizon])
plt.legend()
plt.savefig("{}/square_error.png".format(PATH_PLOT))

plt.figure(4)
plt.plot(time_points, local_lh_losses.flatten(), color="r", label='Local GP loss')
plt.plot(time_points, global_lh_losses.flatten(), color="b", label='Global GP loss')
plt.title('Loss evaluation (log likelihood)')
plt.xlabel('Frame')
plt.ylabel('Log liklihood')
plt.legend(loc='upper right')
plt.xlim([time_start, time_end-time_horizon])
plt.legend()
plt.savefig("{}/log_likelihood.png".format(PATH_PLOT))

plt.show()
