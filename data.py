import copy
import numpy as np
import numpy.random as npr
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter

def poly_data_generation(T, N, mu, Sig, sig2):

    N_priors = len(mu)

    b = {
        i: {
            j: npr.multivariate_normal(mu[j], Sig[j]) 
            for j in range(N_priors)
        }
        for i in range(N) 
    }

    poly_degree = mu[0].shape[0] - 1

    X = range(T)
    Xt = np.array([np.array(X)**i for i in range(poly_degree+1)]).T 
    y_true, y = dict(), dict()
    for i in range(N):
        if N_priors < 2:
            # print(mu[0].shape[0], Xt, b[i][0])
            y_true[i] = Xt @ b[i][0].T
            y[i] = y_true[i] + npr.multivariate_normal(np.zeros(T), np.eye(T)*sig2[0])
        else:
            raise NameError('N_priors >= 2 Need to implement')
    
    return {
        "X": X,
        "y_true": y_true,
        "y": y
    }



def denoise(y, savgol_wlen=30, polyorder=3):
    
    y_denoised = y.copy()
    for i, y_i in y.items():
        y_denoised[i] = savgol_filter(y_i, savgol_wlen, polyorder)
    
    return y_denoised



def standardize(y, std_factors=None):

    if std_factors is None:
        y_max = max([max(y_i) for y_i in y.values()])
        y_min = min([min(y_i) for y_i in y.values()])
    else:
        y_min, y_max = std_factors

    y_std = y.copy()
    for i, y_i in y.items():
        y_std[i] = (y_i - y_min)/(y_max - y_min)

    return y_std, (y_min, y_max)



def data_timewindow(X, y, options):
    
    # get arguments
    denoising = options.get("denoising", True)
    savgol_wlen = options.get("savgol_wlen", 30)
    polyorder = options.get("polyorder", 3)

    standardizing = options.get("standardizing", True)
    
    time_window = options.get("time_window", 10)
    dim_output = options.get("dim_output", 20)
    
    add_time_attr = options.get("add_time_attr", True)
    loss_penalty = options.get("loss_penalty", True)

    batch_size = options.get("batch_size", 20)
    shuffle = options.get("suffle", False)

    std_factors = options.get("std_factors", None)

    # denoising
    if denoising:
        y = denoise(y, savgol_wlen=savgol_wlen, polyorder=polyorder)

    # standardizing 
    if standardizing:
        y, (std_min, std_max) = standardize(y, std_factors)

    # creating datasets based on time window approach
    inputs, targets = [], []
    inputs_bd, targets_bd = {}, {}
    data_original = {}

    # time attribute
    t_max = max(X)
    t_range = np.linspace(start=0, stop=1, num=t_max)

    for i, y_i in y.items():

        inputs_i, targets_i = [], []
        data_original[i] = y_i
        for i_tw in range(y_i.shape[0] - time_window - dim_output + 1):

            if not add_time_attr:
                input_ij = y_i[i_tw:(i_tw + time_window)][:,None]
            else: 
                input_ij = np.array([
                    y_i[i_tw:(i_tw + time_window)], t_range[i_tw:(i_tw + time_window)]
                ]).T
            
            if not loss_penalty:
                target_ij = y_i[
                    (i_tw + time_window):(i_tw + time_window + dim_output)
                ]
            
            else:
                last_obs_val = savgol_filter(input_ij[:,0], time_window-1, 2)[-1]
                target_ij = np.concatenate([
                    y_i[(i_tw+time_window):(i_tw + time_window + dim_output)], 
                    np.array([last_obs_val])
                ])
            
            inputs.append(input_ij)
            targets.append(target_ij)
            
            inputs_i.append(input_ij)
            targets_i.append(target_ij)

        inputs_bd[i] = np.array(inputs_i)
        targets_bd[i] = np.array(targets_i)

    inputs = np.array(inputs)
    targets = np.array(targets)

    loader, _, _ = data_to_loader(inputs, targets, 
                   batch_size=batch_size, shuffle=shuffle)    

    return {
        "loader": loader,
        "inputs": inputs,
        "targets": targets,
        "inputs_bd": inputs_bd,
        "targets_bd": targets_bd,
        "data_original": data_original,
        "std_factors": (std_min, std_max)
    }


def data_IoFT(data_tw, Nm_units=5):
   
    num_units = len(data_tw['input'])

    indx = np.array_split(np.arange(num_units), Nm_units) 
    data = dict(input={}, target={})
    for i, idx in enumerate(indx):
        data['input'][i] = np.concatenate([data_tw['input'][_i] for _i in idx])
        data['target'][i] = np.concatenate([data_tw['target'][_i] for _i in idx])

    return {
        'data': data,
        'cluster': indx
    }


def data_to_loader(x, y, batch_size, shuffle=False):
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    torchData = torch.utils.data.TensorDataset(x, y)
    data_loader = torch.utils.data.DataLoader(torchData, batch_size=batch_size,
                                              shuffle=shuffle)
    return data_loader, x, y   

