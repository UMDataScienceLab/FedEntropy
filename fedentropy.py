import time
import copy

import numpy as np
import numpy.random as npr
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data
import models
from FL.options import args_parser
from FL.train import CentralizedUpdate, LocalUpdate
from FL.utils import average_weights, sampling_weight, grad_weighted_average
from FL.utils import move_weight, count_observations, tensor_weighted_average
from FL.utils import get_FLclients

## arguments
args = args_parser(option='federated', realdata="NASA")

GPU = False   
N = 10
N_TEST = 5
T = 50

## data generation
mu, Sig, sig2 = dict(), dict(), dict()

mu[0] = np.array([2.5,  0.02,   0.035])
het = 100
Sig[0] = np.diag([1e-2, 3e-5, 1e-7])*het
sig2[0] = 1
npr.seed(args.seed)

notgo = True
while notgo:
    
    raw_data_train = data.poly_data_generation(N=N, T=T, mu=mu, Sig=Sig, sig2=sig2)
    raw_data_test = data.poly_data_generation(N=N_TEST, T=T, mu=mu, Sig=Sig, sig2=sig2)

    train_MAX = np.array(list(raw_data_train['y_true'].values()))[:,T-1].max()
    train_MIN = np.array(list(raw_data_train['y_true'].values()))[:,T-1].min()
    test_MAX = np.array(list(raw_data_test['y_true'].values()))[:,T-1].max()
    test_MIN = np.array(list(raw_data_test['y_true'].values()))[:,T-1].min()

    if (test_MAX < train_MAX) & (train_MIN < test_MIN):
        notgo = False

options = {
    "denoising":        False,
    "savgol_wlen":      10, 
    "polyorder":        3, 
    "standardizing":    True, 
    "time_window":      args.time_window, 
    "dim_output":       args.dim_output, 
    "add_time_attr":    False, 
    "loss_penalty":     True, 
    "batch_size":       args.local_bs
}

data_train_tw = data.data_timewindow(
    X=raw_data_train['X'], y=raw_data_train['y'], options=options
)
options['std_factors'] = data_train_tw['std_factors']
data_test_tw = data.data_timewindow(
    X=raw_data_test['X'], y=raw_data_test['y'], options=options
)

dataset = dict(train=data_train_tw, test=data_test_tw)

trainloader = dataset['train']['loader']
x_train = dataset['train']['inputs'] 
y_train = dataset['train']['targets'] 
x_bd_train = dataset['train']['inputs_bd']
y_bd_train = dataset['train']['targets_bd'] 

testloader = dataset['test']['loader']
x_test = dataset['test']['inputs'] 
y_test = dataset['test']['targets'] 
x_bd_test = dataset['test']['inputs_bd']
y_bd_test = dataset['test']['targets_bd'] 

# FL datasets
dict_clients = get_FLclients(
    dataset=dataset, num_clients=args.num_users, iid=True)  
nobs_client = count_observations(dataset, dict_clients)

if GPU:
    torch.cuda.set_device(GPU)
device = 'cuda' if GPU else 'cpu'

################################### model #####################################
model_cfg = getattr(models, args.model)
print("Preparing models")
global_model = model_cfg.base(
    *model_cfg.args,
    len_seq = x_train.shape[1],
    dim_in_feature = x_train.shape[2], 
    dim_output = args.dim_output,
    dim_hidden=[2, 30, 100, 50],
    init_log_noise = 1 if args.criterion == "LogGaussianLoss" else None,
    num_groups=2,
    norm='group'
)

global_model.to(device)
global_model.train()
sample_model = copy.deepcopy(global_model)

# check initial value
outputs = global_model(next(iter(trainloader))[0])
print("Output shape:", outputs.shape)
print("Input shape:", x_train[0].shape)

global_model.to(device)
global_model.train()
save_model = copy.deepcopy(global_model)
sample_model = copy.deepcopy(global_model)

def schedule_global_lr(init_eta, epoch):
    
    # return init_eta
    if epoch < 20:
        return init_eta
    else:
        return init_eta * (1/10)

def scoping(init_gamma, scoping_factor, epoch):
    return init_gamma*(1+scoping_factor)**epoch

start_time = time.time()

for comm_round in range(args.epochs):

    # initialize
    global_model.train()
    m = max(int(args.frac * args.num_users), 1)
    idxs_clients = np.random.choice(range(args.num_users), m, replace=False)

    sample_global_losses = []
    sample_global_grads = {
        name: [] for name in global_model.state_dict().keys()
    }
    weight_clients = np.array(
        [nobs_client[i_client] for i_client in idxs_clients]
    ); weight_clients = weight_clients/np.sum(weight_clients)

    # local lr
    if 40 < comm_round & comm_round <= 80:
        lr = 0.001
    elif 80 < comm_round & comm_round <= 120:
        lr = 0.0001
    elif 120 < comm_round:
        lr = 0.00001
    else: lr = args.lr


    for i_sample in range(args.L):
            
        local_losses = []
        local_grads = {name:[] for name in sample_global_grads.keys()}
        
        # sample weights
        gamma = scoping(init_gamma=args.gamma, scoping_factor=args.scoping, epoch=comm_round)
        sample_model.load_state_dict(
            sampling_weight(model=global_model, gamma=gamma)
        )
        
        # local updates in clients
        for i, i_clients in enumerate(idxs_clients):
            
            client = LocalUpdate(args=args, dataset=dataset,
                client_idx=list(dict_clients[i_clients])
            )
            
            results_client_sample = client.local_update(
                model=copy.deepcopy(sample_model), epoch=comm_round, lr=lr
            )
            
            # save local grads
            for name in local_grads.keys():
                local_grads[name].append(copy.deepcopy(
                        results_client_sample['grad'][name]
                    )
                )
            
            # save local loss
            local_losses.append(
                copy.deepcopy(
                    client.inference(model=sample_model, infer='train')['loss'].detach()
                )
            )
        
        # global loss: average local loss
        sample_global_losses.append(-torch.stack(local_losses).mean())
        
        # global gradients: average local gradients
        for name in local_grads.keys(): 
            sample_global_grads[name].append(
                tensor_weighted_average(local_grads[name], weight_clients)
            )
    
    # calculate density based on the average local loss at each sample
    sample_global_losses = torch.stack(sample_global_losses)
    sample_global_losses = torch.exp(
        sample_global_losses - torch.max(sample_global_losses)
    )
    sample_density = sample_global_losses / torch.sum(sample_global_losses)
    
    # entropy global gradient: weighted average of sample gradients based on density
    entropy_grad = grad_weighted_average(
        grad_list=sample_global_grads, density=sample_density
    )

    # update weight using entropy global gradient 
    if args.lr_schedule == 'decay':
        eta = schedule_global_lr(args.eta, comm_round)
    elif args.lr_schedule == 'const': 
        eta = args.eta
    
    move_weight(grad=entropy_grad, model=global_model, eta=eta)

    # Calculate avg training loss over all users at every epoch
    loss_clients = []
    global_model.eval()
    for i in range(args.num_users):
        client = LocalUpdate(args=args, dataset=dataset,
            client_idx=list(dict_clients[i])
        )
        loss_clients.append(
            client.inference(model=global_model, infer='train')['loss'].detach()
        )
    loss_clients_avg = sum(loss_clients)/len(loss_clients)
    
    # Calculate test loss
    results_global = client.inference(model=global_model, infer='test')
    
    
    # print
    print(
        "** Epoch {} | \tElapsed Time: {:.1f} | \tLR : {:.5f} | \tTrain Loss (client avg)): {:.4f} | \tTest Loss (global)): {:.4f} | \tGamma: {:.2f} |\tEta: {:.1f}".format(
            comm_round+1, 
            time.time()-start_time, 
            lr,
            loss_clients_avg, 
            results_global['loss'], 
            gamma, eta
        )
    )