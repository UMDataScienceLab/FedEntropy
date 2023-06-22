import copy
import numpy as np
import torch
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def average_weights(w, weights=None):
    """
    Returns the average of the weights.
    """
    
    if weights is None:
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg
    else:
        w_avg = {
            key: weights[0] * values for key, values in copy.deepcopy(w[0]).items()
        }
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += weights[i] * w[i][key]
        return w_avg


def sampling_weight(model, gamma):
    """ Sample a set of weights from Gaussian distribution 
    """
    sample_weight = dict()
    for name, param in model.named_parameters():
        sample_weight[name] = torch.normal(
            mean=param, std=1/torch.sqrt(torch.tensor(gamma))
        )
    
    return sample_weight    

def grad_weighted_average(grad_list, density):
    
    grad_avg = {
        name: torch.zeros_like(element[0]) for name, element in grad_list.items()
    }
    
    for name, element in grad_list.items():
        for i in range(len(element)):
            grad_avg[name].add_(density[i] * element[i])
    
    return grad_avg
    

def move_weight(grad, model, eta=1):
    
    model_weights = model.state_dict()
    for key, item in model_weights.items():
        item.add_(grad[key], alpha=-eta)
    
    return model_weights


def count_observations(dataset, dict_clients):
    nobs_device = {
        key: value.shape[0] for key, value in dataset['train']['inputs_bd'].items()
    }
    nobs_client = dict()
    for key, value in dict_clients.items():
        nobs = 0
        for i_client in list(value):
            nobs += nobs_device[i_client]
        nobs_client[key] = nobs
    return nobs_client
        
        
def tensor_weighted_average(tensor_list, weight):
    
    tensor_w_averaged = torch.zeros_like(tensor_list[0])
    
    for i, tensor in enumerate(tensor_list):
        tensor_w_averaged.add_(tensor*weight[i]) 
    
    return tensor_w_averaged
        
        
        
def get_FLclients(dataset, num_clients, iid=True):
    
    if iid:
        dict_clients = NASA_iid(dataset, num_clients)
    return dict_clients


def NASA_iid(dataset, num_clients):
    """
    """
    all_idxs = list(dataset['train']['inputs_bd'].keys())
    num_train = len(all_idxs)
    num_items = int(num_train/num_clients)
    dict_clients = {}
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    
    return dict_clients   
        
        
def local_control_variate_update(c_i, c, grad, K, eta_l):
    
    c_i_plus = dict()
    for key, item in c_i.items():
        c_i_plus[key] = c_i[key] - c[key] + 1/(K*eta_l) * grad[key]

    return c_i_plus


def global_control_variate_update(c, c_delta, S, N):

    c_updated = dict()
    for key, item in c.items():
        c_updated[key] = c[key] + S/N * c_delta[key]
    
    return c_updated


def local_control_variate_grad_calculate(c_i_plus, c_i):

    c_i_delta = dict()
    for key, item in c_i_plus.items():
        c_i_delta[key] = c_i_plus[key] - c_i[key]
    
    return c_i_delta


        
        