import pandas as pd
import numpy as np
import random
from scipy.signal import savgol_filter

import torch

def data_to_loader(x, y, batch_size, shuffle=False):
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    torchData = torch.utils.data.TensorDataset(x, y)
    data_loader = torch.utils.data.DataLoader(torchData, batch_size=batch_size,
                                              shuffle=shuffle)
    return data_loader, x, y   


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


def count_valid_obs(data, time_range):
    
    if time_range is None:
        raise NameError('time_range is None.')
    else:
        if time_range is not None:
            tmp = data[
                (time_range[0]<=data.iloc[:,1]) & (data.iloc[:,1]<=time_range[1])
            ]
    remove_idx = list()
    eng_idx = list(set(data[0]))
    for eng in eng_idx:
        if sum(data[0] == eng) < (time_range[1] - time_range[0] + 1):
            remove_idx.append(eng)
    
    return len(eng_idx) - len(remove_idx)
        

########## tmp code for synthetic data ###########

def synthetic_curve_1(t, w_1, w_2, noise=0.03):
    
    return 0.3*(t**2) - 2*np.sin(w_1*np.pi*t) + w_2 + np.random.normal(0, noise, size=t.shape)

def synthetic_curve_2(t, w_1, w_2, noise=0.03):
    
    return 1.5*t + w_1*np.sin(t) + w_2 + np.random.normal(0, noise, size=t.shape)

def get_synthetic_data(
        num_timepoint=200, num_obs_train=100, num_obs_test=50, 
        batch_size_train=20, batch_size_test=50, last_obs_idx=40, 
        noise=0.03, homogeneous=False, shuffle=True, irregular=False
    ):
    
    t_const = np.linspace(0,8, num=num_timepoint) 

    if homogeneous:
        # train
        num_obs_1 = num_obs_train
        curve_1 = []
        for i_obs in range(num_obs_1):
            if not irregular: t = t_const
            else: t = np.sort(np.random.uniform(0,8, size=num_timepoint))
            w_1, w_2 = np.random.normal(0.4, 0.05), np.random.uniform(0, 7)
            curve_1.append(synthetic_curve_1(t, w_1, w_2, noise))
        curve_train = curve_1
        
        # test
        num_obs_test_1 = num_obs_test
        curve_test_1 = []
        for i_obs in range(num_obs_test_1):
            if not irregular: t = t_const
            else: t = np.sort(np.random.uniform(0,8, size=num_timepoint))
            w_1, w_2 = np.random.normal(0.4, 0.05), np.random.uniform(0, 7)
            curve_test_1.append(synthetic_curve_1(t, w_1, w_2, noise))
        
        curve_test = curve_test_1
    
    else:
        # train
        num_obs_1 = int(num_obs_train/2)
        curve_1 = []
        for i_obs in range(num_obs_1):
            if not irregular: t = t_const
            else: t = np.sort(np.random.uniform(0,8, size=num_timepoint))
            w_1, w_2 = np.random.normal(0.4, 0.05), np.random.uniform(0, 7)
            curve_1.append(synthetic_curve_1(t, w_1, w_2, noise))       
    
        num_obs_2 = num_obs_train - num_obs_1
        curve_2 = []
        for i_obs in range(num_obs_2):
            if not irregular: t = t_const
            else: t = np.sort(np.random.uniform(0,8, size=num_timepoint))
            w_1, w_2 = np.random.uniform(0.8, 1.2), np.random.uniform(0, 7)
            curve_2.append(synthetic_curve_2(t, w_1, w_2, noise))
        
        curve_train = curve_1 + curve_2
        
        # test
        num_obs_test_1 = int(num_obs_test/2)
        curve_test_1 = []
        for i_obs in range(num_obs_test_1):
            if not irregular: t = t_const
            else: t = np.sort(np.random.uniform(0,8, size=num_timepoint))
            w_1, w_2 = np.random.normal(0.4, 0.05), np.random.uniform(0, 7)
            curve_test_1.append(synthetic_curve_1(t, w_1, w_2, noise))       
    
        num_obs_test_2 = num_obs_train - num_obs_test_1
        curve_test_2 = []
        for i_obs in range(num_obs_test_2):
            if not irregular: t = t_const
            else: t = np.sort(np.random.uniform(0,8, size=num_timepoint))
            w_1, w_2 = np.random.uniform(0.8, 1.2), np.random.uniform(0, 7)
            curve_test_2.append(synthetic_curve_2(t, w_1, w_2, noise))
            
        curve_test = curve_test_1 + curve_test_2    
        
    
    # shuffle curve
    random.shuffle(curve_train)
    random.shuffle(curve_test)
    curve_train = np.array(curve_train)
    curve_test = np.array(curve_test)
    
    #X, Y
    inputs_train = curve_train[:,:last_obs_idx]
    targets_train = curve_train[:,last_obs_idx:]
    
    inputs_test = curve_test[:,:last_obs_idx]
    targets_test = curve_test[:,last_obs_idx:]
    
    # normalize
    mean_input = np.mean(inputs_train)
    std_input = np.std(inputs_train) 
    mean_target = np.mean(targets_train)
    std_target = np.std(targets_train)
    
    norm_train = {
            "input": (mean_input, std_input),
            "target": (mean_target, std_target)   
    }

    
    inputs_train = (inputs_train - mean_input)/std_input
    # targets_train = (targets_train - mean_target)/std_target
    inputs_test = (inputs_test - mean_input)/std_input
    # targets_test = (targets_test - mean_target)/std_target
    
    trainloader, _, _ = data_to_loader(inputs_train, targets_train, 
        batch_size=batch_size_train, shuffle=shuffle)
    
    testloader, _, _ = data_to_loader(inputs_test, targets_test, 
        batch_size=batch_size_test, shuffle=shuffle)
    
    return {
        "train": {
            "loader": trainloader,
            "inputs": inputs_train,
            "targets": targets_train,
            "normalize": norm_train
        },
        "test":{
            "loader": testloader,
            "inputs": inputs_test,
            "targets": targets_test,
            "normalize": norm_train
        }
    }


def preprocess(stream, denoising, time_window, dim_output=1, normalize=True,
               add_time_attribute=False, loss_penalty=False):
    
    eng_idx = list(set(stream[:,0]))
    
    # denosing
    if denoising:
        for i_eng, eng in enumerate(eng_idx):
            tmp = stream[stream[:,0]==eng, 1]
            num_savgol = 2*int(len(tmp)/2) - 1 if 51 > len(tmp) else 51 
            stream[stream[:,0]==eng, 1] = savgol_filter(tmp, num_savgol, 3)
            
    # normalize
    if normalize is True:
        mean_stream = np.mean(stream[:,1])
        std_stream = np.std(stream[:,1]) 
    elif normalize is False:
        mean_stream = 0
        std_stream = 1
    else:
        mean_stream = normalize["mean"]
        std_stream = normalize["std"]
    
    stream[:,1] = (stream[:,1] - mean_stream)/std_stream   
        
    # define inputs and targets
    inputs, targets = [], []
    inputs_bd, targets_bd = {}, {}
    data_original = {}
    
    # time attribute
    t_max = np.max(
        [list(stream[:,0]).count(x) for x in list(set(stream[:,0]))]
    ) - dim_output
    t_range = np.linspace(start=0, stop=1, num=t_max)
    
    for i_eng, eng in enumerate(eng_idx):
        
        tmp = stream[stream[:,0]==eng, 1]
        inputs_bd[eng], targets_bd[eng] = [], []
        data_original[eng] = tmp
        
        for i_tw in range(len(tmp) - time_window - dim_output + 1):
            
            if not add_time_attribute:
                input_i = tmp[i_tw:(i_tw + time_window)][:,None]
            else:
                input_i = np.array([
                    tmp[i_tw:(i_tw + time_window)], t_range[i_tw:(i_tw + time_window)]
                ]).T
            if not loss_penalty:
                target_i = tmp[(i_tw+time_window):(i_tw + time_window + dim_output)]
            else:
                last_obs_val = savgol_filter(input_i[:,0], time_window-1, 2)[-1]
                target_i = np.concatenate([
                    tmp[(i_tw+time_window):(i_tw + time_window + dim_output)], 
                    np.array([last_obs_val])
                ])
            
            inputs.append(input_i)
            targets.append(target_i)
            
            inputs_bd[eng].append(input_i)
            targets_bd[eng].append(target_i)
        
        inputs_bd[eng] = np.array(inputs_bd[eng])
        targets_bd[eng] = np.array(targets_bd[eng])
        
    inputs, targets = np.array(inputs), np.array(targets)
    
    return {
        "input": inputs,
        "target": targets,
        "input_bd": inputs_bd,
        "target_bd": targets_bd,
        "data_original":data_original,
        "normalize":{
            "mean": mean_stream,
            "std": std_stream
        }
    }


def get_dataset_timeseries(
        dataset, time_window, dim_output, sensor, 
        batch_size_train, batch_size_test, last_obs,
        shuffle=True, single_stream=True, denoising=False, add_time_attribute=False,
        loss_penalty=False
    ):
    
    data1_train = pd.read_csv("CMAPSSData/train_FD001.csv", sep=" ", header=None)
    data2_train = pd.read_csv("CMAPSSData/train_FD002.csv", sep=" ", header=None)
    data3_train = pd.read_csv("CMAPSSData/train_FD003.csv", sep=" ", header=None)
    data4_train = pd.read_csv("CMAPSSData/train_FD004.csv", sep=" ", header=None)
    
    data1_train = data1_train.drop(
        [data1_train.columns[26],data1_train.columns[27]], axis=1)
    data2_train = data2_train.drop(
        [data2_train.columns[26],data2_train.columns[27]], axis=1)
    data3_train = data3_train.drop(
        [data3_train.columns[26],data3_train.columns[27]], axis=1)
    data4_train = data4_train.drop(
        [data4_train.columns[26],data4_train.columns[27]], axis=1)
    
    data1_test = pd.read_csv("CMAPSSData/test_FD001.csv", sep=" ", header=None)
    data2_test = pd.read_csv("CMAPSSData/test_FD002.csv", sep=" ", header=None)
    data3_test = pd.read_csv("CMAPSSData/test_FD003.csv", sep=" ", header=None)
    data4_test = pd.read_csv("CMAPSSData/test_FD004.csv", sep=" ", header=None)
    
    data1_test = data1_test.drop(
        [data1_test.columns[26],data1_test.columns[27]], axis=1)
    data2_test = data2_test.drop(
        [data2_test.columns[26],data2_test.columns[27]], axis=1)
    data3_test = data3_test.drop(
        [data3_test.columns[26],data3_test.columns[27]], axis=1)
    data4_test = data4_test.drop(
        [data4_test.columns[26],data4_test.columns[27]], axis=1)
    
    ##################### dataset #####################
    if dataset == "FD001":
        # data_train = data1_train
        # data_test = data1_test
        remove_idx = []
        for i in list(set(data1_train.iloc[:,0])):
            if data1_train.loc[data1_train.iloc[:,0] == i,:].shape[0] < last_obs + dim_output :
                remove_idx.append(i)
        data1_train = data1_train.iloc[~np.isin(data1_train[0], remove_idx), :]
        
        train_set = np.random.choice(list(set(data1_train[0])), size=60, replace=False)
        data_train = data1_train.iloc[np.isin(data1_train[0], train_set),:] 
        data_test = data1_train.iloc[~np.isin(data1_train[0], train_set),:] 
        test_set_cluster = None
        
    elif dataset == "FD002":
        data_train = data2_train
        data_test = data2_test
    elif dataset == "FD003":
        
        remove_idx = []
        for i in list(set(data3_train.iloc[:,0])):
            if data3_train.loc[data3_train.iloc[:,0] == i,:].shape[0] < last_obs + dim_output :
                remove_idx.append(i)
        data3_train = data3_train.iloc[~np.isin(data3_train[0], remove_idx), :]
        
        idx_cluster1 = np.array(pd.read_csv("CMAPSSData/cluster_FD003.csv")).squeeze()
        idx_cluster2 = np.array(list(set(data3_train[0]) - set(idx_cluster1)))
        
        tmp_cluster1_train_idx = np.random.choice(idx_cluster1, size=6, replace=False)
        tmp_cluster2_train_idx = np.random.choice(idx_cluster2, size=0, replace=False)
        
        train_set = np.concatenate((tmp_cluster1_train_idx, tmp_cluster2_train_idx))
        test_set = np.array(list(set(data3_train[0]) - set(train_set)))
        test_set_cluster = dict()
        test_set_cluster[0] = np.array(list(set(idx_cluster1) - set(tmp_cluster1_train_idx)))
        test_set_cluster[1] = np.array(list(set(idx_cluster2) - set(tmp_cluster2_train_idx)))
        
        data_train = data3_train.iloc[np.isin(data3_train[0], train_set),:] 
        data_test = data3_train.iloc[np.isin(data3_train[0], test_set),:] 
        
    elif dataset == "FD004":
        data_train = data4_train
        data_test = data4_test
    elif dataset == "all":
        NotImplementedError("all dataset")
        
    
    stream_train = np.array(data_train.iloc[:, [0, sensor]])
    stream_test = np.array(data_test.iloc[:, [0, sensor]])

    norm_train, norm_test = {}, {}
    
    if single_stream:
        preprocessed_train = preprocess(
            stream=stream_train, denoising=denoising, dim_output=dim_output,
            time_window=time_window, normalize=True, 
            add_time_attribute=add_time_attribute, loss_penalty=loss_penalty
        )
        preprocessed_test = preprocess(
            stream=stream_test, denoising=denoising, dim_output=dim_output,
            time_window=time_window, normalize=preprocessed_train["normalize"],
            add_time_attribute=add_time_attribute, loss_penalty=loss_penalty#True
        )  
        
        inputs_train = preprocessed_train["input"]
        targets_train = preprocessed_train["target"]
        norm_train[sensor] = preprocessed_train["normalize"]
        
        inputs_test = preprocessed_test["input"]
        targets_test = preprocessed_test["target"]
        norm_test[sensor] = preprocessed_test["normalize"] 
    
    
    trainloader, _, _ = data_to_loader(inputs_train, targets_train, 
                   batch_size=batch_size_train, shuffle=shuffle)
    
    testloader, _, _ = data_to_loader(inputs_test, targets_test, 
                   batch_size=batch_size_test, shuffle=False)
    
    return {
        "train": {
            "loader": trainloader,
            "inputs": inputs_train,
            "targets": targets_train,
            "inputs_bd": preprocessed_train["input_bd"],
            "targets_bd": preprocessed_train["target_bd"],
            "data_original":preprocessed_train["data_original"],
            "normalize": norm_train
        },
        "test":{
            "loader": testloader,
            "inputs": inputs_test,
            "targets": targets_test,
            "inputs_bd": preprocessed_test["input_bd"],
            "targets_bd": preprocessed_test["target_bd"],
            "data_original":preprocessed_test["data_original"],
            "normalize": norm_test,
            "cluster": test_set_cluster
        }
    }




# args.model = "FixedLSTM"
# args.lr = 0.001
# args.verbose = False
# args.epochs = 300 #200
# args.optimizer = "Adam"
# args.local_ep = 5
# args.local_bs = 100
# args.seed = 9
# args.num_users = 10
# args.frac = 0.3
# args.last_obs = 30
# args.time_window = 20
# args.dim_output = 120
# args.criterion = 'LogGaussianLoss'
# args.loss_penalty = True
# args.init_loss_penalty = 100
# args.plot = False

# dataset='FD003'
# time_window=20
# dim_output=120
# sensor=6
# batch_size_train=100
# batch_size_test=10
# last_obs=30
# shuffle=True
# single_stream=True
# denoising=False
# add_time_attribute=True
# loss_penalty=True



# data1_train = pd.read_csv("CMAPSSData/train_FD001.csv", sep=" ", header=None)
# data2_train = pd.read_csv("CMAPSSData/train_FD002.csv", sep=" ", header=None)
# data3_train = pd.read_csv("CMAPSSData/train_FD003.csv", sep=" ", header=None)
# data4_train = pd.read_csv("CMAPSSData/train_FD004.csv", sep=" ", header=None)

# data1_train = data1_train.drop(
#     [data1_train.columns[26],data1_train.columns[27]], axis=1)
# data2_train = data2_train.drop(
#     [data2_train.columns[26],data2_train.columns[27]], axis=1)
# data3_train = data3_train.drop(
#     [data3_train.columns[26],data3_train.columns[27]], axis=1)
# data4_train = data4_train.drop(
#     [data4_train.columns[26],data4_train.columns[27]], axis=1)

# data1_test = pd.read_csv("CMAPSSData/test_FD001.csv", sep=" ", header=None)
# data2_test = pd.read_csv("CMAPSSData/test_FD002.csv", sep=" ", header=None)
# data3_test = pd.read_csv("CMAPSSData/test_FD003.csv", sep=" ", header=None)
# data4_test = pd.read_csv("CMAPSSData/test_FD004.csv", sep=" ", header=None)

# data1_test = data1_test.drop(
#     [data1_test.columns[26],data1_test.columns[27]], axis=1)
# data2_test = data2_test.drop(
#     [data2_test.columns[26],data2_test.columns[27]], axis=1)
# data3_test = data3_test.drop(
#     [data3_test.columns[26],data3_test.columns[27]], axis=1)
# data4_test = data4_test.drop(
#     [data4_test.columns[26],data4_test.columns[27]], axis=1)

# ##################### dataset #####################
# if dataset == "FD001":
#     # data_train = data1_train
#     # data_test = data1_test
#     remove_idx = []
#     for i in list(set(data1_train.iloc[:,0])):
#         if data1_train.loc[data1_train.iloc[:,0] == i,:].shape[0] < last_obs + dim_output :
#             remove_idx.append(i)
#     data1_train = data1_train.iloc[~np.isin(data1_train[0], remove_idx), :]
    
#     train_set = np.random.choice(list(set(data1_train[0])), size=60, replace=False)
#     data_train = data1_train.iloc[np.isin(data1_train[0], train_set),:] 
#     data_test = data1_train.iloc[~np.isin(data1_train[0], train_set),:] 
#     test_set_cluster = None
    
# elif dataset == "FD002":
#     data_train = data2_train
#     data_test = data2_test
# elif dataset == "FD003":
    
#     remove_idx = []
#     for i in list(set(data3_train.iloc[:,0])):
#         if data3_train.loc[data3_train.iloc[:,0] == i,:].shape[0] < last_obs + dim_output :
#             remove_idx.append(i)
#     data3_train = data3_train.iloc[~np.isin(data3_train[0], remove_idx), :]
    
#     idx_cluster1 = np.array(pd.read_csv("CMAPSSData/cluster_FD003.csv")).squeeze()
#     idx_cluster2 = np.array(list(set(data3_train[0]) - set(idx_cluster1)))
    
#     tmp_cluster1_train_idx = np.random.choice(idx_cluster1, size=25, replace=False)
#     tmp_cluster2_train_idx = np.random.choice(idx_cluster2, size=25, replace=False)
    
#     train_set = np.sort(np.concatenate((tmp_cluster1_train_idx, tmp_cluster2_train_idx)))
#     test_set = np.array(list(set(data3_train[0]) - set(train_set)))
#     test_set_cluster = dict()
#     test_set_cluster[0] = np.array(list(set(idx_cluster1) - set(tmp_cluster1_train_idx)))
#     test_set_cluster[1] = np.array(list(set(idx_cluster2) - set(tmp_cluster2_train_idx)))
    
#     data_train = data3_train.iloc[np.isin(data3_train[0], train_set),:] 
#     data_test = data3_train.iloc[np.isin(data3_train[0], test_set),:] 
    
# elif dataset == "FD004":
#     data_train = data4_train
#     data_test = data4_test
# elif dataset == "all":
#     NotImplementedError("all dataset")
    
    

# stream_train = np.array(data_train.iloc[:, [0, sensor]])
# stream_test = np.array(data_test.iloc[:, [0, sensor]])

# norm_train, norm_test = {}, {}

# if single_stream:
#     preprocessed_train = preprocess(
#         stream=stream_train, denoising=denoising, dim_output=dim_output,
#         time_window=time_window, normalize=True, 
#         add_time_attribute=add_time_attribute, loss_penalty=loss_penalty
#     )
#     preprocessed_test = preprocess(
#         stream=stream_test, denoising=denoising, dim_output=dim_output,
#         time_window=time_window, normalize=preprocessed_train["normalize"],
#         add_time_attribute=add_time_attribute, loss_penalty=loss_penalty#True
#     )  
    
#     inputs_train = preprocessed_train["input"]
#     targets_train = preprocessed_train["target"]
#     norm_train[sensor] = preprocessed_train["normalize"]
    
#     inputs_test = preprocessed_test["input"]
#     targets_test = preprocessed_test["target"]
#     norm_test[sensor] = preprocessed_test["normalize"] 


# trainloader, _, _ = data_to_loader(inputs_train, targets_train, 
#                 batch_size=batch_size_train, shuffle=shuffle)

# testloader, _, _ = data_to_loader(inputs_test, targets_test, 
#                 batch_size=batch_size_test, shuffle=False)




