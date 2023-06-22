#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser(option='federated', realdata='NASA'):
    parser = argparse.ArgumentParser()
    
    if option == "federated":

        # federated arguments (Notation for the arguments followed from paper)
        parser.add_argument('--epochs', type=int, default=10,
                            help="number of rounds of training")
        parser.add_argument('--num_users', type=int, default=100,
                            help="number of users: K")
        # parser.add_argument('--frac', type=float, default=0.1,
        #                     help='the fraction of clients: C')
        parser.add_argument('--local_ep', type=int, default=10,
                            help="the number of local epochs: E")
        parser.add_argument('--local_bs', type=int, default=10,
                            help="local batch size: B")
        parser.add_argument('--lr', type=float, default=0.01,
                            help='learning rate')
        parser.add_argument('--momentum', type=float, default=0.5,
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--lr_schedule', type=str, default='const',
                            help='learning rate schedule')
        parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
        parser.add_argument('--loss_penalty', type=bool, default=False,
                            help='loss penalty for time series prediction')
        parser.add_argument('--init_loss_penalty', type=float, default=0,
                            help='init loss penalty')
        parser.add_argument('--criterion', type=str, default='LogGaussianLoss',
                            help='evaluation criterion')
        
    
        # entropy arguments
        parser.add_argument('--L', type=int, default=5,
                            help='Langevin iterations')
        parser.add_argument('--gamma', type=float, default=1e-4,
                            help='gamma')
        parser.add_argument('--scoping', type=float, default=0,
                            help='scoping')
        parser.add_argument('--eta', type=float, default=1,
                            help='step size at the global model')
        parser.add_argument('--l2', type=float, default=0,
                            help='weight decay (l2 norm penalty)')
        parser.add_argument('--nesterov', type=bool, default=False,
                            help='nesterov acceleration')
        
        # model arguments
        parser.add_argument('--model', type=str, default='mlp', help='model name')
        parser.add_argument('--kernel_num', type=int, default=9,
                            help='number of each kind of kernel')
        parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                            help='comma-separated kernel size to \
                            use for convolution')
        parser.add_argument('--num_channels', type=int, default=1, help="number \
                            of channels of imgs")
        parser.add_argument('--norm', type=str, default='batch_norm',
                            help="batch_norm, layer_norm, or None")
        parser.add_argument('--num_filters', type=int, default=32,
                            help="number of filters for conv nets -- 32 for \
                            mini-imagenet, 64 for omiglot.")
        parser.add_argument('--max_pool', type=str, default='True',
                            help="Whether use max pooling rather than \
                            strided convolutions")
                            
        # NASA arguments
        if realdata == 'NASA':
            parser.add_argument('--denoise', type=bool, default=False,
                                help="denosing NASA signal")
            parser.add_argument('--single_stream', type=bool, default=True,
                                help='single stream')
            parser.add_argument('--dataset_NASA', type=str, default="FD001",
                                help=' ')
            parser.add_argument('--time_window', type=int, default=20,
                                help='')
            parser.add_argument('--dim_output', type=int, default=100,
                                help="Dimension of outputs")
            parser.add_argument('--last_obs', type=int, default=30,
                                help='Index of the last observation')
            parser.add_argument('--sensor', type=int, default=None,
                                help='The sensor of interest')
    
        # other arguments
        parser.add_argument('--dir', type=str, default='/home/seokhc',
                            help='working directory')
        parser.add_argument('--save_path', type=str, default='results',
                            help='save location')   
        parser.add_argument('--dataset', type=str, default='MNIST', help="name \
                            of dataset")
        parser.add_argument('--num_classes', type=int, default=10, help="number \
                            of classes")
        parser.add_argument('--gpu', default=None, help="To use cuda, set \
                            to a specific GPU ID. Default set to use CPU.")
        parser.add_argument('--iid', type=int, default=1,
                            help='Default set to IID. Set to 0 for non-IID.')
        parser.add_argument('--unequal', type=int, default=0,
                            help='whether to use unequal data splits for  \
                            non-i.i.d setting (use 0 for equal splits)')
        parser.add_argument('--stopping_rounds', type=int, default=10,
                            help='rounds of early stopping')
        parser.add_argument('--plot', type=bool, default=True,
                            help='print plot')
        parser.add_argument('--verbose', type=bool, default=False, help='verbose')
        parser.add_argument('--seed', type=int, default=1, help='random seed')
        #args = parser.parse_args()
        args, unknown = parser.parse_known_args()
        
        return args
    
    elif option == "centralized":
        
        # General arguments
        parser.add_argument('--dir', type=str, default='/home/seokhc',
                            help='working directory')
        parser.add_argument('--seed', type=int, default=1, 
                            help='random seed')
        parser.add_argument('--verbose', type=bool, default=True, 
                            help='print intermediate results')
        parser.add_argument('--dataset', type=str, default='MNIST', 
                            help="name of dataset")
        parser.add_argument('--data_path', type=str, default='data/',
                            help='dataset location')
        parser.add_argument('--save_path', type=str, default='results',
                            help='save location')
        parser.add_argument('--save_file_name', type=str, default='save',
                            help='save file name')
        parser.add_argument('--model', type=str, default='LeNet5',
                            help='specify a model to train')
        
        # Dataset arguments
        parser.add_argument('--batch_size_train', type=int, default=32,
                            help='batch size of train dataset')
        parser.add_argument('--batch_size_test', type=int, default=1000,
                            help='batch size of test dataset')
        parser.add_argument('--n_classes', type=int, default=10,
                            help='number of classes')
        parser.add_argument('--img_size', type=int, default=32,
                            help='size of images')
        
        
        # optimiztion arguments
        ## general
        parser.add_argument('--lr', type=float, default=0.01, 
                            help='initial learning rate')
        parser.add_argument('--epochs', type=int, default=10,
                            help="number of rounds of training")
        parser.add_argument('--momentum', type=float, default=0,
                            help='momentum')
        parser.add_argument('--l2', type=float, default=0,
                            help='weight decay (l2 norm penalty)')
        parser.add_argument('--nesterov', type=bool, default=False,
                            help='nesterov acceleration')
        parser.add_argument('--factor', type=float, default=3,
                            help='factor of learning rate decay')
        ## Entropy SGD
        parser.add_argument('--L', type=int, default=5,
                            help='Langevin iterations')
        parser.add_argument('--gamma', type=float, default=1e-4,
                            help='gamma')
        parser.add_argument('--scoping', type=float, default=1e-3,
                            help='scoping')
        
        args = parser.parse_args()
        return args
        
        
        
        

    