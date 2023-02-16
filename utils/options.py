#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse




def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments 
    # define new arg with adverse users with lower default learning rate
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training") #iterations
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K") #number of users to be altered for every experiment
    parser.add_argument('--frac', type=float, default=0.5, help="the fraction of clients: C") # how many clients are randomly used every iteration
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E") #local iterations
    parser.add_argument('--local_bs', type=int, default=1024, help="local batch size: B") #lower local batchsize increase convergeance rate when high DP constraint
    parser.add_argument('--bs', type=int, default=1024, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate") # 0.1 calibrated for the federated learning setting, adversary would be 0.01
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset") # choose her mnistrotated as default if wanted or adjust in simulation directly
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    parser.add_argument('--dp_mechanism', type=str, default='Gaussian',
                        help='differential privacy mechanism')
    parser.add_argument('--dp_epsilon', type=float, default=20,
                        help='differential privacy epsilon')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='differential privacy delta')
    parser.add_argument('--dp_clip', type=float, default=10,
                        help='differential privacy clip')

    args = parser.parse_args()
    return args
