#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import mnist_iid, mnist_noniid, mnistrotated_iid, mnistrotated_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist
from models.Fed import FedAvg
from models.test import test_img
from opacus.grad_sample import GradSampleModule

if __name__ == '__main__':
    # parse args

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist': # load MNIST dataset and split users
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'mnistrotated': # load MNIST rotated dataset and split users
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnistrotated/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnistrotated/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        # sample users 
        if args.iid:
            dict_users = mnistrotated_iid(dataset_train, args.num_users)
        else:
            dict_users = mnistrotated_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'mnistrotated'):
        net_glob = CNNMnist(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')

    dp_epsilon = args.dp_epsilon / (args.frac * args.epochs)
    dp_delta = args.dp_delta
    dp_mechanism = args.dp_mechanism
    dp_clip = args.dp_clip

    # use opacus to wrap model to clip per sample gradient
    net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    all_clients = list(range(args.num_users))


    # training
    acc_test = []
    # learning_rate = [args.lr for i in range(args.num_users)] #default learning rate per user
    learning_rate = [0.1, 0.1, 0.1, 0.1, 0.1] # individual learning rates per user, change to args.lr to use default value for each user
    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        begin_index = iter % (1 / args.frac)
        idxs_users = all_clients[int(begin_index * args.num_users * args.frac):
                                   int((begin_index + 1) * args.num_users * args.frac)]
        for idx in idxs_users:  # in the training loop, use the appropriate learning rate for each user
            args.lr = learning_rate[idx]
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],
                                dp_epsilon=dp_epsilon, dp_delta=dp_delta,
                                dp_mechanism=dp_mechanism, dp_clip=dp_clip)
            w, loss, curLR = local.train(net=copy.deepcopy(net_glob).to(args.device))
            learning_rate[idx] = curLR
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))


        # update global weights
        w_glob = FedAvg(w_locals)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        print("Round {:3d},Testing accuracy: {:.2f}".format(iter, acc_t))

        acc_test.append(acc_t.item())

    rootpath = './log'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    accfile = open(rootpath + '/accfile_fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}.dat'.
                   format(args.dataset, args.model, args.epochs, args.iid,
                          args.dp_mechanism, args.dp_epsilon), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('Test accuracy')
    plt.xlabel('Iterations')
    plt.savefig(rootpath + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_acc.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_epsilon))

