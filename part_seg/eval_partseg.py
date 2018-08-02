from __future__ import print_function,division

import os
import sys
import time
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append('../')
from pointnet_seg import PointNet_seg
from data_utils import shapenet_dataset,pts_collate_seg


is_GPU=torch.cuda.is_available()

parser = argparse.ArgumentParser(description='pointnet_partseg')
parser.add_argument('--data', metavar='DIR',default='/home/gaoyuzhe/Downloads/3d_data/hdf5_data/test_hdf5_file_list.txt',
                    help='txt file to dataset')
parser.add_argument('--data-eval', metavar='DIR',default='/home/gaoyuzhe/Downloads/3d_data/hdf5_data/test_hdf5_file_list.txt',
                    help='txt file to validate dataset')
parser.add_argument('--log', metavar='LOG',default='log',
                    help='dir of log file and resume')
parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('-bs',  '--batch-size', default=2 , type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--resume', default='pointnet_partseg.pth',
                    type=str, metavar='PATH',help='path to latest checkpoint ')

args=parser.parse_args()

LOG_DIR=args.log
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
resume=os.path.join(LOG_DIR,args.resume)
logname=os.path.join(LOG_DIR,'log.txt')

if is_GPU:
    torch.cuda.set_device(args.gpu)

net=PointNet_seg()
if is_GPU:
    net=net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

def evaluate(model_test):
    model_test.eval()
    total_correct = 0

    data_eval = shapenet_dataset(datalist_path=args.data_eval)
    eval_loader = torch.utils.data.DataLoader(data_eval,
                                  batch_size=4, shuffle=True, collate_fn=pts_collate_seg)
    print("dataset size:", len(eval_loader.dataset))

    for batch_idx, (pts, label, seg) in enumerate(eval_loader):
        ## pts [N,P,3] label [N,] seg [N,P]
        if is_GPU:
            pts = Variable(pts.cuda())
            label = Variable(label.cuda())
            seg_label = Variable(seg.cuda())
        else:
            pts = Variable(pts)
            label = Variable(label)
            seg_label = Variable(seg)

        ## pred [N,50,P]  trans [N,64,64]
        pred, trans = net(pts)

        _, pred_index = torch.max(pred, dim=1)  ##[N,P]
        num_correct = (pred_index.eq(seg_label)).data.cpu().sum()
        total_correct += num_correct

    print('the average correct rate:{}'.format(total_correct * 1.0 / (len(eval_loader.dataset)*2048)))

    model_test.train()
    with open(logname, 'a') as f:
        f.write('\nthe evaluate average accuracy:{}'.format(total_correct * 1.0 / (len(eval_loader.dataset)*2048)))
