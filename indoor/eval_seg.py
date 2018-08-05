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

from pointnet_scene import PointNet_seg
from data_indoor import indoor3d_dataset,pts_collate_seg


is_GPU=torch.cuda.is_available()

parser = argparse.ArgumentParser(description='pointnet_scene_seg')
parser.add_argument('--data', metavar='DIR',default='../../3d_data/indoor3d_sem_seg_hdf5_data/all_files.txt',
                    help='txt file to dataset')

parser.add_argument('--log', metavar='LOG',default='log',
                    help='dir of log file and resume')

parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('-bs',  '--batch-size', default=24 , type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--resume', default='pointnet_partseg.pth',
                    type=str, metavar='PATH',help='path to latest checkpoint ')

args=parser.parse_args()

NUM_CLASSES=13

LOG_DIR=args.log
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
resume=os.path.join(LOG_DIR,args.resume)

if is_GPU:
    torch.cuda.set_device(args.gpu)

net=PointNet_seg()
if is_GPU:
    net=net.cuda()

def evaluate(model_test):
    model_test.eval()
    total_correct = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    data_eval = indoor3d_dataset(datalist_path=args.data,training=False)
    eval_loader = torch.utils.data.DataLoader(data_eval,num_workers=4,
                                  batch_size=4, shuffle=True, collate_fn=pts_collate_seg)
    print("Testing dataset size:", len(eval_loader.dataset))

    for batch_idx, (pts,  seg) in enumerate(eval_loader):
        ## pts [N,P,3] label [N,] seg [N,P]
        if is_GPU:
            pts = Variable(pts.cuda())
            seg_label = Variable(seg.cuda())
        else:
            pts = Variable(pts)
            seg_label = Variable(seg)

        ## pred [N,50,P]  trans [N,64,64]
        pred, trans = net(pts)

        _, pred_index = torch.max(pred, dim=1)  ##[N,P]
        num_correct = (pred_index.eq(seg_label)).data.cpu().sum()
        total_correct += num_correct.item()

        ## calculate overall accuracy
        for j in range(4096):
            l = seg[batch_idx, j]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_index[batch_idx, j] == l)

    print('the average correct rate:{}'.format(total_correct * 1.0 /
                                               (len(eval_loader.dataset)*4096)))
    print ('the overall accuarcy:{}'.format(np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))


