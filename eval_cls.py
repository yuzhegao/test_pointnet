from __future__ import print_function,division

import os
import time
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from pointnet_cls import PointNet_cls
from data_utils import pts_cls_dataset,pts_collate

is_GPU=torch.cuda.is_available()

parser = argparse.ArgumentParser(description='pointnet_eval')

parser.add_argument('--data-eval', metavar='DIR',default='/home/gaoyuzhe/Downloads/3d_data/modelnet/test_files.txt',
                    help='txt file to validate dataset')
parser.add_argument('--shape-class', metavar='DIR',default='/home/gaoyuzhe/Downloads/3d_data/modelnet/shape_names.txt',
                    help='txt file of shape class name in dataset')
parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('--num-pts', default=1024 , type=int, metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information')

parser.add_argument('-bs',  '--batch-size', default=4 , type=int,
                    metavar='N', help='mini-batch size (default: 2)')

parser.add_argument('--resume', default='checkpoint.pth',type=str, metavar='PATH',help='path to latest checkpoint ')

args=parser.parse_args()
NUM_CLASSES=40
SHAPE_NAMES = [line.rstrip() for line in open(args.shape_class)]
print (SHAPE_NAMES)

if is_GPU:
    torch.cuda.set_device(args.gpu)

if args.normal:
    pts_featdim = 6
else:
    pts_featdim = 3

net = PointNet_cls(num_pts=args.num_pts,feat_dim=pts_featdim)
if is_GPU:
    net=net.cuda()
critenrion=nn.NLLLoss()
print (args.resume)
print (os.path.exists(args.resume))
if os.path.exists(args.resume):
    print ('ok')
    if is_GPU:
        checkoint = torch.load(args.resume)
    else:
        checkoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    start_epoch = checkoint['epoch']
    net.load = net.load_state_dict(checkoint['model'])
    num_iter = checkoint['iter']
    print('load the resume checkpoint,train from epoch{}'.format(start_epoch))
else:
    print("Warining! No resume checkpoint to load")


def evaluate(model_test):
    model_test.eval()
    total_correct=0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    data_eval = pts_cls_dataset(datalist_path=args.data_eval,data_argument=False,num_points=args.num_pts,use_extra_feature=args.normal)
    eval_loader = torch.utils.data.DataLoader(data_eval,
                    batch_size=args.batch_size, shuffle=True, collate_fn=pts_collate)
    print ("dataset size:",len(eval_loader.dataset))

    for batch_idx, (pts, label) in enumerate(eval_loader):
        if is_GPU:
            pts = Variable(pts.cuda())
            label = Variable(label.cuda())
        else:
            pts = Variable(pts)
            label = Variable(label)
        pred,trans = net(pts)

        loss = critenrion(pred, label)
        K = trans.size(1)
        reg_loss = torch.bmm(trans, trans.transpose(2, 1))
        if is_GPU:
            iden = Variable(torch.eye(K).cuda())
        else:
            iden = Variable(torch.eye(K))
        reg_loss -= iden
        reg_loss = reg_loss * reg_loss

        loss = loss + reg_loss.sum()
        loss_sum += loss.item()

        _, pred_index = torch.max(pred, dim=1)
        num_correct = (pred_index.eq(label)).data.cpu().sum().item()
        total_correct +=num_correct

        for idx,l in enumerate(label):
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_index.eq(label))[idx]

        print ('finish {}/{}'.format(batch_idx*args.batch_size,len(eval_loader.dataset)))

    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        print ('%10s:\t%0.3f' % (name, class_accuracies[i]))

    print ('eval accuracy:{}'.format(total_correct*1.0/(len(eval_loader.dataset))))
    print ('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))



evaluate(net)
