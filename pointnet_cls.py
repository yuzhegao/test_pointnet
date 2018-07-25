import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


################
# transform net
################
class transform_net(nn.Module):
    def __init__(self, featdim=3, num_pts=1024):
        super(transform_net, self).__init__()
        self.num_pts = num_pts
        self.featdim = featdim
        self.conv1 = nn.Sequential(
            nn.Conv1d(featdim, 64, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(64, 128, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv3=nn.Sequential(
            nn.Conv1d(128, 1024, 1, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc2=nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc3=nn.Sequential(
            nn.Linear(256, (self.featdim) ** 2, bias=False),
            nn.BatchNorm1d((self.featdim) ** 2),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool1d(kernel_size=self.num_pts)

        ## transform matrix
        self.trans_weight = Variable()

    def forward(self, x):
        batchsize = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  ##[N,1024,P]

        x = torch.squeeze(self.max_pool(x),dim=2)  ##[N,1024]


        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)  ##[N,K*K]
        iden = Variable(torch.from_numpy(np.eye(self.featdim).astype(np.float32))).view(1,(self.featdim) ** 2).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.featdim, self.featdim)
        return x


class PointNet_cls(nn.Module):
    def __init__(self, num_class=40,num_pts=1024):
        super(PointNet_cls, self).__init__()
        self.num_cls = num_class
        self.num_pts=num_pts
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(64, 64, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv4=nn.Sequential(
            nn.Conv1d(64, 128, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv5=nn.Sequential(
            nn.Conv1d(128, 1024, 1, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.input_transorm_net = transform_net(featdim=3,num_pts=num_pts)
        self.feature_transorm_net = transform_net(featdim=64,num_pts=num_pts)
        self.max_pool = nn.MaxPool1d(kernel_size=self.num_pts)

        self.classifer=nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,self.num_cls)
        )

    def forward(self, x):  ##[N,3,1024]
        trans = self.input_transorm_net(x)  ##[N,3,3]
        x = x.transpose(2, 1)  ##[N,P,3]
        x = torch.bmm(x, trans)  ##[N,P,3]
        x = x.transpose(2, 1)  ##[N,3,P]
        x = self.conv1(x)
        x = self.conv2(x)  ##[N,64,P]

        trans2=self.feature_transorm_net(x)
        x = x.transpose(2, 1)  ##[N,P,64]
        x = torch.bmm(x, trans2)  ##[N,P,64]
        x = x.transpose(2, 1)  ##[N,64,P]

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  ##[N,1024,P]

        feature_vector=torch.squeeze(self.max_pool(x)) ##[N,1024]
        scores=self.classifer(feature_vector) ##[N,40]
        pred=F.log_softmax(scores, dim=-1)

        return pred,trans2



"""
from data_utils import pts_cls_dataset,pts_collate
my_dataset=pts_cls_dataset(datalist_path='/home/gaoyuzhe/Downloads/3d_data/modelnet/test_files.txt',num_points=2048)
data_loader = torch.utils.data.DataLoader(my_dataset,
            batch_size=2, shuffle=True, collate_fn=pts_collate)

net=PointNet_cls()
for batch_idx, (pts, label) in enumerate(data_loader):
    if False:
        pts = Variable(pts.cuda())
        label = Variable(label.cuda())
    else:
        pts = Variable(pts)
        label = Variable(label)
    pred = net(pts)
    exit()
"""