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


######################
## PointNet_Seg
######################

class PointNet_seg(nn.Module):
    def __init__(self, num_class=16,num_pts=2048,num_seg=50):
        super(PointNet_seg, self).__init__()
        self.num_cls = num_class
        self.num_pts = num_pts
        self.num_seg = num_seg

        self.conv1 = nn.Sequential(
            nn.Conv1d(9, 64, 1, 1),
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

        self.feature_transorm_net = transform_net(featdim=64,num_pts=num_pts)
        self.max_pool = nn.MaxPool1d(kernel_size=self.num_pts)

        self.classifer = nn.Sequential(
            nn.Conv1d(1088, 512, 1, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, self.num_seg, 1, 1),
        )

    def forward(self, x):  ##[N,3,1024]
        x = self.conv1(x)
        pts_feature = self.conv2(x)  ##[N,64,P]

        trans2=self.feature_transorm_net(pts_feature)
        pts_feature = pts_feature.transpose(2, 1)  ##[N,P,64]
        pts_feature_align = torch.bmm(pts_feature, trans2)  ##[N,P,64]
        pts_feature_align = pts_feature_align.transpose(2, 1)  ##[N,64,P]

        x = self.conv3(pts_feature_align)
        x = self.conv4(x)
        x = self.conv5(x)  ##[N,1024,P]

        global_feature = torch.squeeze(self.max_pool(x))  ##[N,1024]
        global_feature = global_feature.view(-1, 1024, 1).repeat(1, 1, self.num_pts)

        pts_seg_feature = torch.cat([global_feature, pts_feature_align], dim=1) ##[N,1088,P]
        scores = self.classifer(pts_seg_feature)  ##[N,50,P]
        pred = F.log_softmax(scores, dim=1)

        return pred,trans2


if __name__ == '__main__':
    from data_indoor import indoor3d_dataset,pts_collate_seg

    my_dataset = indoor3d_dataset('../../3d_data/indoor3d_sem_seg_hdf5_data/all_files.txt')
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                batch_size=2, shuffle=True, collate_fn=pts_collate_seg)

    net = PointNet_seg()
    for batch_idx, (pts, label, seg) in enumerate(data_loader):
        if False:
            pts = Variable(pts.cuda())
            label = Variable(label.cuda())
        else:
            pts = Variable(pts)
            label = Variable(label)
        pred,trans = net(pts)
        print (pred.size())
        print (trans.size())
        exit()