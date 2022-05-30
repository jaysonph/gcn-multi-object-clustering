###################################################################
# Reference: https://github.com/Zhongdao/gcn_clustering
# A modified version of the reference above
#
# File Name: gcn.py
# Author: Jayson Ng
# Email: iamjaysonph@gmail.com
###################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class MeanAggregator(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, features, A ):
        x = torch.bmm(A, features)
        return x 

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(
                torch.FloatTensor(in_dim *2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, d = features.shape
        assert(d==self.in_dim)
        agg_feats = self.agg(features,A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out 
        

class GCN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(in_dim, affine=False)
        self.conv1 = GraphConv(in_dim, in_dim, MeanAggregator)
        self.conv2 = GraphConv(in_dim, in_dim, MeanAggregator)
        self.conv3 = GraphConv(in_dim, in_dim//2, MeanAggregator)
        self.conv4 = GraphConv(in_dim//2, in_dim//4,MeanAggregator)
        
        self.classifier = nn.Sequential(
                            nn.Linear(in_dim//4, in_dim//4),
                            nn.PReLU(in_dim//4),
                            nn.Linear(in_dim//4, 2))
    
    def forward(self, x, A, one_hop_idcs, train=True):
        # data normalization l2 -> bn
        B,N,D = x.shape
        #xnorm = x.norm(2,2,keepdim=True) + 1e-8
        #xnorm = xnorm.expand_as(x)
        #x = x.div(xnorm)
        
        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B,N,D)

        x = self.conv1(x,A)
        x = self.conv2(x,A)
        x = self.conv3(x,A)
        x = self.conv4(x,A)
        k1 = one_hop_idcs.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B,k1,dout).cuda()
        for b in range(B):
            edge_feat[b,:,:] = x[b, one_hop_idcs[b]]  
        edge_feat = edge_feat.view(-1,dout)
        pred = self.classifier(edge_feat)
            
        # shape: (B*k1)x2
        return pred