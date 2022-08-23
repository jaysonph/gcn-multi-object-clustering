###################################################################
# File Name: train.py
# Author: Jayson Ng
# Email: iamjaysonph@gmail.com
###################################################################

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model
from feeder import Feeder
from utils.logging import Logger 

from sklearn.metrics import precision_score, recall_score


# Reference: https://github.com/Zhongdao/gcn_clustering
def adjust_lr(opt, epoch):
    scale = 0.1
    print('Current lr {}'.format(args.lr))
    if epoch in [1,2,3,4]:
        args.lr *=0.1
        print('Change lr to {}'.format(args.lr))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * scale

# Reference: https://github.com/Zhongdao/gcn_clustering
def accuracy(pred, label, masks):
    pred = (torch.argmax(pred, dim=1) * masks).long()
    acc = torch.mean((pred == label).float())
    pred = pred.numpy()
    label = label.numpy()
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p,r,acc 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--feat_dim', default=512, type=int, required=True)
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default='./logs')
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--print_intv', default=200, type=int)

    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=1)   # currently only support batch size = 1
    parser.add_argument('--feat_path', type=str, metavar='PATH',
                        default='./features/l2_512_k8/train/feat.json')
    parser.add_argument('--knn_graph_path', type=str, metavar='PATH',
                        default='./features/l2_512_k8/train/knn_graph.json')
    parser.add_argument('--label_path', type=str, metavar='PATH',
                        default='./features/l2_512_k8/train/label.json')
    parser.add_argument('--obj_type_path', type=str, metavar='PATH',
                        default='./features/l2_512_k8/train/obj_type.json')
    parser.add_argument('--k-at-hop', type=int, nargs='+', default=[8,5])
    parser.add_argument('--active_connection', type=int, default=5)

    args = parser.parse_args()

    sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = Feeder(args.feat_path, 
                      args.knn_graph_path, 
                      args.label_path,
                      args.obj_type_path, 
                      args.seed, 
                      args.k_at_hop,
                      args.active_connection)

    trainloader = DataLoader(
            trainset, batch_size=args.batch_size,
            num_workers=args.n_workers, shuffle=True, pin_memory=True, collate_fn=trainset.collate_fn) 

    model = model.gcn(in_dim=args.feat_dim).to(device)
    optimizer = torch.optim.SGD(net.parameters(), args.lr, 
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay) 

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        adjust_lr(opt, epoch)
        train_loss = 0
        train_acc = 0
        train_prec = 0
        train_recall = 0
        model.train()
        for i, (feat, adj, cid, h1id, gtmat, obj_mask) in enumerate(loader):
            feat, adj, cid, h1id, gtmat, obj_mask = map(lambda x: x.to(device), 
                                    (feat, adj, cid, h1id, gtmat, obj_mask))
            # Forward
            pred = model(feat, adj, h1id)
            labels = gtmat.view(-1).long()
            masks = obj_mask.view(-1).long()
            loss = criterion(pred, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            p,r, acc = accuracy(pred, labels, masks)
            train_loss += loss.item()
            train_acc += acc.item()
            train_prec += p
            train_recall += r
        
            if i % args.print_intv == 0:
                print(f'''
                    Epoch {epoch}/{args.epochs}: Avg Tr Loss = {(train_loss/epoch+1):.2f} | Avg Tr Acc = {(train_acc/epoch+1):.2f} 
                    | Avg Tr Prec = {(train_prec/epoch+1):.2f} | Avg Tr Recall = {(train_recall/epoch+1):.2f}
                ''')

        torch.save(net.state_dict(), os.path.join(args.logs_Dir, f'epoch_{epoch}.pth'))
