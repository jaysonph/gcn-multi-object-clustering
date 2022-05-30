###################################################################
# Reference: https://github.com/Zhongdao/gcn_clustering
# A modified version of the reference above
#
# File Name: feeder.py
# Author: Jayson Ng
# Email: iamjaysonph@gmail.com
###################################################################

import numpy as np
import random
import json
import torch
import torch.utils.data as data
class Feeder(data.Dataset):
    '''
    Generate a sub-graph from the feature graph centered at some node, 
    and now the sub-graph has a fixed depth, i.e. 2
    '''
    def __init__(self, feat_path, knn_graph_path, label_path, obj_type_path, seed=1, 
                 k_at_hop=[200,5], active_connection=5, train=True):
        '''
        active_connection: parameter u for selecting top uNN
        '''
        np.random.seed(seed)
        random.seed(seed)
        self.features = self.read_json(feat_path)
        self.knn_graph = [[node_neighs[:k_at_hop[0]+1] for node_neighs in frame_g] for frame_g in self.read_json(knn_graph_path)]
        self.labels = self.read_json(label_path)
        self.obj_types = self.read_json(obj_type_path)
        self.num_samples = len(self.features)
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop
        self.active_connection = active_connection
        self.train = train
        assert np.mean(k_at_hop)>=active_connection

    def __len__(self):
        return self.num_samples

    def read_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, index):
        '''
        Only support single graph training at each iteration (batch size = 1) due to imcompatible shapes of different graphs.

        Args:
        - index: int
            indicating which graph to use
        '''
        frame_features = np.array(self.features[index])
        frame_graph = np.array(self.knn_graph[index])
        frame_labels = np.array(self.labels[index])
        frame_objs = np.array(self.obj_types[index])

        if self.train:
            b_feat, b_A_, b_center_idx, b_one_hop_idcs, b_edge_labels, b_obj_masks = [], [], [], [], [], []
            for node_i in range(frame_features.shape[0]):
                feat, A_, center_idx, one_hop_idcs, edge_labels, obj_mask = self.get_ips_of_node_i(node_i, frame_features, frame_graph, frame_labels, frame_objs)
                b_feat.append(feat)
                b_A_.append(A_)
                b_center_idx.append(center_idx)
                b_one_hop_idcs.append(one_hop_idcs)
                b_edge_labels.append(edge_labels)
                b_obj_masks.append(obj_mask)

            b_feat = torch.stack(b_feat)
            b_A_ = torch.stack(b_A_)
            b_center_idx = torch.stack(b_center_idx)
            b_one_hop_idcs = torch.stack(b_one_hop_idcs)
            b_edge_labels = torch.stack(b_edge_labels)
            b_obj_masks = torch.stack(b_obj_masks)
            return b_feat, b_A_, b_center_idx, b_one_hop_idcs, b_edge_labels, b_obj_masks

        else:
            b_feat, b_A_, b_center_idx, b_one_hop_idcs, b_unique_nodes_list, b_edge_labels, b_obj_masks = [], [], [], [], [], [], []
            for node_i in range(frame_features.shape[0]):
                feat, A_, center_idx, one_hop_idcs, unique_nodes_list, edge_labels, obj_mask = self.get_ips_of_node_i(node_i, frame_features, frame_graph, frame_labels, frame_objs)
                b_feat.append(feat)
                b_A_.append(A_)
                b_center_idx.append(center_idx)
                b_one_hop_idcs.append(one_hop_idcs)
                b_unique_nodes_list.append(unique_nodes_list)
                b_edge_labels.append(edge_labels)
                b_obj_masks.append(obj_mask)

            b_feat = torch.stack(b_feat)
            b_A_ = torch.stack(b_A_)
            b_center_idx = torch.stack(b_center_idx)
            b_one_hop_idcs = torch.stack(b_one_hop_idcs)
            b_unique_nodes_list = torch.stack(b_unique_nodes_list)
            b_edge_labels = torch.stack(b_edge_labels)
            b_obj_masks = torch.stack(b_obj_masks)
            return b_feat, b_A_, b_center_idx, b_one_hop_idcs, b_unique_nodes_list, b_edge_labels, b_obj_masks

    def get_ips_of_node_i(self, node_i: int, frame_features: np.array, frame_graph: np.array, frame_labels: np.array, frame_objs: np.array):
        '''
        return the vertex feature and the adjacent matrix A, together 
        with the indices of the center node and its 1-hop nodes
        '''
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        hops = list()
        center_node = node_i
        hops.append(set(frame_graph[center_node][1:]))

        # Actually we dont need the loop since the depth is fixed here,
        # But we still remain the code for further revision
        for d in range(1,self.depth): 
            hops.append(set())
            for h in hops[-2]:
                hops[-1].update(set(frame_graph[h][1:self.k_at_hop[d]+1]))

        
        hops_set = set([h for hop in hops for h in hop])
        hops_set.update([center_node,])  # center node included in IPS???
        unique_nodes_list = list(hops_set) 
        unique_nodes_map = {j:i for i,j in enumerate(unique_nodes_list)}

        center_idx = torch.Tensor([unique_nodes_map[center_node],]).type(torch.long)
        one_hop_idcs = torch.Tensor([unique_nodes_map[i] for i in hops[0]]).type(torch.long)
        center_feat = torch.Tensor(frame_features[center_node]).type(torch.float)
        feat = torch.Tensor(frame_features[unique_nodes_list]).type(torch.float)
        feat = feat - center_feat
        
        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        num_nodes = len(unique_nodes_list)
        A = torch.zeros(num_nodes, num_nodes)

        _, fdim = feat.shape
        feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, fdim)], dim=0)  # [max_num_nodes, fdim]
      
        # Create Adjacency Matrix
        for node in unique_nodes_list:
            neighbors = frame_graph[node, 1:self.active_connection+1]
            for n in neighbors:
                if n in unique_nodes_list: 
                    A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    A[unique_nodes_map[n], unique_nodes_map[node]] = 1

        D = A.sum(1, keepdim=True)
        A = A.div(D)
        A_ = torch.zeros(max_num_nodes,max_num_nodes)
        A_[:num_nodes,:num_nodes] = A  # [max_num_nodes, max_num_nodes]

        
        labels = frame_labels[np.asarray(unique_nodes_list)]
        labels = torch.from_numpy(labels).type(torch.long)
        #edge_labels = labels.expand(num_nodes,num_nodes).eq(
        #        labels.expand(num_nodes,num_nodes).t())
        one_hop_labels = labels[one_hop_idcs]
        center_label = labels[center_idx]
        edge_labels = (center_label == one_hop_labels).long()
        
        obj_types = frame_objs[np.array(unique_nodes_list)]
        obj_types = torch.from_numpy(obj_types).type(torch.long)
        one_hop_obj_types = obj_types[one_hop_idcs]
        center_obj_type = obj_types[center_idx]
        obj_mask = (center_obj_type != one_hop_obj_types).long()
        
        
        if self.train:
            return feat, A_, center_idx, one_hop_idcs, edge_labels, obj_mask
        # Testing
        unique_nodes_list = torch.Tensor(unique_nodes_list)
        unique_nodes_list = torch.cat(
                [unique_nodes_list, torch.zeros(max_num_nodes-num_nodes)], dim=0)
        return feat, A_, center_idx, one_hop_idcs, unique_nodes_list, edge_labels, obj_mask
    
    def collate_fn(self, data):
        data = list(map(lambda x: torch.cat(x, dim=0), zip(*data)))
        return data