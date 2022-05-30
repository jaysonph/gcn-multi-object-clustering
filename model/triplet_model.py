###################################################################
# File Name: triplet_model.py
# Author: Jayson Ng
# Email: iamjaysonph@gmail.com
###################################################################

import torch
import torch.nn as nn

# Version 1 (best)
class FeatureAggregator(nn.Module):
    def __init__(self, img_encoder, feat_dim=512):
        super().__init__()
        self.img_encoder = img_encoder
        self.img_encoder.eval()
        self.img_head = self.create_hidden_layer(1000, feat_dim)
        self.bbox_head = self.create_hidden_layer(4, feat_dim)
        self.agg_head = self.create_hidden_layer(feat_dim, feat_dim*2)
        self.final_layer = nn.Linear(feat_dim*2, feat_dim)
    
    def forward(self, img, bbox):
        '''
        img: (bs, 3, h, w)
        bbox: (bs, 4)
        '''
        img_feat = self.img_encoder(img)
        img_feat = self.img_head(img_feat)
        bbox_feat = self.bbox_head(bbox)
        feat = img_feat + bbox_feat
        feat = self.agg_head(feat)
        feat = self.final_layer(feat)
        return feat
    
    def create_hidden_layer(self, in_dim, out_dim):
        layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.01)
        )
        return layer

# # Version 2
# class FeatureAggregator(nn.Module):
#     def __init__(self, img_encoder, feat_dim):
#         super().__init__()
#         self.img_encoder = img_encoder
#         self.img_encoder.eval()
#         self.img_head = self.create_hidden_layer(1000, feat_dim)
#         self.bbox_head = self.create_hidden_layer(4, feat_dim)
#         self.agg_head = self.create_hidden_layer(feat_dim, feat_dim)
    
#     def forward(self, img, bbox):
#         '''
#         img: (bs, 3, h, w)
#         bbox: (bs, 4)
#         '''
#         img_feat = self.img_encoder(img)
#         img_feat = self.img_head(img_feat)
#         bbox_feat = self.bbox_head(bbox)
# #         feat = torch.cat([img_feat, bbox_feat], dim=-1)
#         feat = img_feat + bbox_feat
#         feat = self.agg_head(feat)
#         return feat
    
#     def create_hidden_layer(self, in_dim, out_dim):
#         layer = nn.Sequential(
#             nn.Linear(in_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.01),
#             nn.Linear(256, out_dim)
#         )
#         return layer

# # Version 3
# class FeatureAggregator(nn.Module):
#     def __init__(self, img_encoder, feat_dim):
#         super().__init__()
#         self.img_encoder = img_encoder
#         self.img_encoder.eval()
        
#         self.jk_img_head = self.create_hidden_layer(1000, feat_dim)
#         self.jk_bbox_head = self.create_hidden_layer(4, feat_dim)
#         self.jk_agg_head = self.create_hidden_layer(feat_dim*2, feat_dim)
        
#         self.sdcl_img_head = self.create_hidden_layer(1000, feat_dim)
#         self.sdcl_bbox_head = self.create_hidden_layer(4, feat_dim)
#         self.sdcl_agg_head = self.create_hidden_layer(feat_dim*2, feat_dim)
        
#         self.cap_img_head = self.create_hidden_layer(1000, feat_dim)
#         self.cap_bbox_head = self.create_hidden_layer(4, feat_dim)
#         self.cap_agg_head = self.create_hidden_layer(feat_dim*2, feat_dim)
    
#     def forward(self, img, bbox, obj_id):
#         '''
#         img: (bs, 3, h, w)
#         bbox: (bs, 4)
#         '''
#         img_feat = self.img_encoder(img)
#         if obj_id == 0:
#             img_feat = self.jk_img_head(img_feat)
#             bbox_feat = self.jk_bbox_head(bbox)
#             feat = torch.cat([img_feat, bbox_feat], dim=-1)
#             feat = self.jk_agg_head(feat)
#         if obj_id == 1:
#             img_feat = self.sdcl_img_head(img_feat)
#             bbox_feat = self.sdcl_bbox_head(bbox)
#             feat = torch.cat([img_feat, bbox_feat], dim=-1)
#             feat = self.sdcl_agg_head(feat)
#         if obj_id == 2:
#             img_feat = self.cap_img_head(img_feat)
#             bbox_feat = self.cap_bbox_head(bbox)
#             feat = torch.cat([img_feat, bbox_feat], dim=-1)
#             feat = self.cap_agg_head(feat)
            
#         return feat
    
    
#     def create_hidden_layer(self, in_dim, out_dim):
#         layer = nn.Sequential(
#             nn.Linear(in_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.01),
#             nn.Linear(256, out_dim)
#         )
#         return layer