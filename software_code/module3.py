import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer2 import *
from layer3 import *
from utils import *
from egnn_pytorch import EGNN

class EGNN_noedge2(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_noedge2, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim)
        self.egnn2 = EGNN(dim=hidden_dim)
        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords):
        feat1 = self.init_feat1(feat, coords)
        feat1 = self.egnn1(feat1, coords)
        feat1 = self.egnn2(feat1, coords)
        feat_out = average_docked_dim(feat1)
        feat_out = self.fc(feat_out)

        return feat_out
    
class EGNN_edge2(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_noedge2, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim)
        self.egnn2 = EGNN(dim=hidden_dim)
        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        feat1 = self.init_feat1(feat, coords, simi)
        feat1 = self.egnn1(feat1, coords, simi)
        feat1 = self.egnn2(feat1, coords, simi)
        feat_out = average_docked_dim(feat1)
        feat_out = self.fc(feat_out)

        return feat_out
    
class EGNN_noedge3(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_noedge3, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim)
        self.egnn2 = EGNN(dim=hidden_dim)
        self.egnn3 = EGNN(dim=hidden_dim)
        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords):
        feat1 = self.init_feat1(feat)
        feat1 = self.egnn1(feat1, coords)
        feat1 = self.egnn2(feat1, coords)
        feat1 = self.egnn3(feat1, coords)
        feat_out = average_docked_dim(feat1)
        feat_out = self.fc(feat_out)

        return feat_out
    
class EGNN_noedge3_mul_sf1_sf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_noedge3_mul_sf1_sf, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim)
        self.egnn2 = EGNN(dim=hidden_dim)
        self.egnn3 = EGNN(dim=hidden_dim)

        self.sf1 = Selfattention_single_sf(hidden_dim, hidden_dim, 1)
        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords):
        feat1 = self.init_feat1(feat)
        feat2 = self.init_feat2(feat)

        feat1, coords = self.egnn1(feat1, coords)
        feat1, coords = self.egnn2(feat1, coords)
        feat1, coords = self.egnn3(feat1, coords)

        feat2 = self.sf1(feat2) # shape as (batch, dock, 1)

        feat_out = torch.mul(feat1, feat2) # shape as (batch, dock, hidden_dim)
        feat_out = average_docked_dim(feat_out) # shape as (batch, hidden_dim)
        feat_out = self.fc(feat_out)

        return feat_out

class EGNN_noedge2_mul_sf1_sf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_noedge2_mul_sf1_sf, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim)
        self.egnn2 = EGNN(dim=hidden_dim)

        self.sf1 = Selfattention_single_sf(hidden_dim, hidden_dim, 1)
        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords):
        feat1 = self.init_feat1(feat)
        feat2 = self.init_feat2(feat)

        feat1, coords = self.egnn1(feat1, coords)
        feat1, coords = self.egnn2(feat1, coords)

        feat2 = self.sf1(feat2) # shape as (batch, dock, 1)

        feat_out = torch.mul(feat1, feat2) # shape as (batch, dock, hidden_dim)
        feat_out = average_docked_dim(feat_out) # shape as (batch, hidden_dim)
        feat_out = self.fc(feat_out)

        return feat_out
    
class EGNN_edge2_mul_sf1_sf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_edge2_mul_sf1_sf, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim, edge_dim=1)
        self.egnn2 = EGNN(dim=hidden_dim, edge_dim=1)

        self.sf1 = Selfattention_single_sf(hidden_dim, hidden_dim, 1)
        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        simi = simi.unsqueeze(3)
        feat1 = self.init_feat1(feat)
        feat2 = self.init_feat2(feat)

        feat1, coords = self.egnn1(feat1, coords, simi)
        feat1, coords = self.egnn2(feat1, coords, simi)

        feat2 = self.sf1(feat2)

        feat_out = torch.mul(feat1, feat2)
        feat_out = average_docked_dim(feat_out)
        feat_out = self.fc(feat_out)

        return feat_out
    
class EGNN_edge3_mul_sf1_sf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_edge3_mul_sf1_sf, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim, edge_dim=1)
        self.egnn2 = EGNN(dim=hidden_dim, edge_dim=1)
        self.egnn3 = EGNN(dim=hidden_dim, edge_dim=1)

        self.sf1 = Selfattention_single_sf(hidden_dim, hidden_dim, 1)
        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        simi = simi.unsqueeze(3)
        feat1 = self.init_feat1(feat)
        feat2 = self.init_feat2(feat)

        feat1, coords = self.egnn1(feat1, coords, simi)
        feat1, coords = self.egnn2(feat1, coords, simi)
        feat1, coords = self.egnn3(feat1, coords, simi)

        feat2 = self.sf1(feat2)

        feat_out = torch.mul(feat1, feat2)
        feat_out = average_docked_dim(feat_out)
        feat_out = self.fc(feat_out)

        return feat_out

class EGNN_noedge3_cat_sf1(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_noedge3_cat_sf1, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim)
        self.egnn2 = EGNN(dim=hidden_dim)
        self.egnn3 = EGNN(dim=hidden_dim)

        self.sf1 = Selfattention_single(hidden_dim, hidden_dim, hidden_dim)
        self.fc = FC(hidden_dim * 2, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords):
        feat1 = self.init_feat1(feat)
        feat2 = self.init_feat2(feat)

        feat1, coords = self.egnn1(feat1, coords)
        feat1, coords = self.egnn2(feat1, coords)
        feat1, coords = self.egnn3(feat1, coords)

        feat2 = self.sf1(feat2) # shape as (batch, dock, 1)

        feat_out = torch.cat([feat1, feat2], dim=-1) # shape as (batch, dock, hidden_dim)
        feat_out = average_docked_dim(feat_out) # shape as (batch, hidden_dim)
        feat_out = self.fc(feat_out)

        return feat_out

class EGNN_noedge2_cat_sf1(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_noedge2_cat_sf1, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim)
        self.egnn2 = EGNN(dim=hidden_dim)

        self.sf1 = Selfattention_single(hidden_dim, hidden_dim, hidden_dim)
        self.fc = FC(hidden_dim * 2, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords):
        feat1 = self.init_feat1(feat)
        feat2 = self.init_feat2(feat)

        feat1, coords = self.egnn1(feat1, coords)
        feat1, coords = self.egnn2(feat1, coords)

        feat2 = self.sf1(feat2) # shape as (batch, dock, 1)

        feat_out = torch.cat([feat1, feat2], dim=-1) # shape as (batch, dock, hidden_dim)
        feat_out = average_docked_dim(feat_out) # shape as (batch, hidden_dim)
        feat_out = self.fc(feat_out)

        return feat_out

class EGNN_edge3_cat_sf1(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_edge3_cat_sf1, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim, edge_dim=1)
        self.egnn2 = EGNN(dim=hidden_dim, edge_dim=1)
        self.egnn3 = EGNN(dim=hidden_dim, edge_dim=1)

        self.sf1 = Selfattention_single(hidden_dim, hidden_dim, hidden_dim)
        self.fc = FC(hidden_dim * 2, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        simi = simi.unsqueeze(3)
        feat1 = self.init_feat1(feat)
        feat2 = self.init_feat2(feat)

        feat1, coords = self.egnn1(feat1, coords, simi)
        feat1, coords = self.egnn2(feat1, coords, simi)
        feat1, coords = self.egnn3(feat1, coords, simi) # shape as (batch, dock, hidden_dim)

        feat2 = self.sf1(feat2) # shape as (batch, dock, hidden_dim)

        feat_out = torch.cat([feat1, feat2], dim=-1)
        feat_out = average_docked_dim(feat_out)
        feat_out = self.fc(feat_out)

        return feat_out
    
class EGNN_edge2_cat_sf1(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_edge3_cat_sf1, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim, edge_dim=1)
        self.egnn2 = EGNN(dim=hidden_dim, edge_dim=1)

        self.sf1 = Selfattention_single(hidden_dim, hidden_dim, hidden_dim)
        self.fc = FC(hidden_dim * 2, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        simi = simi.unsqueeze(3)
        feat1 = self.init_feat1(feat)
        feat2 = self.init_feat2(feat)

        feat1, coords = self.egnn1(feat1, coords, simi)
        feat1, coords = self.egnn2(feat1, coords, simi) # shape as (batch, dock, hidden_dim)

        feat2 = self.sf1(feat2) # shape as (batch, dock, hidden_dim)

        feat_out = torch.cat([feat1, feat2], dim=-1)
        feat_out = average_docked_dim(feat_out)
        feat_out = self.fc(feat_out)

        return feat_out

class EGNN_noedge3_cat_sf1_asliu(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_noedge3_cat_sf1_asliu, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim)
        self.egnn2 = EGNN(dim=hidden_dim)
        self.egnn3 = EGNN(dim=hidden_dim)

        self.sf1 = Selfattention_single(hidden_dim, hidden_dim, hidden_dim)
        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, hidden_dim)
        self.last_linear = nn.Linear(hidden_dim * 2)

    def forward(self, feat, coords):
        feat1 = self.init_feat1(feat)
        feat1 = self.egnn1(feat1, coords)
        feat1 = self.egnn2(feat1, coords)
        feat1 = self.egnn3(feat1, coords)
        feat1 = self.fc(feat1)
        
        feat2 = self.init_feat2(feat)
        feat2 = self.sf1(Selfattention_single)

        feat_out = torch.cat([feat1, feat2], dim=-1)
        feat_out = average_docked_dim(feat_out)
        feat_out = self.last_linear(feat_out)

        return feat_out

class EGNN_noedge2_cat_sf1_asliu(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_noedge2_cat_sf1_asliu, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim)
        self.egnn2 = EGNN(dim=hidden_dim)

        self.sf1 = Selfattention_single(hidden_dim, hidden_dim, hidden_dim)
        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, hidden_dim)
        self.last_linear = nn.Linear(hidden_dim * 2, n_tasks)

    def forward(self, feat, coords):
        feat1 = self.init_feat1(feat)
        feat1, coords = self.egnn1(feat1, coords)
        feat1, coords = self.egnn2(feat1, coords)
        feat1 = self.fc(feat1)
        
        feat2 = self.init_feat2(feat)
        feat2 = self.sf1(feat2)

        feat_out = torch.cat([feat1, feat2], dim=-1)
        feat_out = average_docked_dim(feat_out)
        feat_out = self.last_linear(feat_out)

        return feat_out
    
class EGNN_edge2_cat_sf1_asliu(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_edge2_cat_sf1_asliu, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim, edge_dim=1)
        self.egnn2 = EGNN(dim=hidden_dim, edge_dim=1)

        self.sf1 = Selfattention_single(hidden_dim, hidden_dim, 60)
        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, 60)
        self.last_linear = nn.Linear(120, n_tasks)
    
    def forward(self, feat, coords, simi):
        simi = simi.unsqueeze(3)
        feat1 = self.init_feat1(feat)
        feat2 = self.init_feat2(feat)

        feat1, coords = self.egnn1(feat1, coords, simi)
        feat1, coords = self.egnn2(feat1, coords, simi)
        feat1 = self.fc(feat1)
        feat2 = self.sf1(feat2)

        feat_out = torch.cat([feat1, feat2], dim=-1)
        feat_out = average_docked_dim(feat_out)
        feat_out = self.last_linear(feat_out)

        return feat_out

class EGNN_embed2_cat_sf1(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_embed2_cat_sf1, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_edge = nn.Linear(1, 16, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim)
        self.egnn2 = EGNN(dim=hidden_dim)

        self.sf1 = Selfattention_single(hidden_dim, hidden_dim, hidden_dim)
        self.fc = FC(hidden_dim * 2, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        simi = simi.unsqueeze(3)
        simi = self.init_edge(simi)
        feat1 = self.init_feat1(feat)
        feat2 = self.init_feat2(feat)

        feat1, coords = self.egnn1(feat1, coords, simi)
        feat1, coords = self.egnn2(feat1, coords, simi) # shape as (batch, dock, hidden_dim)

        feat2 = self.sf1(feat2) # shape as (batch, dock, hidden_dim)

        feat_out = torch.cat([feat1, feat2], dim=-1)
        feat_out = average_docked_dim(feat_out)
        feat_out = self.fc(feat_out)

        return feat_out

class EGNN_embed2_mul_sf1(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_embed2_mul_sf1, self).__init__()
        self.init_feat1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_feat2 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.init_edge = nn.Linear(1, 16, bias=False)

        self.egnn1 = EGNN(dim=hidden_dim)
        self.egnn2 = EGNN(dim=hidden_dim)

        self.sf1 = Selfattention_single(hidden_dim, hidden_dim, hidden_dim)
        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)
        self.last_linear = nn.Linear(hidden_dim * 2, n_tasks)

    def forward(self, feat, coords, simi):
        simi = simi.unsqueeze(3)
        simi = self.init_edge(simi)
        feat1 = self.init_feat1(feat)
        feat2 = self.init_feat2(feat)

        feat1, coords = self.egnn1(feat1, coords, simi)
        feat1, coords = self.egnn2(feat1, coords, simi) # shape as (batch, dock, hidden_dim)

        feat2 = self.sf1(feat2) # shape as (batch, dock, hidden_dim)

        feat_out = torch.mul(feat1, feat2)
        feat_out = average_docked_dim(feat_out)
        feat_out = self.fc(feat_out)

        return feat_out        

class ProtEGNN_ConforEGNN_edge(nn.Module):
    def __init__(self, fconfor_dim, fprot_dim, hidden_dim_c, hidden_dim_p, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(ProtEGNN_ConforEGNN_edge, self).__init__()
        self.init_fconfor = nn.Sequential(nn.Linear(fconfor_dim, hidden_dim_c), nn.LeakyReLU(0.1))
        self.init_fprot = nn.Sequential(nn.Linear(fprot_dim, hidden_dim_p), nn.LeakyReLU(0.1))

        self.egnn_c1 = EGNN(dim=hidden_dim_c, edge_dim=1)
        self.egnn_c2 = EGNN(dim=hidden_dim_c, edge_dim=1)
        self.egnn_p1 = EGNN(dim=hidden_dim_p)
        self.egnn_p2 = EGNN(dim=hidden_dim_p)
        self.fc = FC(hidden_dim_p + hidden_dim_c, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, fconfor, fprot, coord_c, coord_p, simi):
        simi = simi.unsqueeze(3)
        fconfor = self.init_fconfor(fconfor)
        fprot = self.init_fprot(fprot)
        coord_p = coord_p.to(torch.float32)

        fconfor, coord_c = self.egnn_c1(fconfor, coord_c, simi)
        fconfor, coord_c = self.egnn_c2(fconfor, coord_c, simi)
        fconfor_out = average_docked_dim(fconfor)

        fprot, coord_p = self.egnn_p1(fprot, coord_p)
        fprot, coord_p = self.egnn_p2(fprot, coord_p)
        fprot_out = average_docked_dim(fprot)

        feat_out = torch.cat([fconfor_out, fprot_out], dim=-1)
        feat_out = self.fc(feat_out)

        return feat_out
        
class ProtEGNN_cross_ConforEGNN_edge(nn.Module):
    def __init__(self, fconfor_dim, fprot_dim, hidden_dim_c, hidden_dim_p, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(ProtEGNN_cross_ConforEGNN_edge, self).__init__()
        self.init_fconfor = nn.Sequential(nn.Linear(fconfor_dim, hidden_dim_c), nn.LeakyReLU(0.1))
        self.init_fprot = nn.Sequential(nn.Linear(fprot_dim, hidden_dim_p), nn.LeakyReLU(0.1))

        self.egnn_c1 = EGNN(dim=hidden_dim_c, edge_dim=1)
        self.egnn_c2 = EGNN(dim=hidden_dim_c, edge_dim=1)
        self.egnn_p1 = EGNN(dim=hidden_dim_p)
        self.egnn_p2 = EGNN(dim=hidden_dim_p)

        self.cross_attn = cross_attention(in_dim_c=hidden_dim_c, in_dim_p=hidden_dim_p, out_dim=d_FC_layer)
        self.fc = FC(d_FC_layer, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, fconfor, fprot, coord_c, coord_p, simi):
        simi = simi.unsqueeze(3)
        fconfor = self.init_fconfor(fconfor)
        fprot = self.init_fprot(fprot)
        coord_p = coord_p.to(torch.float32)

        fconfor, coord_c = self.egnn_c1(fconfor, coord_c, simi)
        fconfor, coord_c = self.egnn_c2(fconfor, coord_c, simi)

        fprot, coord_p = self.egnn_p1(fprot, coord_p)
        fprot, coord_p = self.egnn_p2(fprot, coord_p)

        feat_out = self.cross_attn(fconfor, fprot)
        feat_out = average_docked_dim(feat_out)
        feat_out = self.fc(feat_out)

        return feat_out

class ProtEGNN1_cross_ConforEGNN_edge(nn.Module):
    def __init__(self, fconfor_dim, fprot_dim, hidden_dim_c, hidden_dim_p, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(ProtEGNN1_cross_ConforEGNN_edge, self).__init__()
        self.init_fconfor = nn.Sequential(nn.Linear(fconfor_dim, hidden_dim_c), nn.LeakyReLU(0.1))
        self.init_fprot = nn.Sequential(nn.Linear(fprot_dim, hidden_dim_p), nn.LeakyReLU(0.1))

        self.egnn_c1 = EGNN(dim=hidden_dim_c, edge_dim=1)
        self.egnn_c2 = EGNN(dim=hidden_dim_c, edge_dim=1)
        self.egnn_p1 = EGNN(dim=hidden_dim_p)

        self.cross_attn = cross_attention(in_dim_c=hidden_dim_c, in_dim_p=hidden_dim_p, out_dim=d_FC_layer)
        self.fc = FC(d_FC_layer, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, fconfor, fprot, coord_c, coord_p, simi):
        simi = simi.unsqueeze(3)
        fconfor = self.init_fconfor(fconfor)
        fprot = self.init_fprot(fprot)
        coord_p = coord_p.to(torch.float32)

        fconfor, coord_c = self.egnn_c1(fconfor, coord_c, simi)
        fconfor, coord_c = self.egnn_c2(fconfor, coord_c, simi)

        fprot, coord_p = self.egnn_p1(fprot, coord_p)

        feat_out = self.cross_attn(fconfor, fprot)
        feat_out = average_docked_dim(feat_out)
        feat_out = self.fc(feat_out)

        return feat_out

class GIGN_ave(nn.Module):
    def __init__(self, fconfor_dim, hidden_dim_c, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(GIGN_ave, self).__init__()
        self.init_fconfor = nn.Sequential(nn.Linear(fconfor_dim, hidden_dim_c), nn.LeakyReLU(0.1))
        self.fc = FC(d_FC_layer, d_FC_layer, n_FC_layer, dropout, n_tasks)
    def forward(self, fconfor):
        fconfor = self.init_fconfor(fconfor)
        feat_out = average_docked_dim(fconfor)
        feat_out = self.fc(feat_out)

        return feat_out
