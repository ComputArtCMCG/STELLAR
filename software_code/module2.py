import torch
import torch.nn as nn
from utils import *
from layer2 import *

# GCN
class GCN_simi3(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(GCN_simi3, self).__init__()
        self.init_feat = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.gconv1 = GCNLayer_simi(hidden_dim, hidden_dim)
        self.gconv2 = GCNLayer_simi(hidden_dim, hidden_dim)
        self.gconv3 = GCNLayer_simi(hidden_dim, hidden_dim)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        x = self.init_feat(feat)
        x = self.gconv1(x, simi)
        x = self.gconv2(x, simi)
        x = self.gconv3(x, simi)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

# GAT
class GAT_gate3_and_SF1_tanh(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num, simi_threshold):
        super(GAT_gate3_and_SF1_tanh, self).__init__()
        self.init_feat1 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.init_feat2 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.gat1 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat2 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat3 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)

        self.sf1 = Selfattention_tanh(hidden_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        a = self.init_feat1(feat)
        a = self.gat1(a, simi)
        a = self.gat2(a, simi)
        a = self.gat3(a, simi) # shape as (batch, docked, out_dim)

        b = self.init_feat2(feat)
        b = self.sf1(b) # shape as (batch, docked, 1)

        x = torch.mul(a, b) # shape as (batch, docked, out_dim)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

# EGNN
class EGNN_simi3(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_simi3, self).__init__()
        self.init_feat = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.egnn1 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn2 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn3 = EGNN(hidden_dim, hidden_dim, hidden_dim)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        x = self.init_feat(feat)
        x = self.egnn1(x, coords, simi)
        x = self.egnn2(x, coords, simi)
        x = self.egnn3(x, coords, simi)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class EGNN_simi2(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_simi2, self).__init__()
        self.init_feat = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.egnn1 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn2 = EGNN(hidden_dim, hidden_dim, hidden_dim)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        x = self.init_feat(feat)
        x = self.egnn1(x, coords, simi)
        x = self.egnn2(x, coords, simi)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class EGNN_simi1(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(EGNN_simi1, self).__init__()
        self.init_feat = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.egnn1 = EGNN(hidden_dim, hidden_dim, hidden_dim)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        x = self.init_feat(feat)
        x = self.egnn1(x, coords, simi)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class EGNN_simi2_and_sf1_sf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(EGNN_simi2_and_sf1_sf, self).__init__()
        self.init_feat1 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.init_feat2 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.egnn1 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn2 = EGNN(hidden_dim, hidden_dim, hidden_dim)

        self.sf1 = Selfattention_sf(hidden_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        a = self.init_feat1(feat)
        a = self.egnn1(a, coords, simi)
        a = self.egnn2(a, coords, simi)
        
        b = self.init_feat2(feat)
        b = self.sf1(b)

        x = torch.mul(a, b)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class EGNN_simi2_and_sf1_tanh(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(EGNN_simi2_and_sf1_sf, self).__init__()
        self.init_feat1 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.init_feat2 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.egnn1 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn2 = EGNN(hidden_dim, hidden_dim, hidden_dim)

        self.sf1 = Selfattention_tanh(hidden_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        a = self.init_feat1(feat)
        a = self.egnn1(a, coords, simi)
        a = self.egnn2(a, coords, simi)
        
        b = self.init_feat2(feat)
        b = self.sf1(b)

        x = torch.mul(a, b)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class EGNN_simi3_and_sf1_sf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(EGNN_simi3_and_sf1_sf, self).__init__()
        self.init_feat1 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.init_feat2 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.egnn1 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn2 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn3 = EGNN(hidden_dim, hidden_dim, hidden_dim)

        self.sf1 = Selfattention_sf(hidden_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        a = self.init_feat1(feat)
        a = self.egnn1(a, coords, simi)
        a = self.egnn2(a, coords, simi)
        a = self.egnn3(a, coords, simi)
        
        b = self.init_feat2(feat)
        b = self.sf1(b)

        x = torch.mul(a, b)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class EGNN_simi3_and_sf1_tanh(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(EGNN_simi3_and_sf1_tanh, self).__init__()
        self.init_feat1 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.init_feat2 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.egnn1 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn2 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn3 = EGNN(hidden_dim, hidden_dim, hidden_dim)

        self.sf1 = Selfattention_sf(hidden_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        a = self.init_feat1(feat)
        a = self.egnn1(a, coords, simi)
        a = self.egnn2(a, coords, simi)
        a = self.egnn3(a, coords, simi)
        
        b = self.init_feat2(feat)
        b = self.sf1(b)

        x = torch.mul(a, b)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class EGNN_simi3_cat_sf1(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(EGNN_simi3_cat_sf1, self).__init__()
        self.init_feat1 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.init_feat2 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.egnn1 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn2 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn3 = EGNN(hidden_dim, hidden_dim, hidden_dim)

        self.sf1 = Selfattention(hidden_dim, hidden_dim, hidden_dim, head_num)

        self.fc = FC(hidden_dim * 2, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        a = self.init_feat1(feat)
        a = self.egnn1(a, coords, simi)
        a = self.egnn2(a, coords, simi)
        a = self.egnn3(a, coords, simi)
        
        b = self.init_feat2(feat)
        b = self.sf1(b)

        x = torch.cat([a, b], dim=-1)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class EGNN_simi2_cat_sf1(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(EGNN_simi2_cat_sf1, self).__init__()
        self.init_feat1 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.init_feat2 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False), 
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.egnn1 = EGNN(hidden_dim, hidden_dim, hidden_dim)
        self.egnn2 = EGNN(hidden_dim, hidden_dim, hidden_dim)

        self.sf1 = Selfattention(hidden_dim, hidden_dim, hidden_dim, head_num)

        self.fc = FC(hidden_dim * 2, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, coords, simi):
        a = self.init_feat1(feat)
        a = self.egnn1(a, coords, simi)
        a = self.egnn2(a, coords, simi)
        
        b = self.init_feat2(feat)
        b = self.sf1(b)

        x = torch.cat([a, b], dim=-1)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    

