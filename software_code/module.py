import torch
import torch.nn as nn
from utils import *
from layer import *

class GCN_norm(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, simi_threshold):
        super(GCN_norm, self).__init__()
        self.gconv1 = GCNLayer_norm(feat_dim, hidden_dim, simi_threshold)
        self.gconv2 = GCNLayer_norm(hidden_dim, hidden_dim, simi_threshold)
        self.gconv3 = GCNLayer_norm(hidden_dim, hidden_dim, simi_threshold)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        x = self.gconv1(feat, simi)
        x = self.gconv2(x, simi)
        x = self.gconv3(x, simi)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class GCN_simi(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(GCN_simi, self).__init__()
        self.gconv1 = GCNLayer_simi(feat_dim, hidden_dim)
        self.gconv2 = GCNLayer_simi(hidden_dim, hidden_dim)
        self.gconv3 = GCNLayer_simi(hidden_dim, hidden_dim)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        x = self.gconv1(feat, simi)
        x = self.gconv2(x, simi)
        x = self.gconv3(x, simi)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class GAT_norm(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num, simi_threshold):
        super(GAT_norm, self).__init__()
        self.gat1 = GATLayer_norm(feat_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat2 = GATLayer_norm(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat3 = GATLayer_norm(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        x = self.gat1(feat, simi)
        x = self.gat2(x, simi)
        x = self.gat3(x, simi)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class GAT_gate(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num, simi_threshold):
        super(GAT_gate, self).__init__()
        self.gat1 = GATLayer_gate(feat_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat2 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat3 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        x = self.gat1(feat, simi)
        x = self.gat2(x, simi)
        x = self.gat3(x, simi)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class GAT_simi(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(GAT_simi, self).__init__()
        self.gat1 = GATLayer_simi(feat_dim, hidden_dim, head_num, dropout)
        self.gat2 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)
        self.gat3 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        x = self.gat1(feat, simi)
        x = self.gat2(x, simi)
        x = self.gat3(x, simi)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class GCN_norm_and_SF(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, n_tasks, dropout, head_num, simi_threshold):
        super(GCN_norm_and_SF, self).__init__()
        self.gconv1 = GCNLayer_norm(feat_dim, hidden_dim, simi_threshold)
        self.gconv2 = GCNLayer_norm(hidden_dim, hidden_dim, simi_threshold)
        self.gconv3 = GCNLayer_norm(hidden_dim, hidden_dim, simi_threshold)

        self.sf1 = Selfattention(feat_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        a = self.gconv1(feat, simi)
        a = self.gconv2(a, simi)
        a = self.gconv3(a, simi) # shape as (batch, docked, out_dim)

        b = self.sf1(feat) # shape as (batch, docked, 1)

        x = torch.mul(a, b) # shape as (batch, docked, out_dim)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class GCN_simi_and_SF(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(GCN_simi_and_SF, self).__init__()
        self.gconv1 = GCNLayer_simi(feat_dim, hidden_dim)
        self.gconv2 = GCNLayer_simi(hidden_dim, hidden_dim)
        self.gconv3 = GCNLayer_simi(hidden_dim, hidden_dim)

        self.sf1 = Selfattention(feat_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)
    
    def forward(self, feat, simi):
        a = self.gconv1(feat, simi)
        a = self.gconv2(a, simi)
        a = self.gconv3(a, simi) # shape as (batch, docked, out_dim)

        b = self.sf1(feat)

        x = torch.mul(a, b) # shape as (batch, docked, out_dim)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class GAT_norm_and_SF(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num, simi_threshold):
        super(GAT_norm_and_SF, self).__init__()
        self.gat1 = GATLayer_norm(feat_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat2 = GATLayer_norm(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat3 = GATLayer_norm(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)

        self.sf1 = Selfattention(feat_dim, hidden_dim, hidden_dim, head_num)
        self.sf2 = Selfattention(hidden_dim, hidden_dim, hidden_dim, head_num)
        self.sf3 = Selfattention(hidden_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.gat3(a, simi) # shape as (batch, docked, out_dim)

        b = self.sf1(feat)
        b = self.sf2(b)
        b = self.sf3(b) # shape as (batch, docked, 1)

        x = torch.mul(a, b) # shape as (batch, docked, out_dim)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class GAT_gate_and_SF(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num, simi_threshold):
        super(GAT_gate_and_SF, self).__init__()
        self.gat1 = GATLayer_gate(feat_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat2 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat3 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)

        self.sf1 = Selfattention(feat_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.gat3(a, simi) # shape as (batch, docked, out_dim)

        b = self.sf1(feat) # shape as (batch, docked, 1)

        x = torch.mul(a, b) # shape as (batch, docked, out_dim)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class GAT_simi_and_SF(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(GAT_simi_and_SF, self).__init__()
        self.gat1 = GATLayer_simi(feat_dim, hidden_dim, head_num, dropout)
        self.gat2 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)
        self.gat3 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)

        self.sf1 = Selfattention(feat_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.gat3(a, simi) # shape as (batch, docked, out_dim)

        b = self.sf1(feat) # shape as (batch, docked, 1)

        x = torch.mul(a, b) # shape as (batch, docked, out_dim)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class GAT_simi_and_SF(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(GAT_simi_and_SF, self).__init__()
        self.gat1 = GATLayer_simi(feat_dim, hidden_dim, head_num, dropout)
        self.gat2 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)
        self.gat3 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)

        self.sf1 = Selfattention(feat_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.gat3(a, simi) # shape as (batch, docked, out_dim)

        b = self.sf1(feat) # shape as (batch, docked, 1)

        x = torch.mul(a, b) # shape as (batch, docked, out_dim)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class GAT_simi2_and_SFsf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(GAT_simi2_and_SFsf, self).__init__()
        self.gat1 = GATLayer_simi(feat_dim, hidden_dim, head_num, dropout)
        self.gat2 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)

        self.sf1 = SelfattentionV3(feat_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi) # shape as (batch, docked, out_dim)

        b = self.sf1(feat) # shape as (batch, docked, 1)

        x = torch.mul(a, b) # shape as (batch, docked, out_dim)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class GAT_simi_and_SFsf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(GAT_simi_and_SFsf, self).__init__()
        self.gat1 = GATLayer_simi(feat_dim, hidden_dim, head_num, dropout)
        self.gat2 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)
        self.gat3 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)

        self.sf1 = SelfattentionV3(feat_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.gat3(a, simi) # shape as (batch, docked, out_dim)

        b = self.sf1(feat) # shape as (batch, docked, 1)

        x = torch.mul(a, b) # shape as (batch, docked, out_dim)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class GAT_gate_and_SFsf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num, simi_threshold):
        super(GAT_gate_and_SF, self).__init__()
        self.gat1 = GATLayer_gate(feat_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat2 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat3 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)

        self.sf1 = SelfattentionV3(feat_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.gat3(a, simi) # shape as (batch, docked, out_dim)

        b = self.sf1(feat) # shape as (batch, docked, 1)

        x = torch.mul(a, b) # shape as (batch, docked, out_dim)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class SF_and_GAT_gate(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num, simi_threshold):
        super(SF_and_GAT_gate, self).__init__()
        self.gat1 = GATLayer_gate(feat_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat2 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat3 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)

        self.linear = nn.Sequential(nn.Linear(hidden_dim, 1), 
                                    nn.Tanh())

        self.sf1 = SelfattentionV2(feat_dim, hidden_dim, hidden_dim, head_num)
        self.sf2 = SelfattentionV2(hidden_dim, hidden_dim, hidden_dim, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.gat3(a, simi)
        a = self.linear(a)

        b = self.sf1(feat)
        b = self.sf2(b)

        x = torch.mul(b, a)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class GAT_gate_and_SF2(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num, simi_threshold):
        super(GAT_gate_and_SF2, self).__init__()
        self.gat1 = GATLayer_gate(feat_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat2 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)
        self.gat3 = GATLayer_gate(hidden_dim, hidden_dim, head_num, dropout, simi_threshold)

        self.sf1 = SelfattentionV2(feat_dim, hidden_dim, hidden_dim, head_num)
        self.sf2 = Selfattention(hidden_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)
    
    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.gat3(a, simi)

        b = self.sf1(feat)
        b = self.sf2(b)

        x = torch.mul(a, b)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class GAT_simi_and_SF2(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(GAT_simi_and_SF2, self).__init__()
        self.gat1 = GATLayer_simi(feat_dim, hidden_dim, head_num, dropout)
        self.gat2 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)
        self.gat3 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)

        self.sf1 = SelfattentionV2(feat_dim, hidden_dim, hidden_dim, head_num)
        self.sf2 = Selfattention(hidden_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)
    
    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.gat3(a, simi)

        b = self.sf1(feat)
        b = self.sf2(b)

        x = torch.mul(a, b)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class GAT_simi2_and_SFsf2(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks, head_num):
        super(GAT_simi2_and_SFsf2, self).__init__()
        self.gat1 = GATLayer_simi(feat_dim, hidden_dim, head_num, dropout)
        self.gat2 = GATLayer_simi(hidden_dim, hidden_dim, head_num, dropout)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        self.sf1 = SelfattentionV2(feat_dim, hidden_dim, hidden_dim, head_num)
        self.sf2 = SelfattentionV3(hidden_dim, hidden_dim, 1, head_num)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)
    
    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.linear(a)

        b = self.sf1(feat)
        b = self.sf2(b)

        x = torch.mul(a, b)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
    
class GAT_simi2_and_SFsf_1h(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(GAT_simi2_and_SFsf_1h, self).__init__()
        self.gat1 = GATLayer_simi_1h(feat_dim, hidden_dim, dropout)
        self.gat2 = GATLayer_simi_1h(hidden_dim, hidden_dim, dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

        self.sf1 = SelfattentionV2_1h(feat_dim, hidden_dim, hidden_dim)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)
    
    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.linear1(a)

        b = self.sf1(feat)
        b = self.linear2(b)
        b = torch.softmax(b, dim=-1)
        
        x = torch.mul(a, b)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x

class GAT_simi2_and_SFsf2_1h(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(GAT_simi2_and_SFsf2_1h, self).__init__()
        self.gat1 = GATLayer_simi_1h(feat_dim, hidden_dim, dropout)
        self.gat2 = GATLayer_simi_1h(hidden_dim, hidden_dim, dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        self.sf1 = SelfattentionV2_1h(feat_dim, hidden_dim, hidden_dim)
        self.sf2 = SelfattentionV2_1h(feat_dim, hidden_dim, 1)

        self.fc = FC(hidden_dim, d_FC_layer, n_FC_layer, dropout, n_tasks)
    
    def forward(self, feat, simi):
        a = self.gat1(feat, simi)
        a = self.gat2(a, simi)
        a = self.linear1(a)

        b = self.sf1(feat)
        b = self.sf2(b)
        b = torch.softmax(b, dim=-1)
        
        x = torch.mul(a, b)
        x = average_docked_dim(x)
        x = self.fc(x)

        return x
