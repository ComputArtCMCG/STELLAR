import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer_simi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer_simi, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.mlp = nn.Linear(in_dim, out_dim)

    def forward(self, feat, simi):
        '''
        feat: node features of each graph, shape as (batch, docked, feat_dim256)
        simi: simi_matrix, shape as (batch, docked, docked)
        '''
        D = torch.sum(simi, dim=1)
        D = torch.where(D == 0, torch.ones_like(D), D)
        D = torch.diag_embed(D)
        D_inv = torch.sqrt(torch.inverse(D))
        DAD = torch.matmul(torch.matmul(D_inv, simi), D_inv)
        update_feat = torch.matmul(DAD, feat)
        update_feat = self.mlp(update_feat)

        return update_feat

class GATLayer_gate(nn.Module):
    def __init__(self, in_dim, out_dim, head_num, dropout, simi_threshold):
        super(GATLayer_gate, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num
        self.dropout = dropout
        self.simi_threshold = simi_threshold

        self.q_transform = nn.Linear(in_dim, out_dim * head_num, bias=False)
        self.k_transform = nn.Linear(in_dim, out_dim * head_num, bias=False)
        self.v_transform = nn.Linear(in_dim, out_dim * head_num, bias=False)
        self.msg_transform = nn.Linear(out_dim * 2, 1)
        self.head_transform = nn.Linear(out_dim * head_num, out_dim)
        self.linear_last = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), 
            nn.Dropout(dropout), 
            nn.LeakyReLU())
    
    def forward(self, feat, simi):
        '''
        feat: node features of each graph, shape as (batch, docked, feat_dim256)
        simi: simi_matrix, shape as (batch, docked, docked)
        '''
        batch_size = feat.size()[0]
        docked = feat.size()[1]
        q = self.q_transform(feat).view(batch_size, -1, self.head_num, self.out_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, out_dim)
        k = self.k_transform(feat).view(batch_size, -1, self.head_num, self.out_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, out_dim)
        v = self.v_transform(feat).view(batch_size, -1, self.head_num, self.out_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, out_dim)

        simi_h = simi.unsqueeze(1).repeat(1, self.head_num, 1, 1)
        zero_vec = -9e15 * torch.ones_like(simi_h) # shape as (batch, head, docked, docked)
        attn_msg = torch.cat([q.unsqueeze(2).repeat(1, 1, docked, 1, 1), k.unsqueeze(3).repeat(1, 1, 1, docked, 1)], dim=-1) # shape as(batch, head, docked, docked, out_dim * 2)
        attn_msg = self.msg_transform(attn_msg).squeeze(-1) # shape as (batch, head, docked, docked)
        attn_matrix = torch.where(simi_h > self.simi_threshold, attn_msg, zero_vec)
        attn_score = torch.softmax(attn_matrix, dim=-1) # shape as (batch, head, docked, docked)

        update_msg = torch.matmul(attn_score, v).permute(0, 2, 1, 3).reshape(batch_size, -1, self.head_num * self.out_dim) # shape as (batch, docked, head * out_dim)
        update_msg = self.head_transform(update_msg) # shape as (batch, docked, out_dim)
        update_feat = torch.cat([feat, update_msg], dim=-1) # shape as (batch, docked, in_dim + out_dim)
        update_feat = self.linear_last(update_feat) # shape as (batch, docked, out_dim)

        return update_feat

class GATLayer_simi(nn.Module):
    def __init__(self, in_dim, out_dim, head_num, dropout):
        super(GATLayer_simi, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num
        self.dropout = dropout

        self.q_transform = nn.Linear(in_dim, out_dim * head_num, bias=False)
        self.k_transform = nn.Linear(in_dim, out_dim * head_num, bias=False)
        self.v_transform = nn.Linear(in_dim, out_dim * head_num, bias=False)
        self.msg_transform = nn.Linear(out_dim * 2, 1)
        self.head_transform = nn.Linear(out_dim * head_num, out_dim)
        self.linear_last = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), 
            nn.Dropout(dropout), 
            nn.LeakyReLU())
    
    def forward(self, feat, simi):
        '''
        feat: node features of each graph, shape as (batch, docked, feat_dim256)
        simi: simi_matrix, shape as (batch, docked, docked)
        '''
        batch_size = feat.size()[0]
        docked = feat.size()[1]
        q = self.q_transform(feat).view(batch_size, -1, self.head_num, self.out_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, out_dim)
        k = self.k_transform(feat).view(batch_size, -1, self.head_num, self.out_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, out_dim)
        v = self.v_transform(feat).view(batch_size, -1, self.head_num, self.out_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, out_dim)

        simi_h = simi.unsqueeze(1).repeat(1, self.head_num, 1, 1)
        attn_msg = torch.cat([q.unsqueeze(2).repeat(1, 1, docked, 1, 1), k.unsqueeze(3).repeat(1, 1, 1, docked, 1)], dim=-1) # shape as(batch, head, docked, docked, out_dim * 2)
        attn_msg = self.msg_transform(attn_msg).squeeze(-1) # shape as (batch, head, docked, docked)
        attn_matrix = simi_h * attn_msg
        attn_score = torch.softmax(attn_matrix, dim=-1) # shape as (batch, head, docked, docked)

        update_msg = torch.matmul(attn_score, v).permute(0, 2, 1, 3).reshape(batch_size, -1, self.head_num * self.out_dim) # shape as (batch, docked, head * out_dim)
        update_msg = self.head_transform(update_msg) # shape as (batch, docked, out_dim)
        update_feat = torch.cat([feat, update_msg], dim=-1) # shape as (batch, docked, in_dim + out_dim)
        update_feat = self.linear_last(update_feat) # shape as (batch, docked, out_dim)

        return update_feat

class Selfattention_tanh(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, head_num):
        super(Selfattention_tanh, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num

        self.q_transform = nn.Linear(in_dim, hidden_dim * head_num, bias=False)
        self.k_transform = nn.Linear(in_dim, hidden_dim * head_num, bias=False)
        self.v_transform = nn.Linear(in_dim, hidden_dim * head_num, bias=False)
        self.head_transform = nn.Linear(hidden_dim * head_num, hidden_dim)
        self.linear_last = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, out_dim), 
            nn.Tanh())

    def forward(self, feat):
        '''
        feat: node features of each graph, shape as (batch, docked, feat_dim256)
        simi: simi_matrix, shape as (batch, docked, docked)
        '''
        batch_size = feat.size()[0]
        q = self.q_transform(feat).view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, hidden_dim)
        k = self.k_transform(feat).view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, hidden_dim)
        v = self.v_transform(feat).view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, hidden_dim)
        
        attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5) # shape as (batch, head, docked, docked)
        attn_score = torch.softmax(attn_matrix, dim=-1)
        out_msg = torch.matmul(attn_score, v).permute(0, 2, 1, 3).reshape(batch_size, -1, self.head_num * self.hidden_dim)
        out_msg = self.head_transform(out_msg) # shape as (batch, docked, hidden_dim)
        out = self.linear_last(torch.cat([feat, out_msg], dim=-1)) # shape as (batch, docked, out)

        return out

class Selfattention_sf(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, head_num):
        super(Selfattention_sf, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num

        self.q_transform = nn.Linear(in_dim, hidden_dim * head_num, bias=False)
        self.k_transform = nn.Linear(in_dim, hidden_dim * head_num, bias=False)
        self.v_transform = nn.Linear(in_dim, hidden_dim * head_num, bias=False)
        self.head_transform = nn.Linear(hidden_dim * head_num, hidden_dim)
        self.linear_last = nn.Linear(in_dim + hidden_dim, out_dim)

    def forward(self, feat):
        '''
        feat: node features of each graph, shape as (batch, docked, feat_dim256)
        simi: simi_matrix, shape as (batch, docked, docked)
        '''
        batch_size = feat.size()[0]
        q = self.q_transform(feat).view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, hidden_dim)
        k = self.k_transform(feat).view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, hidden_dim)
        v = self.v_transform(feat).view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, hidden_dim)
        
        attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5) # shape as (batch, head, docked, docked)
        attn_score = torch.softmax(attn_matrix, dim=-1)
        out_msg = torch.matmul(attn_score, v).permute(0, 2, 1, 3).reshape(batch_size, -1, self.head_num * self.hidden_dim)
        out_msg = self.head_transform(out_msg) # shape as (batch, docked, hidden_dim)
        out = self.linear_last(torch.cat([feat, out_msg], dim=-1)) # shape as (batch, docked, out)
        out = torch.softmax(out, dim=-1)

        return out

class Selfattention(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, head_num):
        super(Selfattention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num

        self.q_transform = nn.Linear(in_dim, hidden_dim * head_num, bias=False)
        self.k_transform = nn.Linear(in_dim, hidden_dim * head_num, bias=False)
        self.v_transform = nn.Linear(in_dim, hidden_dim * head_num, bias=False)
        self.head_transform = nn.Linear(hidden_dim * head_num, hidden_dim)
        self.linear_last = nn.Linear(in_dim + hidden_dim, out_dim)

    def forward(self, feat):
        '''
        feat: node features of each graph, shape as (batch, docked, feat_dim256)
        simi: simi_matrix, shape as (batch, docked, docked)
        '''
        batch_size = feat.size()[0]
        q = self.q_transform(feat).view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, hidden_dim)
        k = self.k_transform(feat).view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, hidden_dim)
        v = self.v_transform(feat).view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, hidden_dim)
        
        attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5) # shape as (batch, head, docked, docked)
        attn_score = torch.softmax(attn_matrix, dim=-1)
        out_msg = torch.matmul(attn_score, v).permute(0, 2, 1, 3).reshape(batch_size, -1, self.head_num * self.hidden_dim)
        out_msg = self.head_transform(out_msg) # shape as (batch, docked, hidden_dim)
        out = self.linear_last(torch.cat([feat, out_msg], dim=-1)) # shape as (batch, docked, out)

        return out

class EGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(EGNN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear_e = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, hidden_dim), 
            nn.LeakyReLU(negative_slope=0.1), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.linear_h = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.LeakyReLU(negative_slope=0.1), 
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, feat, coordinates, simi):
        dock_num = feat.size()[1]
        coor_diff = coordinates.unsqueeze(2) - coordinates.unsqueeze(1) # shape as (batch, dock, dock, 3)
        m_input = torch.cat([feat.unsqueeze(1).repeat(1, dock_num, 1, 1),
                             feat.unsqueeze(2).repeat(1, 1, dock_num, 1), 
                             torch.square(coor_diff).sum(dim=-1, keepdim=True)], dim=-1)
        m = self.linear_e(m_input) # shape as (batch, dock, dock, feat)
        m_simi = torch.sum(simi.unsqueeze(-1) * m, dim=2) # shape as (batch, dock, feat)
        update_feat = self.linear_h(torch.cat((feat, m_simi), dim=-1))
        
        return update_feat

# class EGNN_adj(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super(EGNN_adj, self).__init__()
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
        

class LSTM(nn.Module):
    def __init__(self, in_dim, out_dim, num_layer):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer

        self.lstm = nn.LSTM(in_dim, out_dim, num_layer, batch_first=True)

    def forward(self, feat):
        h0 = torch.zeros(self.num_layer, feat.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layer, feat.size(0), self.hidden_dim)

        out, _ = self.lstm(feat, (h0, c0))
        out = out[:, -1, :] # shape as (batch_size, out_dim)

        return out

class FC(nn.Module):
    def __init__(self, d_in_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_in_layer = d_in_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_in_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
    
    def forward(self, h):
        for layer in self.predict:
            h = layer(h).squeeze(-1)

        return h
