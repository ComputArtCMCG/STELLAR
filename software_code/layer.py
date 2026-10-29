import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer_norm(nn.Module):
    def __init__(self, in_dim, out_dim, simi_threshold):
        super(GCNLayer_norm, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.simi_threshold = simi_threshold

        self.mlp = nn.Linear(in_dim, out_dim)
    
    def forward(self, feat, simi):
        '''
        feat: node features of each graph, shape as (batch, docked, feat_dim256)
        simi: simi_matrix, shape as (batch, docked, docked)
        '''
        zero_vec = torch.zeros_like(simi)
        one_vec = torch.ones_like(simi)
        adj = torch.where(simi > self.simi_threshold, one_vec, zero_vec)
        D = torch.sum(adj, dim=1)
        D = torch.diag_embed(D) # shape as (batch, docked, docked)
        D_inv = torch.sqrt(torch.inverse(D))
        DAD = torch.matmul(torch.matmul(D_inv, adj), D_inv) # shape as (batch, docked, docked)
        update_feat = torch.matmul(DAD, feat) # shape as (batch, docked, feat_dim)
        update_feat = self.mlp(update_feat) # shape as (batch, docked, feat_dim)

        return update_feat

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
        D = torch.diag_embed(D)
        D_inv = torch.sqrt(torch.inverse(D))
        DAD = torch.matmul(torch.matmul(D_inv, simi), D_inv)
        update_feat = torch.matmul(DAD, feat)
        update_feat = self.mlp(update_feat)

        return update_feat

class GATLayer_norm(nn.Module):
    def __init__(self, in_dim, out_dim, head_num, dropout, simi_threshold):
        super(GATLayer_norm, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num
        self.dropout = dropout
        self.simi_threshold = simi_threshold

        self.q_transform = nn.Linear(in_dim, out_dim * head_num, bias=False)
        self.k_transform = nn.Linear(in_dim, out_dim * head_num, bias=False)
        self.v_transform = nn.Linear(in_dim, out_dim * head_num, bias=False)
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
        q = self.q_transform(feat).view(batch_size, -1, self.head_num, self.out_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, out_dim)
        k = self.k_transform(feat).view(batch_size, -1, self.head_num, self.out_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, out_dim)
        v = self.v_transform(feat).view(batch_size, -1, self.head_num, self.out_dim).permute(0, 2, 1, 3) # shape as(batch, head, docked, out_dim)
    
        simi_h = simi.unsqueeze(1).repeat(1, self.head_num, 1, 1)
        zero_vec = -9e15 * torch.ones_like(simi_h) # shape as (batch, head, docked, docked)
        attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / (self.out_dim ** 0.5) # shape as (batch, head, docked, docked)
        attn_matrix = torch.where(simi_h > self.simi_threshold, attn_matrix, zero_vec)
        attn_score = torch.softmax(attn_matrix, dim=-1) # shape as (batch, head, docked, docked)

        update_msg = torch.matmul(attn_score, v) # shape as (batch, head, docked, out_dim)
        update_msg = update_msg.transpose(1, 2).reshape(batch_size, -1, self.head_num * self.out_dim) # shape as (batch, docked, head * out_dim)
        update_msg = self.head_transform(update_msg) # shape as (batch, docked, out_dim)
        update_feat = torch.cat([feat, update_msg], dim=-1) # shape as (batch, docked, out_dim + out_dim)
        update_feat = self.linear_last(update_feat)

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

class GATLayer_simi_1h(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(GATLayer_simi_1h, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.q_transform = nn.Linear(in_dim, out_dim, bias=False)
        self.k_transform = nn.Linear(in_dim, out_dim, bias=False)
        self.v_transform = nn.Linear(in_dim, out_dim, bias=False)
        self.msg_transform = nn.Linear(out_dim * 2, 1)
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
        q = self.q_transform(feat).view(batch_size, -1, self.out_dim) # shape as(batch, docked, out_dim)
        k = self.k_transform(feat).view(batch_size, -1, self.out_dim) # shape as(batch, docked, out_dim)
        v = self.v_transform(feat).view(batch_size, -1, self.out_dim) # shape as(batch, docked, out_dim)

        attn_msg = torch.cat([q.unsqueeze(1).repeat(1, docked, 1, 1), k.unsqueeze(2).repeat(1, 1, docked, 1)], dim=-1) # shape as(batch, docked, docked, out_dim * 2)
        attn_msg = self.msg_transform(attn_msg).squeeze(-1) # shape as (batch, docked, docked)
        attn_matrix = simi * attn_msg
        attn_score = torch.softmax(attn_matrix, dim=-1) # shape as (batch, docked, docked)

        update_msg = torch.matmul(attn_score, v) # shape as (batch, docked, out_dim)
        update_feat = torch.cat([feat, update_msg], dim=-1) # shape as (batch, docked, in_dim + out_dim)
        update_feat = self.linear_last(update_feat) # shape as (batch, docked, out_dim)

        return update_feat
    
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

class SelfattentionV2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, head_num):
        super(SelfattentionV2, self).__init__()
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

class SelfattentionV2_1h(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SelfattentionV2_1h, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.q_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.k_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.v_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear_last = nn.Linear(in_dim + hidden_dim, out_dim)

    def forward(self, feat):
        '''
        feat: node features of each graph, shape as (batch, docked, feat_dim256)
        simi: simi_matrix, shape as (batch, docked, docked)
        '''
        batch_size = feat.size()[0]
        q = self.q_transform(feat).view(batch_size, -1, self.hidden_dim) # shape as(batch, docked, hidden_dim)
        k = self.k_transform(feat).view(batch_size, -1, self.hidden_dim) # shape as(batch, docked, hidden_dim)
        v = self.v_transform(feat).view(batch_size, -1, self.hidden_dim) # shape as(batch, docked, hidden_dim)
        
        attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5) # shape as (batch, docked, docked)
        attn_score = torch.softmax(attn_matrix, dim=-1)
        out_msg = torch.matmul(attn_score, v)
        out = self.linear_last(torch.cat([feat, out_msg], dim=-1)) # shape as (batch, docked, out)

        return out
    
class SelfattentionV3(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, head_num):
        super(SelfattentionV3, self).__init__()
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













