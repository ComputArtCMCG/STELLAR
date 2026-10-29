import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from egnn_pytorch import EGNN

class Selfattention_single_tanh(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Selfattention_single_tanh, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.q_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.k_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.v_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear_last = nn.Sequential(nn.Linear(in_dim + hidden_dim, out_dim), nn.Tanh())

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
    
class Selfattention_single_sf(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Selfattention_single_sf, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.q_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.k_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.v_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear_last = nn.Sequential(nn.Linear(in_dim + hidden_dim, out_dim))

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
        out = torch.softmax(out, dim=-1)

        return out

class Selfattention_single(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Selfattention_single, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.q_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.k_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.v_transform = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear_last = nn.Sequential(nn.Linear(in_dim + hidden_dim, out_dim))

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
    
class cross_attention(nn.Module):
    '''
    fconfor: feat of conformation, shape as(batch, docked, fconfor)
    fprot: feat of protein pocket, shape as(batch, res_num, fprot)
    '''
    def __init__(self, in_dim_p, in_dim_c, out_dim):
        super(cross_attention, self).__init__()
        self.in_dim_p = in_dim_p
        self.in_dim_c = in_dim_c
        self.out_dim = out_dim

        self.q_transform = nn.Linear(in_dim_c, out_dim, bias=False)
        self.k_transform = nn.Linear(in_dim_p, out_dim, bias=False)
        self.v_transform = nn.Linear(in_dim_p, out_dim, bias=False)
        self.linear_last = nn.Linear(in_dim_c + out_dim, out_dim)

    def forward(self, fconfor, fprot):
        q = self.q_transform(fconfor) # shape as(batch, docked, out_dim)
        k = self.k_transform(fprot) # shape as(batch, res_num, out_dim)
        v = self.v_transform(fprot) # shape as(batch, res_num, out_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1) / self.out_dim ** 0.5) # shape as(batch, docked, res_num)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out_msg = torch.matmul(attn_weights, v) # shape as(batch, docked, out_dim)  
        out = self.linear_last(torch.cat([fconfor, out_msg], dim=-1))

        return out
        

