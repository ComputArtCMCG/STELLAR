import os
import pickle
import torch
import datetime
import numpy as np
from protein_feat_generate import WholeProteinPocket, Residue

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def GIGNfeatcollect(keys, labels):
    dic = {}
    for key, value in zip(keys, labels):
        dic[key] = value
    return dic

def get_select_dict(selected_path):
    selected_dict = {}
    with open(selected_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key = line.rstrip().split('-')[0]
        selected_list = eval(line.rstrip().split('-')[-1])
        selected_list = [x + 1 for x in selected_list]
        selected_dict[key] = selected_list

    return selected_dict

def get_mass_center(mc_path):
    mc_matrix = np.loadtxt(mc_path)
    mc_matrix = np.array(mc_matrix).astype(np.float32)

    n = mc_matrix.shape[0]
    mc_distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(mc_matrix[i] - mc_matrix[j])
            mc_distance_matrix[i, j] = dist
            mc_distance_matrix[j, i] = dist
    min_mcd = np.min(mc_distance_matrix)
    max_mcd = np.max(mc_distance_matrix)
    one_mcd = np.ones_like(mc_distance_matrix)
    mc_distance_matrix = one_mcd - (mc_distance_matrix - min_mcd) / (max_mcd - min_mcd)
    mc_distance_matrix = np.array(mc_distance_matrix).astype(np.float32)

    return mc_matrix, mc_distance_matrix

def get_num_dict(num_path):
    num_dict = {}
    with open(num_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key = line.rstrip().split(' ')[0]
        num = line.rstrip().split(' ')[-1]
        num_list = list(range(1, int(num)+1))
        num_dict[key] = num_list

    return num_dict

def get_graph_feat(feat_path):
    with open(feat_path, 'rb') as f:
        feat = pickle.load(f)

    return feat

def get_graph_similarity(similarity_path, selected_list):
    selected_list = [x - 1 for x in selected_list]
    simi_matrix = np.loadtxt(similarity_path, skiprows=1)
    simi_matrix = np.array(simi_matrix).astype(np.float32)
    simi_matrix = simi_matrix[np.ix_(selected_list, selected_list)]

    return simi_matrix

def get_adj_matrix(adj_path):
    adj_matrix = np.loadtxt(adj_path)
    adj_matrix = np.array(adj_matrix).astype(np.float32)

    return adj_matrix

def get_prot_feat(fprot_path):
    with open(fprot_path, 'rb') as f:
        prot:WholeProteinPocket = pickle.load(f)
    residue_feat_matrix = prot.res_feats
    residue_mc_matrix = prot.res_mcs

    return residue_feat_matrix, residue_mc_matrix

class BestMeter(object):
    """Computes and stores the best value"""
    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

class stopping(object):
    def __init__(self, patience, tolerance, repetition_th, save_path):
        dt = datetime.datetime.now()
        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.save_path = save_path
        self.repetiton_th = repetition_th
        self.best_score = None
        self.early_stop = False
        self.file_name = save_path + '/' + 'rp{}_{}_{:02d}_{:02d}_{:02d}.pth'.format(repetition_th, dt.date(), dt.hour, dt.minute, dt.second)
    
    def _check_lower(self, score, prev_best_score):
        return prev_best_score / score > 1 + self.tolerance
    
    def save_checkpoint(self, model):
        torch.save({'model_state_dict': model.state_dict()}, self.file_name)
    
    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.file_name)['model_state_dict'])
        
    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check_lower(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'Earlystopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
def average_docked_dim(out_feat):
    '''
    out_feat shape as (batch, docked, feat_dim)
    '''
    docked_dim = out_feat.size()[1]
    sum_docked_dim = torch.sum(out_feat, dim=1) # shape as (batch, feat_dim)
    result = sum_docked_dim / docked_dim
    
    return result
