import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from itertools import repeat
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from utils import *
import multiprocessing
import pickle
import torch

def tensorize_feat_matrix(label, feat_path, similarity_path, mc_path, selected_list, complex_id, fprot_path, graph_info_path):
    feat_matrix = torch.from_numpy(get_graph_feat(feat_path))
    simi_matrix = torch.from_numpy(get_graph_similarity(similarity_path, selected_list))
    mc_matrix, mcd_matrix = get_mass_center(mc_path)
    mc_matrix = torch.from_numpy(mc_matrix)
    mcd_matrix = torch.from_numpy(mcd_matrix)
    residue_feat_matrix, residue_mc_matrix = get_prot_feat(fprot_path)
    residue_feat_matrix = torch.from_numpy(residue_feat_matrix)
    residue_mc_matrix = torch.from_numpy(residue_mc_matrix)
    # print(residue_mc_matrix.shape)

    # adj_matrix = torch.from_numpy(get_adj_matrix(adj_path))
    label = torch.FloatTensor([label])

    data = {'key':complex_id, 'label':label, 'feat_matrix':feat_matrix, 'simi_matrix':simi_matrix, 'mc_matrix':mc_matrix, 'mcd_matrix': mcd_matrix, 
            'residue_feat_matrix': residue_feat_matrix, 'residue_mc_matrix': residue_mc_matrix}
    with open(os.path.join(graph_info_path, complex_id), 'wb') as f:
        pickle.dump(data, f)

class GraphConvertMul(object):
    '''
    using multiprocessing to generate model data
    '''
    def __init__(self, keys, labels, complex_path_list, selected_dict, graph_collect_path, graph_info_path, num_process=1):
        self.org_keys = keys
        self.org_labels = labels
        self.complex_path_list = complex_path_list
        self.graph_info_path = graph_info_path
        self.graph_collect_path = graph_collect_path
        self.num_process = num_process
        self.selected_dict = selected_dict
        self._pre_process()
    
    def _pre_process(self):
        if not os.path.exists(self.graph_collect_path):
            os.mkdir(self.graph_collect_path)
            os.mkdir(self.graph_info_path)

        if os.path.exists(self.graph_collect_path + '/' + 'feat_matrix_list.bin'):
            print('Loading previously saved graph information')
            with open(self.graph_collect_path + '/' + 'feat_matrix_list.bin', 'rb') as f:
                self.feat_matrix_list = pickle.load(f)
            with open(self.graph_collect_path + '/' + 'simi_matrix_list.bin', 'rb') as f:
                self.simi_matrix_list = pickle.load(f)
            with open(self.graph_collect_path + '/' + 'mc_matrix_list.bin', 'rb') as f:
                self.mc_matrix_list = pickle.load(f)
            with open(self.graph_collect_path + '/' + 'mcd_matrix_list.bin', 'rb') as f:
                self.mcd_matrix_list = pickle.load(f)
            with open(self.graph_collect_path + '/' + 'residue_feat_matrix_list.bin', 'rb') as f:
                self.residue_feat_matrix_list = pickle.load(f)
            with open(self.graph_collect_path + '/' + 'residue_mc_matrix_list.bin', 'rb') as f:
                self.residue_mc_matrix_list = pickle.load(f)
            # with open(self.graph_collect_path + '/' + 'adj_matrix_list.bin', 'rb') as f:
            #     self.adj_matrix_list = pickle.load(f)
            with open(self.graph_collect_path + '/' + 'keys.bin', 'rb') as f:
                self.keys = pickle.load(f)
            with open(self.graph_collect_path + '/' + 'labels_list.bin', 'rb') as f:
                self.labels = pickle.load(f)
        
        else:
            graph_info_path_list = repeat(self.graph_info_path, len(self.org_keys))
            feat_path_list = []
            simi_path_list = []
            mc_path_list = []
            # adj_path_list = []
            prot_path_list = []
            selected_list_list = []
            
            for complex_path in self.complex_path_list:
                cid = complex_path.split('/')[-1]
                feat_path = os.path.join(complex_path, f'{cid}_feat.pkl')
                simi_path = os.path.join(complex_path, f'myrmsd_similarity.txt')
                mc_path = os.path.join(complex_path, f'{cid}_mc.txt')
                prot_path = os.path.join(complex_path, f'{cid}_protfeat.pkl')
                # adj_path = os.path.join(complex_path, f'{cid}_adj.txt')

                feat_path_list.append(feat_path)
                simi_path_list.append(simi_path)
                mc_path_list.append(mc_path)
                prot_path_list.append(prot_path)
                # adj_path_list.append(adj_path)
                selected_list_list.append(self.selected_dict[cid])

            print('Generate complex graph')

            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(tensorize_feat_matrix, zip(self.org_labels, feat_path_list, simi_path_list, mc_path_list, selected_list_list, self.org_keys, prot_path_list, graph_info_path_list))
            pool.close()
            pool.join()

            self.feat_matrix_list = []
            self.simi_matrix_list = []
            self.mc_matrix_list = []
            self.mcd_matrix_list = []
            self.residue_feat_matrix_list = []
            self.residue_mc_matrix_list = []
            # self.adj_matrix_list = []
            self.labels = []
            self.keys = os.listdir(self.graph_info_path)

            for key in self.keys:
                with open(self.graph_info_path + '/' + key, 'rb') as f:
                    graph_info = pickle.load(f)
                    self.feat_matrix_list.append(graph_info['feat_matrix'])
                    self.simi_matrix_list.append(graph_info['simi_matrix'])
                    self.mc_matrix_list.append(graph_info['mc_matrix'])
                    self.mcd_matrix_list.append(graph_info['mcd_matrix'])
                    self.residue_feat_matrix_list.append(graph_info['residue_feat_matrix'])
                    self.residue_mc_matrix_list.append(graph_info['residue_mc_matrix'])
                    # self.adj_matrix_list.append(graph_info['adj_matrix'])
                    self.labels.append(graph_info['label'])
            
            with open(self.graph_collect_path + '/' + 'feat_matrix_list.bin', 'wb') as f:
                pickle.dump(self.feat_matrix_list, f)
            with open(self.graph_collect_path + '/' + 'simi_matrix_list.bin', 'wb') as f:
                pickle.dump(self.simi_matrix_list, f)
            with open(self.graph_collect_path + '/' + 'mc_matrix_list.bin', 'wb') as f:
                pickle.dump(self.mc_matrix_list, f)
            with open(self.graph_collect_path + '/' + 'mcd_matrix_list.bin', 'wb') as f:
                pickle.dump(self.mcd_matrix_list, f)
            with open(self.graph_collect_path + '/' + 'residue_feat_matrix_list.bin', 'wb') as f:
                pickle.dump(self.residue_feat_matrix_list, f)
            with open(self.graph_collect_path + '/' + 'residue_mc_matrix_list.bin', 'wb') as f:
                pickle.dump(self.residue_mc_matrix_list, f)
            # with open(self.graph_collect_path + '/' + 'adj_matrix_list.bin', 'wb') as f:
            #     pickle.dump(self.adj_matrix_list, f)
            with open(self.graph_collect_path + '/' + 'keys.bin', 'wb') as f:
                pickle.dump(self.keys, f)
            with open(self.graph_collect_path + '/' + 'labels_list.bin', 'wb') as f:
                pickle.dump(self.labels, f)

        cmdline = 'rm -rf %s' % (self.graph_info_path + '/' + '*')
        os.system(cmdline)
        
    def __getitem__(self, index):
        return self.keys[index], self.labels[index], self.feat_matrix_list[index], self.simi_matrix_list[index], self.mc_matrix_list[index], self.mcd_matrix_list[index], self.residue_feat_matrix_list[index], self.residue_mc_matrix_list[index]
    
    def __len__(self):
        return len(self.labels)

def my_collate_fn_fix(data_batch):
    '''
    feat_matrix shape as (docked, feat_dim)
    mc_matrix shape as (docked, 3)
    simi_matrix shape as (docked, docked)
    '''
    keys, labels, feat_matrix_list, simi_matrix_list, mc_matrix_list, mcd_matrix_list, residue_feat_matrix_list, residue_mc_matrix_list = map(list, zip(*data_batch))
    b_feat_matrix_list = torch.stack(feat_matrix_list, dim=0)
    b_simi_matrix_list = torch.stack(simi_matrix_list, dim=0)
    b_mc_matrix_list = torch.stack(mc_matrix_list, dim=0)
    b_mcd_matrix_list = torch.stack(mcd_matrix_list, dim=0)
    b_residue_feat_matrix = torch.stack(residue_feat_matrix_list, dim=0)
    b_residue_mc_matrix = torch.stack(residue_mc_matrix_list, dim=0)
    # b_adj_matrix_list = torch.stack(adj_matrix_list, dim=0)
    b_labels = torch.cat(labels)

    return b_feat_matrix_list, b_simi_matrix_list, b_mc_matrix_list, b_mcd_matrix_list, b_residue_feat_matrix, b_residue_mc_matrix, b_labels, keys

def my_collate_fn_dynamic(data_batch):
    '''
    each complex dock num must be same length
    feat_matrix shape as (same_docked, feat_dim)
    mc_matrix shape as (same_docked, 3)
    simi_matrix shape as (same_docked, docked)
    '''
    keys, labels, feat_matrix_list, simi_matrix_list, mc_matrix_list, mcd_matrix_list, residue_feat_matrix_list, residue_mc_matrix_list = map(list, zip(*data_batch))
    docked_num_list = []
    for i, v in enumerate(feat_matrix_list):
        docked_num_list.append(v.size(0))
    max_docked_num = max(docked_num_list)
    residue_num_list = []
    for i, v in enumerate(residue_mc_matrix_list):
        residue_num_list.append(v.size(0))
    max_residue_num = max(residue_num_list)

    paded_feat_matrix_list = []
    paded_simi_matrix_list = []
    paded_mc_matrix_list = []
    paded_mcd_matrix_list = []
    paded_residue_feat_matrix_list = []
    paded_residue_mc_matrix_list = []
    # paded_adj_matrix_list = []
    for i in range(len(feat_matrix_list)):
        docked_num = feat_matrix_list[i].size(0)
        residue_num = residue_mc_matrix_list[i].size(0)
        pad_docked_num = max_docked_num - docked_num
        pad_residue_num = max_residue_num - residue_num
        paded_feat_matrix = F.pad(feat_matrix_list[i], (0, 0, 0, pad_docked_num), 'constant', 0)
        paded_simi_matrix = F.pad(simi_matrix_list[i], (0, pad_docked_num, 0, pad_docked_num), 'constant', 0)
        paded_mc_matrix = F.pad(mc_matrix_list[i], (0, 0, 0, pad_docked_num), 'constant', 0)
        paded_mcd_matrix = F.pad(mcd_matrix_list[i], (0, pad_docked_num, 0, pad_docked_num), 'constant', 0)
        paded_residue_feat_matrix = F.pad(residue_feat_matrix_list[i], (0, 0, 0, pad_residue_num), 'constant', 0)
        paded_residue_mc_matrix = F.pad(residue_mc_matrix_list[i], (0, 0, 0, pad_residue_num), 'constant', 0)
        # paded_adj_matrix = F.pad(adj_matrix_list[i], (0, pad_docked_num, 0, pad_docked_num), 'constant', 0)

        paded_feat_matrix_list.append(paded_feat_matrix)
        paded_simi_matrix_list.append(paded_simi_matrix)
        paded_mc_matrix_list.append(paded_mc_matrix)
        paded_mcd_matrix_list.append(paded_mcd_matrix)
        paded_residue_feat_matrix_list.append(paded_residue_feat_matrix)
        paded_residue_mc_matrix_list.append(paded_residue_mc_matrix)
        # paded_adj_matrix_list.append(paded_adj_matrix)
    
    b_feat_matrix_list = torch.stack(paded_feat_matrix_list, dim=0)
    b_simi_matrix_list = torch.stack(paded_simi_matrix_list, dim=0)
    b_mc_matrix_list = torch.stack(paded_mc_matrix_list, dim=0)
    b_mcd_matrix_list = torch.stack(paded_mcd_matrix_list, dim=0)
    b_residue_feat_matrix_list = torch.stack(paded_residue_feat_matrix_list)
    b_residue_mc_matrix_list = torch.stack(paded_residue_mc_matrix_list)
    # b_adj_matrix_list = torch.stack(paded_adj_matrix_list, dim=0)
    b_labels = torch.cat(labels)

    return b_feat_matrix_list, b_simi_matrix_list, b_mc_matrix_list, b_mcd_matrix_list, b_residue_feat_matrix_list, b_residue_mc_matrix_list, b_labels, keys

if __name__ == '__main__':
    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())
    num_lst = [6]
    for num in num_lst:
        data_root = f'/disk1/my_project_final/data32/in_data_{num}'
        set_list = ['train', 'valid']
        for n in set_list:
            for i in range(1, 6):
                set_name = f'{n}_{i}'
                os.mkdir(os.path.join(data_root, f'{set_name}_in'))
                graph_info_path = os.path.join(data_root, f'{set_name}_in', 'graph_info')
                graph_collect_path = os.path.join(data_root, f'{set_name}_in', 'graph_collect')
                set_path = os.path.join(data_root, set_name)
                selected_path = os.path.join(set_path, 'num_dock.txt')
                data_csv = os.path.join(set_path, f'{n}.csv')

                selected_dict = get_num_dict(selected_path)
                data_df = pd.read_csv(data_csv)
                test_keys = list(data_df['cid'])
                test_labels = list(data_df['pkoff'])
                complex_path_list = []
                for key in test_keys:
                    complex_path_list.append(os.path.join(set_path, key))
                test_set = GraphConvertMul(keys=test_keys, labels=test_labels, complex_path_list=complex_path_list, selected_dict=selected_dict, graph_collect_path=graph_collect_path, graph_info_path=graph_info_path, num_process=1)
                test_dataloader = DataLoaderX(test_set, 5, shuffle=True, num_workers=0, collate_fn=my_collate_fn_dynamic)


        set_list = ['focal', 'p38', 'crf1']
        for set_name in set_list:
            os.mkdir(os.path.join(data_root, f'{set_name}_in'))
            graph_info_path = os.path.join(data_root, f'{set_name}_in', 'graph_info')
            graph_collect_path = os.path.join(data_root, f'{set_name}_in', 'graph_collect') 
            set_path = os.path.join(data_root, set_name)
            selected_path = os.path.join(set_path, 'num_dock.txt')
            data_csv = os.path.join(set_path, f'{set_name}.csv')

            selected_dict = get_num_dict(selected_path)
            data_df = pd.read_csv(data_csv)
            test_keys = list(data_df['cid'])
            test_labels = list(data_df['pkoff'])
            complex_path_list = []
            for key in test_keys:
                complex_path_list.append(os.path.join(set_path, key))
            test_set = GraphConvertMul(keys=test_keys, labels=test_labels, complex_path_list=complex_path_list, selected_dict=selected_dict, graph_collect_path=graph_collect_path, graph_info_path=graph_info_path, num_process=1)
            test_dataloader = DataLoaderX(test_set, 5, shuffle=True, num_workers=0, collate_fn=my_collate_fn_dynamic)
    print(f'{num} finished')












