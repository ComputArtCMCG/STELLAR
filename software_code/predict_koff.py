import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from utils import *
from feat_aggregate_p import *
from module2 import *
from module3 import *
from torch.utils.data import DataLoader

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def run_a_eval_epochV1(model, validation_dataloader, device):
    true = []
    pred = []
    key_list = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(validation_dataloader):
            b_fconfor, b_simi, b_mc, b_mcd, b_fprot, b_pmc, ys, keys = batch
            b_fconfor, b_simi, b_mc, b_mcd, b_fprot, b_pmc, ys = b_fconfor.to(device), b_simi.to(device), b_mc.to(device), b_mcd.to(device), b_fprot.to(device), b_pmc.to(device), ys.to(device)
            outputs = model(b_fconfor, b_fprot, b_mc, b_pmc, b_simi)
            true.append(ys.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
            key_list.append(keys)

    return true, pred, key_list

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', type=str, help='data root of dir need to pred')
    argparser.add_argument('--input_csv', type=str, help='csv file of complex_name')
    argparser.add_argument('--output_path', type=str, help='pred output dir')
    args = argparser.parse_args()
    prep_path, prep_csv, output_path = args.input_path, args.input_csv, args.output_path

    batch_size=16
    gpuid='0'
    fconfor_dim=256
    fprot_dim=20
    hidden_dim_c=128
    hidden_dim_p=32
    d_FC_layer=128
    n_FC_layer=2
    dropout=0.0
    n_tasks=1
    num_workers=0
    num_process=12

    test_data = pd.read_csv(prep_csv)
    test_complex_paths = []
    test_labels = []
    test_keys = list(test_data['cid'])

    for key in test_keys:
        test_complex_paths.append(os.path.join(prep_path, key))
        test_labels.append(0)
    test_selected_dict = get_num_dict(os.path.join(prep_path, 'num_dock.txt'))

    # collate 固定用 dynamic（各复合物对接构象数不同）
    my_collate_fn = my_collate_fn_dynamic

    in_path = os.path.join(prep_path, 'data_in')
    graph_info_path = f'{prep_path}/data_in/graph_info'
    graph_collect_path = f'{prep_path}/data_in/graph_collect'
    test_dataset = GraphConvertMul(keys=test_keys, labels=test_labels, complex_path_list=test_complex_paths, selected_dict=test_selected_dict, 
                                        graph_info_path=f'{graph_info_path}', graph_collect_path=f'{graph_collect_path}', 
                                        num_process=num_process)

    test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=my_collate_fn)

    # model_name = 'ProtEGNN_cross_ConforEGNN_edge_fc3_lr1e-4_dr0.0_32p_cross0'
    my_model = ProtEGNN_cross_ConforEGNN_edge(fconfor_dim=fconfor_dim, fprot_dim=fprot_dim, hidden_dim_c=hidden_dim_c, hidden_dim_p=hidden_dim_p, d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout, 
                                n_tasks=n_tasks)
    
    device = torch.device('cuda:%s' % gpuid)
    my_model.to(device)
    for n, model_name in enumerate(['rp0_2024-03-20_06_45_37']):
        model_path = f'./model_save_cross/{model_name}.pth'
        my_model.load_state_dict(torch.load(model_path)['model_state_dict'])

        test_true, test_pred, keys = run_a_eval_epochV1(my_model, test_dataloader, device)
        key_list = sum(keys, [])
        test_true = np.concatenate(test_true, 0)
        test_pred = np.concatenate(test_pred, 0)
        pd_te = pd.DataFrame({'complex_name': key_list, 'true': test_true, 'pred': test_pred})
        pd_te.to_csv(os.path.join(output_path, f'prediction.csv'))
