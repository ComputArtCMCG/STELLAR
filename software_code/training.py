import os
import torch
import random
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import datetime
from utils import *
from feat_aggregate_p import *
from module2 import *
from module3 import *
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_a_train_epochV1(model, loss_fn, train_dataloader, optimizer, device):
    model.train()
    for i, batch in enumerate(train_dataloader):
        model.zero_grad()
        b_fconfor, b_simi, b_mc, b_mcd, b_fprot, b_pmc, ys, _ = batch
        b_fconfor, b_simi, b_mc, b_mcd, b_fprot, b_pmc, ys = b_fconfor.to(device), b_simi.to(device), b_mc.to(device), b_mcd.to(device), b_fprot.to(device), b_pmc.to(device), ys.to(device)
        outputs = model(b_fconfor, b_fprot, b_mc, b_pmc, b_simi)
        loss = loss_fn(outputs, ys)
        loss.backward()
        optimizer.step()

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

def compute_metrics(true, pred):
    true = np.asarray(true).ravel()
    pred = np.asarray(pred).ravel()
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    r2 = float(r2_score(true, pred))
    mae = float(mean_absolute_error(true, pred))
    if true.std() < 1e-12 or pred.std() < 1e-12:
        rp = float('nan')
    else:
        rp = float(np.corrcoef(true, pred)[0, 1])
    return rmse, r2, mae, rp

def evaluate_and_report(model, dataloader, device, set_name, repetition_th, model_name, dt,
                        csv_dir='./conclusion_cross_n', tag=''):
    """评估一个集合：打印 rmse / r2 / mae / pearson，并导出预测 csv。"""
    true_list, pred_list, keys = run_a_eval_epochV1(model, dataloader, device)
    key_list = sum(keys, [])
    true = np.concatenate(true_list, 0)
    pred = np.concatenate(pred_list, 0)
    rmse, r2, mae, rp = compute_metrics(true, pred)

    msg = ("rp%d, set:%s, rmse:%.4f \t r2:%.4f \t mae:%.4f \t rp:%.4f" % (
        repetition_th, set_name, rmse, r2, mae, rp))
    print(msg)

    if csv_dir is not None:
        pd_te = pd.DataFrame({'keys': key_list, 'test_true': true.ravel(), 'test_pred': pred.ravel()})
        pd_te.to_csv('{}/{}_rp{}_{}_{:02d}_{:02d}_{}.csv'.format(
            csv_dir, model_name, repetition_th, dt.date(), dt.hour, dt.minute, tag))
    return rmse, r2, mae, rp

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpuid', type=str, default='0', help="gpu id for training model")
    argparser.add_argument('--lr', type=float, default=10 ** -4.0, help="Learning rate")
    argparser.add_argument('--epochs', type=int, default=2000, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=24, help="Batch size")
    argparser.add_argument('--repetitions', type=int, default=1, help="the number of independent runs")
    argparser.add_argument('--fconfor_dim', type=int, default=256)
    argparser.add_argument('--fprot_dim', type=int, default=20)
    argparser.add_argument('--hidden_dim_c', type=int, default=128)
    argparser.add_argument('--hidden_dim_p', type=int, default=32)
    argparser.add_argument('--simi_threshold', type=float, default=0.3)
    argparser.add_argument('--head_num', type=int, default=1)
    argparser.add_argument('--d_FC_layer', type=int, default=128, help='the number of hidden layers of FC')
    argparser.add_argument('--n_FC_layer', type=int, default=2, help='the number of hidden layers of task networks')
    argparser.add_argument('--dropout', type=float, default=0.0)
    argparser.add_argument('--n_tasks', type=int, default=1)
    argparser.add_argument('--num_workers', type=int, default=0, help='number of workers for loading data in Dataloader')
    argparser.add_argument('--num_process', type=int, default=12, help='number of process for convert graph')
    argparser.add_argument('--home_path', type=str, help='data root dir of 5 fold validcross')
    argparser.add_argument('--model_save_path', type=str)
    args = argparser.parse_args()

    gpuid, lr, epochs, batch_size, num_workers, repetitions = args.gpuid, args.lr, args.epochs, args.batch_size, args.num_workers, args.repetitions
    fconfor_dim, fprot_dim, simi_threshold, hidden_dim_c, hidden_dim_p, head_num = args.fconfor_dim, args.fprot_dim, args.simi_threshold, args.hidden_dim_c, args.hidden_dim_p, args.head_num
    n_FC_layer, d_FC_layer, dropout, n_tasks = args.n_FC_layer, args.d_FC_layer, args.dropout, args.n_tasks
    home_path, num_process, model_save_path = args.home_path, args.num_process, args.model_save_path

    # 本脚本只做 5 折交叉验证（train / valid / focal / p38）。
    # cdk2 与 test 不参与训练过程，它们用某一折保存下来的权重单独评估，
    # 见 predict_koff.py（随包发布的 rp0_2024-03-20_06_45_37.pth 即其中一折）。
    eval_set_names = ['train', 'valid', 'focal', 'p38']
    fold_results = []          # [(cross_num, set_name, rmse, r2, mae, rp), ...]

    for cross_num in range(1, 6):
        train_path = os.path.join(home_path, f'train_{cross_num}')
        valid_path = os.path.join(home_path, f'valid_{cross_num}')
        focal_path = os.path.join(home_path, f'focal')
        p38_path = os.path.join(home_path, f'p38')
        train_data = pd.read_csv(os.path.join(train_path, 'train.csv'))
        valid_data = pd.read_csv(os.path.join(valid_path, 'valid.csv'))
        focal_data = pd.read_csv(os.path.join(focal_path, 'focal.csv'))
        p38_data = pd.read_csv(os.path.join(p38_path, 'p38.csv'))
        # train_data
        train_complex_paths = []
        train_keys = list(train_data['cid'])
        train_labels = list(train_data['pkoff'])
        for key in train_keys:
            train_complex_paths.append(os.path.join(train_path, key))
        train_selected_dict = get_num_dict(os.path.join(train_path, 'num_dock.txt'))
        
        # valid_data
        valid_complex_paths = []
        valid_keys = list(valid_data['cid'])
        valid_labels = list(valid_data['pkoff'])
        for key in valid_keys:
            valid_complex_paths.append(os.path.join(valid_path, key))
        valid_selected_dict = get_num_dict(os.path.join(valid_path, 'num_dock.txt'))
        
        # external_data
        focal_complex_paths = []
        focal_keys = list(focal_data['cid'])
        focal_labels = list(focal_data['pkoff'])
        for key in focal_keys:
            focal_complex_paths.append(os.path.join(focal_path, key))
        focal_selected_dict = get_num_dict(os.path.join(focal_path, 'num_dock.txt'))

        p38_complex_paths = []
        p38_keys = list(p38_data['cid'])
        p38_labels = list(p38_data['pkoff'])
        for key in p38_keys:
            p38_complex_paths.append(os.path.join(p38_path, key))
        p38_selected_dict = get_num_dict(os.path.join(p38_path, 'num_dock.txt'))

        # generate dataset
        # collate 固定用 dynamic：各复合物的对接构象数不同（本次发布的数据为 69~250），
        # my_collate_fn_fix 要求同一 batch 内张量等长，会直接报错。
        my_collate_fn = my_collate_fn_dynamic

        train_dataset = GraphConvertMul(keys=train_keys, labels=train_labels, complex_path_list=train_complex_paths, selected_dict=train_selected_dict, 
                                        graph_info_path=f'./new_train_data_5/train_{cross_num}_in/graph_info', graph_collect_path=f'./new_train_data_5/train_{cross_num}_in/graph_collect', 
                                        num_process=num_process)
        
        valid_dataset = GraphConvertMul(keys=valid_keys, labels=valid_labels, complex_path_list=valid_complex_paths, selected_dict=valid_selected_dict, 
                                        graph_info_path=f'./new_train_data_5/valid_{cross_num}_in/graph_info', graph_collect_path=f'./new_train_data_5/valid_{cross_num}_in/graph_collect', 
                                        num_process=num_process)
        
        focal_dataset = GraphConvertMul(keys=focal_keys, labels=focal_labels, complex_path_list=focal_complex_paths, selected_dict=focal_selected_dict, 
                                        graph_info_path=f'./new_train_data_5/focal_in/graph_info', graph_collect_path=f'./new_train_data_5/focal_in/graph_collect', 
                                        num_process=num_process)
        
        p38_dataset = GraphConvertMul(keys=p38_keys, labels=p38_labels, complex_path_list=p38_complex_paths, selected_dict=p38_selected_dict, 
                                        graph_info_path=f'./new_train_data_5/p38_in/graph_info', graph_collect_path=f'./new_train_data_5/p38_in/graph_collect', 
                                        num_process=num_process)
        
        for repetition_th in range(repetitions):
            set_random_seed(repetition_th)
            print('=' * 78)
            print('cross_num = %d | repetition = %d' % (cross_num, repetition_th))
            print('num of trainset:', len(train_dataset))
            print('num of validset:', len(valid_dataset))
            print('num of focalset:', len(focal_dataset))
            print('num of p38set:', len(p38_dataset))

            train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=my_collate_fn)
            valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=my_collate_fn)
            focal_dataloader = DataLoaderX(focal_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=my_collate_fn)
            p38_dataloader = DataLoaderX(p38_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=my_collate_fn)

            # set model
            model_name = f'ProtEGNN_cross_ConforEGNN_edge_fc2_lr1e-4_dr0.0_32p_2000_cross{cross_num}'
            my_model = ProtEGNN_cross_ConforEGNN_edge(fconfor_dim=fconfor_dim, fprot_dim=fprot_dim, hidden_dim_c=hidden_dim_c, hidden_dim_p=hidden_dim_p, d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout, 
                                n_tasks=n_tasks)
            
            print('num of parameters:', sum(p.numel() for p in my_model.parameters() if p.requires_grad))
            if repetition_th == 0:
                print(my_model)
            device = torch.device('cuda:%s' % gpuid)
            my_model.to(device)
            # my_model = nn.DataParallel(my_model, device_ids=[0, 1])
            optimizer = torch.optim.Adam(my_model.parameters(), lr=lr, weight_decay=1e-6)
            loss_fn = nn.MSELoss()

            # run and record
            dt = datetime.datetime.now()
            log_file = './log_cross/{}_rp{}_{}_{:02d}_{:02d}.txt'.format(model_name, repetition_th, dt.date(), dt.hour, dt.minute)
            stopper = stopping(patience=1000, tolerance=0.0, repetition_th=repetition_th, save_path=model_save_path)
            print('training begin')
            for epoch in range(epochs):
                # train
                run_a_train_epochV1(my_model, loss_fn, train_dataloader, optimizer, device)

                # 每个 epoch 只看 train / valid（不碰 test、不碰 cdk2）
                train_true, train_pred, _ = run_a_eval_epochV1(my_model, train_dataloader, device)
                valid_true, valid_pred, _ = run_a_eval_epochV1(my_model, valid_dataloader, device)

                train_true = np.concatenate(train_true, 0)
                train_pred = np.concatenate(train_pred, 0)

                valid_true = np.concatenate(valid_true, 0)
                valid_pred = np.concatenate(valid_pred, 0)

                train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
                valid_rmse = np.sqrt(mean_squared_error(valid_true, valid_pred))
                valid_pr = np.corrcoef(valid_pred, valid_true)[0, 1]

                msg = "epoch-%d, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" % (
                    epoch, train_rmse, valid_rmse, valid_pr)

                with open(log_file, 'a') as f:
                    f.write(f'{msg}\n')
                print(msg)

                early_stop = stopper.step(valid_rmse, my_model)
                if early_stop:
                    break
            
            stopper.load_checkpoint(my_model)

            # ---- 训练结束后：评估 train / valid / focal / p38（不含 test、cdk2） ----
            print('-' * 78)
            print('training finished: %s (cross_num=%d)' % (model_name, cross_num))
            for set_name in eval_set_names:
                if set_name == 'train':
                    dataloader = train_dataloader
                elif set_name == 'valid':
                    dataloader = valid_dataloader
                elif set_name == 'focal':
                    dataloader = focal_dataloader
                else:
                    dataloader = p38_dataloader

                tag = {'train': 'tr', 'valid': 'va', 'focal': 'fo', 'p38': 'p38'}[set_name]
                rmse, r2, mae, rp = evaluate_and_report(my_model, dataloader, device, set_name,
                                                        repetition_th, model_name, dt, tag=tag)
                fold_results.append((cross_num, set_name, rmse, r2, mae, rp))

                with open(f'./stats_n/{model_name}', 'a') as f:
                    f.write('rp%d, cross%d, set:%s, rmse:%.4f \t r2:%.4f \t mae:%.4f \t rp:%.4f\n' % (
                        repetition_th, cross_num, set_name, rmse, r2, mae, rp))

    # ---- 5 折汇总（mean ± std） ----
    if len(fold_results) > 0:
        print()
        print('=' * 78)
        print('5-fold cross-validation summary (mean +- std over folds)')
        print('note: cdk2 / test are NOT part of this table; they are evaluated')
        print('      separately with one fold\'s saved weights via predict_koff.py')
        print('=' * 78)
        print('%-8s %6s  %-17s %-17s %-17s %-17s' % ('set', 'folds', 'rmse', 'r2', 'mae', 'pearson'))
        for set_name in eval_set_names:
            rows = [r for r in fold_results if r[1] == set_name]
            if not rows:
                continue
            n = len(rows)
            def ms(vals):
                m = float(np.nanmean(vals))
                s = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0
                return m, s
            rmse_m, rmse_s = ms([r[2] for r in rows])
            r2_m, r2_s = ms([r[3] for r in rows])
            mae_m, mae_s = ms([r[4] for r in rows])
            rp_m, rp_s = ms([r[5] for r in rows])
            print('%-8s %6d  %6.4f+-%.4f  %6.4f+-%.4f  %6.4f+-%.4f  %6.4f+-%.4f' % (
                set_name, n, rmse_m, rmse_s, r2_m, r2_s, mae_m, mae_s, rp_m, rp_s))
        print('=' * 78)
