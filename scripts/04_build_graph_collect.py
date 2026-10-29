#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_build_graph_collect.py
=========================
Bundle the per-complex feature files produced by input_feat_generate.py into the
8 pickles that training.py / predict_koff.py load.

For each complex <cid> the following files must already exist in --input_path/<cid>/ :
    <cid>_feat.pkl           (K, 256) GIGN conformer features
    <cid>_mc.txt             (K, 3)   mass centres
    myrmsd_similarity.txt    N x N     AutoRMSD similarity
    <cid>_protfeat.pkl       protein residue features

Output (this is exactly the layout predict_koff.py expects):
    <out_dir>/data_in/graph_collect/{feat,simi,mc,mcd,residue_feat,residue_mc}_matrix_list.bin
    <out_dir>/data_in/graph_collect/{keys,labels_list}.bin

Example
-------
    python software_code/../scripts/04_build_graph_collect.py \
        --input_path <dataset_dir> --input_csv <dataset_dir>/<name>.csv --out_dir <dataset_dir>
"""
from __future__ import print_function

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', 'software_code')))

import pandas as pd
from utils import get_num_dict
from feat_aggregate_p import GraphConvertMul


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_path', required=True, help='目录: 每个复合物一个子文件夹')
    ap.add_argument('--input_csv', required=True, help='csv, 需含 cid 与 pkoff 两列')
    ap.add_argument('--out_dir', required=True, help='输出目录, 会生成 <out_dir>/data_in/graph_collect/')
    ap.add_argument('--num_process', type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    keys = [str(x).strip() for x in df['cid']]
    labels = [float(x) for x in df['pkoff']]
    complex_paths = [os.path.join(args.input_path, k) for k in keys]
    selected_dict = get_num_dict(os.path.join(args.input_path, 'num_dock.txt'))

    missing = [k for k in keys if k not in selected_dict]
    if missing:
        sys.exit('num_dock.txt 里缺这些 cid: %s' % missing[:5])

    gc = os.path.join(args.out_dir, 'data_in', 'graph_collect')
    gi = os.path.join(args.out_dir, 'data_in', 'graph_info')
    os.makedirs(gc, exist_ok=True)
    os.makedirs(gi, exist_ok=True)

    print('input_path : %s' % args.input_path)
    print('csv        : %s (%d complexes)' % (args.input_csv, len(keys)))
    print('num_dock   : %d complexes' % len(selected_dict))
    print('out        : %s' % gc)

    ds = GraphConvertMul(keys=keys, labels=labels, complex_path_list=complex_paths,
                         selected_dict=selected_dict, graph_collect_path=gc,
                         graph_info_path=gi, num_process=args.num_process)

    print('\ndone: %d samples' % len(ds))
    for f in sorted(os.listdir(gc)):
        print('  %-32s %8.1f MB' % (f, os.path.getsize(os.path.join(gc, f)) / 1024.0 / 1024.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
