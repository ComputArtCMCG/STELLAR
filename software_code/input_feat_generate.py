import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from GIGN import GIGN
from dataset_GIGN import GraphDataset, PLIDataLoader
from utils import *
import multiprocessing
import argparse
import pymol
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# generate complex file of GIGN
# data_dir: path of complex
# complex_name
def generate_pocket(data_dir, complex_name, selected_docked_id_list, distance=5):
    for docked_id in selected_docked_id_list:
        docked_path = os.path.join(data_dir, docked_id)
        ligand_path = os.path.join(docked_path, f"{docked_id}_ligand.mol2")
        protein_path= os.path.join(docked_path, f"{complex_name}_protein.pdb")

        if os.path.exists(os.path.join(docked_path, f'Pocket_{distance}A.pdb')):
            continue

        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(ligand_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {docked_id}_ligand around {distance}')
        pymol.cmd.save(os.path.join(docked_path, f'Pocket_{distance}A.pdb'), 'Pocket')
        pymol.cmd.delete('all')

def generate_complex(data_dir, selected_docked_id_list, distance=5, input_ligand_format='mol2'):
    pbar = tqdm(total=len(selected_docked_id_list))
    for docked_id in selected_docked_id_list:
        docked_path = os.path.join(data_dir, docked_id)
        pocket_path = os.path.join(data_dir, docked_id, f'Pocket_{distance}A.pdb')
        if input_ligand_format != 'pdb':
            ligand_input_path = os.path.join(docked_path, f'{docked_id}_ligand.{input_ligand_format}')
            ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
            os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
        else:
            ligand_path = os.path.join(docked_path, f'{docked_id}_ligand.pdb')

        save_path = os.path.join(docked_path, f"{docked_id}_{distance}A.rdkit")
        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
        if ligand == None:
            print(f"Unable to process ligand of {docked_id}")
            with open('/disk1/my_project/energy_screen/error.txt', 'a') as f:
                f.write(f'{docked_id} ligand')

        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
        if pocket == None:
            print(f"Unable to process protein of {docked_id}")
            with open('/disk1/my_project/energy_screen/error.txt', 'a') as f:
                f.write(f'{docked_id} pocket')

        complex = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)
        
        pbar.update(1)

def generate_input_file(data_dir, complex_name, selected_list):
    selected_docked_id_list = []
    for dock_index in selected_list:
        selected_docked_id_list.append(f'{complex_name}_mc_{dock_index}')
    generate_pocket(data_dir, complex_name, selected_docked_id_list)
    generate_complex(data_dir, selected_docked_id_list)

# get GIGN feature
# gpuid, batch_size, input_csv, input_path
def get_GIGN_feature(selected_dict, input_csv, input_path):
    graph_type = 'Graph_GIGN'
    layer_name = 'fc'
    batch_size = 50
    feat_in_list = []

    def hook(module, feat_in, feat_out):
        # feat_out_list.append(feat_out)
        # feat_in_list.append(feat_in)
        feat_in = feat_in[0].detach().cpu().numpy()
        feat_in_list.append(feat_in)

    input_df = pd.read_csv(input_csv)
    input_set = GraphDataset(input_path, input_df, selected_dict, graph_type=graph_type, create=True)
    input_loader = PLIDataLoader(input_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda:0')
    base_model = GIGN(35, 256).to(device)
    load_model_dict(base_model, '/disk1/my_project/energy_screen/model/20230721_165213_GIGN_repeat2/model/epoch-322, train_loss-0.1589, train_rmse-0.3986, valid_rmse-1.1543, valid_pr-0.7814.pt')
    base_model = base_model.cuda()

    for (name, module) in base_model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook=hook)
    base_model.eval()

    key_list = []
    for data in input_loader:
        keys, x, edge_index_intra, edge_index_inter, pos, batch = data.key, data.x, data.edge_index_intra, data.edge_index_inter, data.pos, data.batch
        x, edge_index_intra, edge_index_inter, pos, batch = x.to(device), edge_index_intra.to(device), edge_index_inter.to(device), pos.to(device), batch.to(device)
        pred = base_model(x, edge_index_intra, edge_index_inter, pos, batch)
        for key in keys:
            key_list.append(key)

    feat_in = np.concatenate(feat_in_list, axis=0)
    GIGN_Feat_Dataset = GIGNfeatcollect(key_list, feat_in)

    for i, row in input_df.iterrows():
        cid = row['keys']
        selected_list = selected_dict[cid]
        feat_list = []
        for j in selected_list:
            cid_docked = cid + '_mc_' + f'{j}'
            feat_list.append(GIGN_Feat_Dataset[cid_docked])
        feat_array = np.stack(feat_list, axis=0)
        save_path = os.path.join(input_path, cid, f'{cid}_feat.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(feat_array, f)

def input_feat_generate_workflow(set_path, csv_path, num_process):
    selected_path = os.path.join(set_path, 'num_dock.txt')
    # selected_dict, main_node_dict = get_select_dict(selected_path)
    selected_dict = get_num_dict(selected_path)
    data_df = pd.read_csv(csv_path)
    keys = list(data_df['keys'])
    selected_list_list = []
    complex_path_list = []
    for key in keys:
        complex_path_list.append(os.path.join(set_path, key))
        selected_list_list.append(selected_dict[key])
    get_GIGN_feature(selected_dict, csv_path, set_path)

    pool = multiprocessing.Pool(num_process)
    pool.starmap(generate_input_file, zip(complex_path_list, keys, selected_list_list))
    pool.close()
    pool.join()

if __name__=='__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--set_path', type=str)
    argparse.add_argument('--num_process', type=int, default=12)
    args = argparse.parse_args()
    set_path, num_process = args.set_path, args.num_process
    set_name = set_path.split('/')[-1]
    csv_path = f'{set_path}/{set_name}.csv'

    input_feat_generate_workflow(set_path, csv_path, num_process)




