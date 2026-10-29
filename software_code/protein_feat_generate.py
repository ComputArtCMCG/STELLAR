import os
import argparse
import openbabel
import pandas as pd
import numpy as np
import rdkit
import rdkit.Chem as Chem
import multiprocessing
from utils import *

periodic_table = Chem.GetPeriodicTable()
BLOSUM62 = {
    'ALA':np.array([[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0]],dtype=np.float32),
    'ARG':np.array([[-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3]],dtype=np.float32),
    'ASN':np.array([[-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3]],dtype=np.float32),
    'ASP':np.array([[-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3]],dtype=np.float32),
    'CYS':np.array([[ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1]],dtype=np.float32),
    'GLN':np.array([[-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2]],dtype=np.float32),
    'GLU':np.array([[-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2]],dtype=np.float32),
    'GLY':np.array([[ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3]],dtype=np.float32),
    'HIS':np.array([[-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3]],dtype=np.float32),
    'ILE':np.array([[-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3]],dtype=np.float32),
    'LEU':np.array([[-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1]],dtype=np.float32),
    'LYS':np.array([[-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2]],dtype=np.float32),
    'MET':np.array([[-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1]],dtype=np.float32),
    'PHE':np.array([[-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1]],dtype=np.float32),
    'PRO':np.array([[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2]],dtype=np.float32),
    'SER':np.array([[ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2]],dtype=np.float32),
    'THR':np.array([[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0]],dtype=np.float32),
    'TRP':np.array([[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3]],dtype=np.float32),
    'TYR':np.array([[-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1]],dtype=np.float32),
    'VAL':np.array([[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4]],dtype=np.float32)
}
PROTEIN_ELEMENTS = ['H','C','N','O','S','P','Se']

def get_ligand_atom_coords(lig_path):
    conv = openbabel.OBConversion()
    mol = openbabel.OBMol()
    conv.ReadFile(mol,lig_path)
    mass = []
    data2 = []
    for atom in openbabel.OBMolAtomIter(mol):
        x = atom.GetX()
        y = atom.GetY()
        z = atom.GetZ()
        data2.append([x,y,z])
        atm_mass = atom.GetAtomicMass()
        mass.append(atm_mass)
    lig_coord = np.array(data2, dtype=np.float32)

    return lig_coord

def near_pocket(pdb_line, lig_coord):
    coordinates = pdb_line[30:55]
    x = float(coordinates[0:8])
    y = float(coordinates[8:16])
    z = float(coordinates[16:24])
    distances = np.sqrt(np.sum(np.square(lig_coord - np.array([x, y, z], dtype=np.float32)), axis=-1))
    if np.all(distances > 5.0):
        return False
    else:
        return True

def mass_center_from_pdb(pdb_lines):
    pdb_lines = [line for line in pdb_lines if line[76:78].strip().capitalize() in PROTEIN_ELEMENTS]
    atoms_mass = np.reshape(np.array([periodic_table.GetAtomicWeight(line[76:78].strip().capitalize()) for line in pdb_lines]),[-1,1])
    coordinates = np.array([
        [float(line[30:38]),float(line[38:46]),float(line[46:54])] 
        for line in pdb_lines],dtype=np.float32)
    mass_center = np.sum(coordinates * atoms_mass,axis=0,keepdims=True) / np.sum(atoms_mass)

    return mass_center, coordinates

class Residue():
    def __init__(self, residue_content):
        residue_content = self.process_alternate(residue_content)
        self.mass_center,self.coordinates = mass_center_from_pdb(residue_content)
        residue_type = residue_content[0][17:20]
        if residue_type == 'MSE':
            residue_type = 'MET'
        self.residue_type = residue_type
        self.feature = BLOSUM62[self.residue_type]
    
    def get_feature(self):
        return self.feature

    def get_mass_center(self):
        return self.mass_center

    def process_alternate(self,contents):
        alternate_flag = list(set([line[16] for line in contents if line[16]!=" "]))
        if len(alternate_flag) > 0:
            contents = [line for line in contents if line[16]==alternate_flag[0] or line[16]==" "]
        return contents

class WholeProteinPocket():
    def __init__(self, complex_path, complex_name, num_dock_lst):
        self.select_pocket_residues(complex_path, complex_name, num_dock_lst)
        self.res_feats = np.concatenate([residue.get_feature() for residue in self.pocket_residues], axis=0)
        self.res_mcs = np.concatenate([residue.get_mass_center() for residue in self.pocket_residues], axis=0)

    def select_pocket_residues(self, complex_path, complex_name, num_dock_lst):
        self.pocket_residues = []
        all_conformation_pocket_res_id = []
        pdb_path = os.path.join(complex_path, f'{complex_name}_protein.pdb')
        with open(pdb_path, 'r') as f:
            pdb_content = [line.strip() for line in f if line.startswith('ATOM')]
        for n in num_dock_lst:
            lig_path = os.path.join(complex_path, f'{complex_name}_mc_{n}', f'{complex_name}_mc_{n}_ligand.mol2')
            lig_coord = get_ligand_atom_coords(lig_path)
            pocket_atoms_content = [line for line in pdb_content if near_pocket(line, lig_coord)]
            for atom_line in pocket_atoms_content:
                res_num, chain_id = str(atom_line[22: 27]).strip(), atom_line[21]
                all_conformation_pocket_res_id.append((res_num, chain_id))
        all_conformation_pocket_res_id = set(all_conformation_pocket_res_id)

        for (index, chain) in all_conformation_pocket_res_id:
            try:
                residue_content = [line for line in pdb_content if str(line[22:27]).strip()==index and line[21] == chain]
                self.pocket_residues.append(Residue(residue_content))
            except:
                continue

def protein_feat_generate_workflow(set_path, key):
    complex_path = os.path.join(set_path, key)
    num_dock_path = os.path.join(set_path, 'num_dock.txt')
    output_path = os.path.join(set_path, key, f'{key}_protfeat.pkl')
    num_dict = get_num_dict(num_dock_path)
    num_lst = num_dict[key]
    try:
        whole_protein_pocket = WholeProteinPocket(complex_path, key, num_lst)
        with open(output_path, 'wb') as f:
            pickle.dump(whole_protein_pocket, f)
    except:
        print(key)

if __name__=='__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--input_path', type=str)
    argparse.add_argument('--input_csv', type=str)
    argparse.add_argument('--num_process', type=int, default=12)
    args = argparse.parse_args()
    input_path, csv_path, num_process = args.input_path, args.input_csv, args.num_process

    key_lst = list(pd.read_csv(csv_path)['cid'])
    set_path_lst = []
    for key in key_lst:
        set_path_lst.append(input_path)
    
    pool = multiprocessing.Pool(num_process)
    pool.starmap(protein_feat_generate_workflow, zip(set_path_lst, key_lst))
    pool.close()
    pool.join()
    


