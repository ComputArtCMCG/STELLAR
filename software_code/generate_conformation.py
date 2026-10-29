import openbabel
import shutil
import pandas as pd
import numpy as np
import pybel
import os
import multiprocessing
import argparse

MGL_BIN_DIR = '/disk1/vina/mgltools_x86_64Linux2_1.5.6/bin'
pythonsh = os.path.join(MGL_BIN_DIR, 'pythonsh')
UTILS_PATH = '/disk1/vina/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/'
prep_ligand = os.path.join(UTILS_PATH, 'prepare_ligand4.py')
prep_receptor = os.path.join(UTILS_PATH, 'prepare_receptor4.py')

# generate docking pdbqt file
# in: complex file path
def ligand2pdbqt(file):
    mol_name = file.split('/')[-1]
    mol2_mol_path = '%s/%s_ligand.mol2' % (file, mol_name)
    pdbqt_mol_path = '%s/%s_ligand.pdbqt' % (file, mol_name)
    cmd = '%s %s -l %s -o %s -U \'nphs\' > %s_out1.tmp' % (pythonsh, prep_ligand, mol2_mol_path, pdbqt_mol_path, mol_name)
    os.system(cmd)

def protein2pdbqt(file):
    mol_name = file.split('/')[-1]
    pdb_mol_path = '%s/%s_protein.pdb' % (file, mol_name)
    pdbqt_mol_path = '%s/%s_protein.pdbqt' % (file, mol_name)
    cmd = '%s %s -r %s -o %s -U \'nphs\' > %s_out2.tmp' % (pythonsh, prep_receptor, pdb_mol_path, pdbqt_mol_path, mol_name)
    os.system(cmd)

def pdbqt_generate(file):
    ligand2pdbqt(file)
    protein2pdbqt(file)

# write config
# in: complex file path
def get_mass_center(ligand):
    conv=openbabel.OBConversion()
    mol=openbabel.OBMol()
    conv.ReadFile(mol,ligand)
    mass=[]
    data2=[]
    for atom in openbabel.OBMolAtomIter(mol):
        x=atom.GetX()
        y=atom.GetY()
        z=atom.GetZ()
        data2.append([x,y,z])
        atm_mass=atom.GetAtomicMass()
        mass.append(atm_mass)
    sum9=mol.GetMolWt()

    coord=np.array(data2)
    co_mass=np.array(mass)
    mass_center=np.dot(np.transpose(coord),np.transpose(co_mass))
    mass_center=mass_center/sum9

    return mass_center

def write_conf(file):
    mol_name = file.split('/')[-1]
    mol2_mol_path = '%s/%s_ligand.mol2' % (file, mol_name)

    mass_center = get_mass_center(mol2_mol_path)
    ligand = f'{file}/{mol_name}_ligand.pdbqt'
    receptor = f'{file}/{mol_name}_protein.pdbqt'
    center_x = mass_center[0]
    center_y = mass_center[1]
    center_z = mass_center[2]

    size = 25
    energy_range = 20
    exhaustiveness = 16
    num_modes = 250
    seed = 666

    conf = f'receptor = {receptor}\nligand = {ligand}\nscoring = vina\ncenter_x = {center_x}\ncenter_y = {center_y}\ncenter_z = {center_z}\nsize_x = {size}\nsize_y = {size}\nsize_z = {size}\nenergy_range = {energy_range}\nexhaustiveness = {exhaustiveness}\nnum_modes = {num_modes}\ncpu = 12\nseed = {seed}'
    with open(f'{file}/{mol_name}_conf.txt', 'w') as f:
        f.write(conf)

# run vina dock
# in: complex file path
def vina_docking(file):
    mol_name = file.split('/')[-1]
    mol2_mol_path = '%s/%s_conf.txt' % (file, mol_name)
    cmd = f'/disk1/vina/vina_1.2.5_linux_x86_64 --config {mol2_mol_path}'
    os.system(cmd)

# after docking transfer pdbqt to mol2 and split multi-molucular mol2 file to single one
# in: complex file path
def pdbqt_split(file, num_dock_path):
    mol_name = file.split('/')[-1]
    pdbqt_file = f'{file}/{mol_name}_ligand_out.pdbqt'
    vina_score = f'{file}/{mol_name}_vina_score.txt'
    with open(pdbqt_file, 'r') as f:
        lines = f.readlines()
    
    conformation_number = 1
    current_conformation_lines = []

    for line in lines:
        output_file = f'{file}/{mol_name}_dock_pdbqt/{mol_name}_mc_{conformation_number}_ligand.pdbqt'
        if line.startswith('REMARK VINA RESULT'):
            score = f'{conformation_number} {line.split()[-3]}\n'
            with open(vina_score, 'a') as f:
                f.write(score)
        if line.startswith('ENDMDL'):
            current_conformation_lines.append(line)
            with open(output_file, 'w') as g:
                g.writelines(current_conformation_lines)
            current_conformation_lines = []
            conformation_number += 1
        else:
            current_conformation_lines.append(line)
    with open(num_dock_path, 'a') as f:
        f.write(f'{mol_name} {conformation_number - 1}\n')

def get_coord_dict(format, inputmol):
   molH = next(pybel.readfile(format, inputmol))
   molH.OBMol.DeleteHydrogens()
   return {atom.idx: atom.coords for atom in molH}

def pdbqt_mol2(file):
    mol_name = file.split('/')[-1]
    output_path = f'{file}/{mol_name}_dock_mol2'
    undocked_pdbqt = get_coord_dict('pdbqt', f'{file}/{mol_name}_ligand.pdbqt')
    original_mol2 = get_coord_dict('mol2', f'{file}/{mol_name}_ligand.mol2')
    docked_pdbqt_list = os.listdir(f'{file}/{mol_name}_dock_pdbqt')
    for docked_pdbqt_file in docked_pdbqt_list:
        docked_pdbqt_path = f'{file}/{mol_name}_dock_pdbqt/{docked_pdbqt_file}'
        docked_pdbqt = get_coord_dict('pdbqt', docked_pdbqt_path)
        conformation_num = int(docked_pdbqt_file.split('_')[-2])
        output_mol2 = f'{output_path}/{mol_name}_mc_{conformation_num}_ligand.mol2'

        if len(undocked_pdbqt) == len(docked_pdbqt) == len(original_mol2):
            original_coord = {}
            for key in original_mol2:
                coord_update = [round(x, 3) for x in original_mol2[key]]
                coord_update = tuple(coord_update)
                original_coord.update({key: coord_update})

            coord_map = {}
            for idx, coord in original_coord.items():
                # potential bottleneck for large molecules
                for ind, coordinates in undocked_pdbqt.items():
                    n = 0
                    if coord[0] == coordinates[0]:
                        n = n + 1
                        if coord[1] == coordinates[1]:
                            n = n + 1
                            if coord[2] == coordinates[2]:
                                n = n + 1
                        else:
                            if coord[2] == coordinates[2]:
                                n = n + 1
                    else:
                        if coord[1] == coordinates[1]:
                            n = n + 1
                            if coord[2] == coordinates[2]:
                                n = n + 1
                        else:
                            if coord[2] == coordinates[2]:
                                n = n + 1
                    if n == 3:
                        coord_map.update({idx: ind})
                    elif n == 2:
                        if idx in coord_map:
                            pass
                        else:
                            coord_map.update({idx: ind})
                    elif n == 1:
                        if idx in coord_map:
                            pass
                        else:
                            coord_map.update({idx: ind})
                    else:
                        pass
                
            if len(coord_map) == len(original_mol2):
                coord_conform = {}
                for index1, index2 in coord_map.items():
                    coord_conform.update({index1:docked_pdbqt.get(index2)})
                mol2 = next(pybel.readfile('mol2', f'{file}/{mol_name}_ligand.mol2'))
                mol2.OBMol.DeleteHydrogens()

                for atom in mol2:
                    atom.OBAtom.SetVector(coord_conform.get(atom.idx)[0], coord_conform.get(atom.idx)[1], coord_conform.get(atom.idx)[2])
                mol2.write('mol2', output_mol2, overwrite=True)
            
        else:
            print('Not equal number of atoms in molecules')
            print(docked_pdbqt_file)

        command0 = 'obabel -imol2 {0} -omol2 -h -p 7.4 -O {1}'.format(output_mol2, output_mol2)
        os.system(command0)

def pdbqt_split_mol2(file, num_dock_path):
    mol_name = file.split('/')[-1]
    mol2_path = f'{file}/{mol_name}_dock_mol2'
    pdbqt_path = f'{file}/{mol_name}_dock_pdbqt'
    os.mkdir(f'{file}/{mol_name}_dock_pdbqt')
    os.mkdir(f'{file}/{mol_name}_dock_mol2')
    pdbqt_split(file, num_dock_path)
    pdbqt_mol2(file)
    mol2_list = os.listdir(mol2_path)
    for mol2 in mol2_list:
        dock_num = mol2.split('_')[-2]
        os.mkdir(f'{file}/{mol_name}_mc_{dock_num}')
        shutil.move(f'{mol2_path}/{mol2}', f'{file}/{mol_name}_mc_{dock_num}')
        shutil.copy(f'{file}/{mol_name}_protein.pdb', f'{file}/{mol_name}_mc_{dock_num}/{mol_name}_protein.pdb')
    shutil.rmtree(mol2_path)
    shutil.rmtree(pdbqt_path)

def vina_dock_workflow(file, num_dock_path, cg_error_path):
    mol_name = file.split('/')[-1]
    try:
        pdbqt_generate(file)
        write_conf(file)
        vina_docking(file)
        pdbqt_split_mol2(file, num_dock_path)
    except:
        with open(cg_error_path, 'a') as f:
            f.write(f'{file}\n')
    os.remove(f'./{mol_name}_out1.tmp')
    os.remove(f'./{mol_name}_out2.tmp')

if __name__=='__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--input_path', type=str)
    argparse.add_argument('--input_csv', type=str)
    argparse.add_argument('--error_path', type=str)
    argparse.add_argument('--num_process', type=int, default=12)
    args = argparse.parse_args()
    set_path, error_path, num_process, csv_path = args.input_path, args.error_path, args.num_process, args.input_csv

    conformation_generate_error = os.path.join(error_path, 'cg_error.txt')
    df = pd.read_csv(csv_path)
    keys = list(df['cid'])
    file_path_list = []
    for key in keys:
        file_path_list.append(os.path.join(set_path, key))
    num = len(file_path_list)
    num_dock_path_list = [f'{set_path}/num_dock.txt'] * num
    conformation_generate_error_lst = [conformation_generate_error] * num

    pool = multiprocessing.Pool(num_process)
    pool.starmap(vina_dock_workflow, zip(file_path_list, num_dock_path_list, conformation_generate_error_lst))
    pool.close()
    pool.join()
    # name_lst = ['1xal_ligand_501_1', '1xal_ligand_501_2', '1xal_ligand_501_3', '2aay_ligand_501_1', '2aay_ligand_501_2', '6c4d_ligand_501_1']
    # name_lst = ['6c4d_ligand_501_1']
    # for n in name_lst:
    #     num_dock_path = '/disk1/final_project/data_test/data_before/num_dock.txt'
    #     file_path = f'/disk1/final_project/data_test/data_before/{n}'
    #     cg_error_path = '/disk1/final_project/data_test/no_error/cg_error.txt'
    #     vina_dock_workflow(file_path, num_dock_path, cg_error_path)
        # pdbqt_split_mol2(file_path, num_dock_path)



    