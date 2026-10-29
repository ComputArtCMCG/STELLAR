import os
import pandas as pd
import openbabel
import numpy as np
import multiprocessing
import argparse

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

def get_graph_similarity(complex_path, complex_id, num_dock):
    multi_docked_path = os.path.join(complex_path, f'{complex_id}_decoys.mol2')
    ref_path = os.path.join(complex_path, f'{complex_id}_ligand.mol2')
    current_path = os.getcwd()
    similarity_path = os.path.join(current_path, 'myrmsd_similarity.txt')

    if not os.path.exists(multi_docked_path):
        g = open(multi_docked_path, 'w', encoding='utf8')
        for i in range(1, num_dock + 1):
            complex_docked = complex_id + '_mc_' + f'{i}'
            ligand_docked = complex_id + '_mc_' + f'{i}_ligand.mol2'
            ligand_docked_path = os.path.join(complex_path, complex_docked, ligand_docked)
            for line in open(ligand_docked_path, encoding='utf8'):
                g.writelines(line)
            g.write('\n')
        g.close()

    generate_similarity_cmd = f'/disk1/STELLAR/final_project/software_code/AutoRMSD/AutoRMSD-clear.AutoRMSD-clear {ref_path} {multi_docked_path}'
    move_similarity_cmd = f'mv {similarity_path} {complex_path}'
    os.system(generate_similarity_cmd)
    os.system(move_similarity_cmd)

def cal_mass_center_workflow(data_dir, complex_name, num_dock, error_path):
    mc_output_path = os.path.join(data_dir, f'{complex_name}_mc.txt')
    try:
        mc_list = []
        for i in range(1, num_dock + 1):
            dock_path = os.path.join(data_dir, f'{complex_name}_mc_{i}', f'{complex_name}_mc_{i}_ligand.mol2')
            mc_list.append(get_mass_center(dock_path))
        mc_array = np.array(mc_list)
        np.savetxt(mc_output_path, mc_array, fmt='%f', delimiter='\t')
    except:
        with open(error_path, 'a') as f:
            f.write(f'{complex_name} can not cal mass center\n')

if __name__=='__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--input_path', type=str)
    argparse.add_argument('--input_csv', type=str)
    argparse.add_argument('--error_path', type=str)
    argparse.add_argument('--num_process', type=int, default=12)
    args = argparse.parse_args()
    input_path, input_csv, error_path, num_process = args.input_path, args.input_csv, args.error_path, args.num_process
    cal_mc_error = os.path.join(error_path, 'mc_error.txt')
    cal_simi_error = os.path.join(error_path, 'simi_error.txt')

    csv_cid_lst = list(pd.read_csv(input_csv)['cid'])
    num_dock_path = os.path.join(input_path, 'num_dock.txt')
    num_cid_lst = []
    num_dock_lst = []
    with open(num_dock_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        num_cid_lst.append(line.split()[0])
        num_dock_lst.append(int(line.split()[-1]))
    new_cid_lst = []
    new_num_dock_list = []
    for i, cid in enumerate(num_cid_lst):
        if cid in csv_cid_lst:
            new_cid_lst.append(cid)
            new_num_dock_list.append(num_dock_lst[i])

    file_list = []
    cal_mc_error_path_list = []
    for cid in new_cid_lst:
        file_list.append(os.path.join(input_path, cid))
        cal_mc_error_path_list.append(cal_mc_error)

    pool = multiprocessing.Pool(num_process)
    pool.starmap(cal_mass_center_workflow, zip(file_list, new_cid_lst, new_num_dock_list, cal_mc_error_path_list))
    pool.close()
    pool.join()

    for path, cid, num in zip(file_list, new_cid_lst, new_num_dock_list):
        try:
            get_graph_similarity(path, cid, num)
        except:
            with open(cal_simi_error, 'a') as f:
                f.write(f'{path} {num}\n')
    # num_dock_path = '/disk1/final_project/data_test/data_before/num_dock.txt'
    # num_cid_lst = []
    # num_dock_lst = []
    # with open(num_dock_path, 'r') as f:
    #     lines = f.readlines()
    # for line in lines:
    #     num_cid_lst.append(line.split()[0])
    #     num_dock_lst.append(int(line.split()[-1]))

    # name_lst = ['1xal_ligand_501_1', '1xal_ligand_501_2', '1xal_ligand_501_3', '2aay_ligand_501_1', '2aay_ligand_501_2', '6c4d_ligand_501_1']
    # for n in name_lst:
    #     i = num_cid_lst.index(n)
    #     num = num_dock_lst[i]
    #     file_path = f'/disk1/final_project/data_test/data_before/{n}'
    #     cal_mc_error_path = '/disk1/final_project/data_test/no_error/cg_error.txt'
    #     get_graph_similarity(file_path, n, num)
    #     cal_mass_center_workflow(file_path, n, num, cal_mc_error_path)







    