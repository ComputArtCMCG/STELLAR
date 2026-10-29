# example/cdk2 — a worked example: from raw structures to prediction

20 CDK2 complexes with measured koff. The folder gives you **two ways** to use it:

| way | starting point | time |
|---|---|---|
| **Quick** | pre-computed features (`data_in/graph_collect/`) 
| **Tutorial** | raw structures (`structures/`) → docking → similarity → features → bins → prediction 

```
example/cdk2/
├── structures/<cid>/
│   ├── <cid>_ligand.mol2        the ligand to be docked
│   ├── <cid>_protein.pdb        the receptor
│   └── <cid>_temp_ligand.mol2   co-crystal template, used only to place the docking box
├── cdk2.csv                     cid, pkoff
├── num_dock.txt                 cid, number of docked conformers
├── data_in/graph_collect/       8 pre-computed feature pickles (what the model actually reads)
└── output/                      predictions are written here
```

## Quick path

```bash
cd <repo>/example/cdk2
python ../../software_code/predict_koff.py \
    --input_path . --input_csv cdk2.csv --output_path output   # see 01_predict_cdk2.sh
# -> output/prediction.csv   (complex_name, true, pred)
```

---

# Tutorial: rebuild the features from the raw structures

All commands below are run from the **repository root** (`STELLAR_git/`).

## Step 0 — prerequisites and paths to fix

External programs:

| program | used by |
|---|---|
| MGLTools (`pythonsh`, `prepare_ligand4.py`, `prepare_receptor4.py`) | `generate_conformation.py` |
| Vina 1.2.5 | `generate_conformation.py` |
| `AutoRMSD-clear` (shipped in `software_code/AutoRMSD/`) | `cal_space_relationship.py` |
| PyMOL, OpenBabel, RDKit | `input_feat_generate.py` |
| a CUDA GPU | `input_feat_generate.py`, `predict_koff.py` |

Before running anything, put your own paths into these **6 lines**:

| file | line | what |
|---|---|---|
| `software_code/generate_conformation.py` | 10 | `MGL_BIN_DIR` |
| `software_code/generate_conformation.py` | 12 | `UTILS_PATH` |
| `software_code/generate_conformation.py` | 86 | the `vina_1.2.5_linux_x86_64` binary |
| `software_code/cal_space_relationship.py` | 47 | the `AutoRMSD-clear` binary |
| `software_code/input_feat_generate.py` | 54, 60 | an error-log path |
| `software_code/input_feat_generate.py` | 96 | the GIGN feature-extractor checkpoint — point it at `software_code/model/20230721_165213_GIGN_repeat2/epoch-322, … .pt` (shipped in this repo) |

## Step 1 — prepare a working copy

```bash
mkdir -p work/cdk2/err
cp example/cdk2/structures/* -r work/cdk2/
# the csv needs a `keys` column for input_feat_generate.py
python - <<'PY'
import pandas as pd
d = pd.read_csv('example/cdk2/cdk2.csv')
d['keys'] = d['cid']
d[['keys', 'cid', 'pkoff']].to_csv('work/cdk2/cdk2.csv', index=False)
PY
```

Working layout afterwards (this is also the layout every script expects):

```
work/cdk2/
├── cdk2.csv          keys, cid, pkoff          ← note: the file name must equal the folder name
├── <cid>/            <cid>_ligand.mol2 + <cid>_protein.pdb + <cid>_temp_ligand.mol2
└── num_dock.txt      written by Step 2
```

## Step 2 — docking (Vina)

```bash
python software_code/generate_conformation.py \
    --input_path work/cdk2 --input_csv work/cdk2/cdk2.csv \
    --error_path work/cdk2/err --num_process 8
```

Produces, per complex: `<cid>_ligand.pdbqt`, `<cid>_protein.pdbqt`, `<cid>_conf.txt`,
`<cid>_ligand_out.pdbqt`, `<cid>_decoys.mol2`, `<cid>_vina_score.txt`, the conformer folders
`<cid>_mc_1 … <cid>_mc_K`, and the global `num_dock.txt`.
Parameters: `num_modes=250`, `exhaustiveness=16`, `energy_range=20`, `size=25 Å`, `seed=666`.
About **25 min** for these 20 complexes on 16 cores.

> ⚠️ **Box centre.** `write_conf()` (line 62) centres the box on the mass centre of
> `<cid>_ligand.mol2`. For CDK2 the features shipped in `data_in/` were produced with the box
> centred on the **co-crystal template** `<cid>_temp_ligand.mol2` (the `dock_model=m` behaviour of
> the original `predict_dock_generate.py`). To reproduce them, edit that line to
> `'%s/%s_temp_ligand.mol2'`. Without this edit the docking poses — and therefore all downstream
> features — will differ from the shipped ones.

## Step 3 — mass centres + AutoRMSD similarity

```bash
python software_code/cal_space_relationship.py \
    --input_path work/cdk2 --input_csv work/cdk2/cdk2.csv \
    --error_path work/cdk2/err --num_process 8
```

Adds to each complex folder: `<cid>_mc.txt` (K × 3 mass centres) and `myrmsd_similarity.txt`
(K × K AutoRMSD similarity, 1.0 on the diagonal). About **10 min**.

## Step 4 — pocket + GIGN conformer features + protein residue features

```bash
python software_code/input_feat_generate.py --set_path work/cdk2 --num_process 8
```

Note the different interface: it takes only `--set_path` and reads `<set_path>/num_dock.txt` plus
`<set_path>/<folder name>.csv` — which is why the working folder and the csv must share the name
`cdk2`. Adds per complex: `<cid>_feat.pkl` (K × 256), `<cid>_protfeat.pkl`, and inside every
`<cid>_mc_i/` folder `Pocket_5A.pdb`, `<cid>_mc_i_5A.rdkit`, `Graph_GIGN-<cid>_mc_i_5A.pyg`.
Needs a GPU; a few minutes to ~30 min depending on the card.

## Step 5 — bundle everything into the 8 `.bin` files

```bash
python scripts/04_build_graph_collect.py \
    --input_path work/cdk2 --input_csv work/cdk2/cdk2.csv \
    --out_dir    work/cdk2 --num_process 8
```

Produces `work/cdk2/data_in/graph_collect/` with `feat_matrix_list.bin`, `simi_matrix_list.bin`,
`mc_matrix_list.bin`, `mcd_matrix_list.bin`, `residue_feat_matrix_list.bin`,
`residue_mc_matrix_list.bin`, `keys.bin`, `labels_list.bin` — the exact format `training.py` and
`predict_koff.py` consume.

> ✅ Verified: running this step from the per-complex files of this CDK2 set reproduces the
> shipped `data_in/graph_collect/` **bit for bit** (max element-wise difference 0.0 for all 8 bins).

## Step 6 — predict

```bash
cd work/cdk2
ln -sfn ../../software_code/model model_save_cross   # predict_koff.py reads ./model_save_cross/
mkdir -p output
python ../../software_code/predict_koff.py \
    --input_path . --input_csv cdk2.csv --output_path output
# -> output/prediction.csv   (complex_name, true, pred)
```

RMSE / Pearson computed on that file are the reported CDK2 numbers. The default checkpoint is
`rp0_2024-03-20_06_45_37.pth`; edit the `model_name` list in `predict_koff.py` (line 82) or the
symlink target to use another fold.

## One-line summary

```
structures/  →(2) docking→  <cid>_mc_i, num_dock.txt
             →(3) AutoRMSD→  <cid>_mc.txt, myrmsd_similarity.txt
             →(4) GIGN→      <cid>_feat.pkl, <cid>_protfeat.pkl, Pocket_5A.pdb
             →(5) bundle→    data_in/graph_collect/*.bin
             →(6) predict→   prediction.csv
```
