# STELLAR-koff: a deep learning model for predicting ligand dissociation rate constants based on a protein-ligand interaction landscape

Reproduction kit for the STELLAR koff-prediction model.
It contains everything needed to (i) reproduce the 5-fold training results and
(ii) run a prediction on the CDK2 test set.

```
STELLAR_git/
├── README.md
├── requirements.txt
├── software_code/
├── data/           data for training
├── example/          data for test
│   ├── cdk2/
│   └── test/
└── scripts/
```

---


## Quick start

The evaluation protocol has two stages:

**Stage 1 — 5-fold cross-validation (overall performance).**
`training.py` trains 5 models (one per fold). During training only `train` and `valid` are
monitored (RMSE / Pearson printed every epoch, early stopping on `valid_rmse`); after training
each fold is evaluated on `train / valid / focal / p38` and a mean ± std table over the 5 folds
is printed at the end.

**Stage 2 — held-out proteins (`test`, `cdk2`).**
One fold's saved weights are used to predict `test` and `cdk2`. The released
`software_code/model/rp0_2024-03-20_06_45_37.pth` is that fold's checkpoint.

```bash
# 0) environment + data sanity check
bash scripts/00_check_env.sh

# 1) stage 1: 5-fold cross-validation
bash scripts/02_train_5fold.sh

# 2) stage 2: predict the held-out sets with one fold's weights
bash scripts/03_predict_test.sh
bash scripts/01_predict_cdk2.sh
```

---

## System requirements

* Python 3.7, PyTorch 1.13 (CUDA), numpy / pandas / scikit-learn / scipy / tqdm / networkx
* `egnn_pytorch`, `prefetch_generator` (`pip install -r requirements.txt`)
* A CUDA GPU is required: `training.py` and `predict_koff.py` both hard-code `cuda:0`.
* `rdkit / openbabel / pymol` are only needed if you want to regenerate features from raw
  structures (`generate_conformation.py` → `cal_space_relationship.py` → `input_feat_generate.py`);
  they are **not** needed for steps 1 and 2 above.
* Autodock Vina is needed for STELLAR-koff to generate conformations, so please install it before using STELLAR-koff.
  You have to install Autodock Vina and change the hard path in software code to help STELLAR-koff use Vina successfully.

---
## How to predict
See `../example/cdk2/README.md` for a worked example (CDK2, 20 complexes) that starts from
`ligand.mol2` / `protein.pdb` and ends with `prediction.csv`.




