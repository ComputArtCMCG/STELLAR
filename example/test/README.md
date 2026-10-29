# example/test — held-out test set (130 complexes)

Prediction input for the second stage of the protocol: one fold's saved weights are used to
predict this set, which was never seen during training.

Identical layout to `example/cdk2/`, except that the raw structures are **not** redistributable,
so only the pre-computed features are shipped.

```
example/test/
├── test.csv                 cid, pkoff
├── num_dock.txt             cid, number of docked conformers
├── data_in/graph_collect/   8 pre-computed feature pickles
└── output/                  predictions are written here
```

```bash
bash ../../scripts/03_predict_test.sh
# -> output/prediction.csv   (columns: complex_name, true, pred)
```

`predict_koff.py` loads `./model_save_cross/rp0_2024-03-20_06_45_37.pth` **relative to the current
working directory**, which is why the script creates a `model_save_cross` symlink pointing at
`software_code/model/`. Pass a different checkpoint name as the first argument to use your own
fold, e.g. `bash ../../scripts/03_predict_test.sh rp0_2024-03-21_07_43_31`.
A CUDA GPU is required.
