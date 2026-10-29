#!/usr/bin/env bash
# 环境与数据自检（不训练、不预测）
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
FAIL=0
say(){ printf '%-46s %s\n' "$1" "$2"; }

say "python" "$(python -V 2>&1)"
python - <<'PY' || FAIL=1
import sys
mods = ['torch','numpy','pandas','sklearn','scipy','tqdm','networkx','egnn_pytorch','prefetch_generator']
for m in mods:
    try:
        __import__(m); print('  %-20s OK' % m)
    except Exception as e:
        print('  %-20s MISSING (%s)' % (m, e))
import torch
print('  cuda available   ', torch.cuda.is_available())
PY

echo; echo "数据完整性:"
python - "$HERE" <<'PY'
import os, sys, pickle
root = sys.argv[1]
splits = ['train_1','valid_1','train_2','valid_2','train_3','valid_3','train_4','valid_4','train_5','valid_5','focal','p38','test']
bad = 0
for s in splits:
    sp = os.path.join(root,'data','splits',s)
    g  = os.path.join(root,'data','new_train_data_5',s+'_in','graph_collect')
    n_csv = sum(1 for x in os.listdir(sp) if x.endswith('.csv'))
    n_bin = len([x for x in os.listdir(g) if x.endswith('.bin')]) if os.path.isdir(g) else 0
    ok = (n_csv==1 and n_bin==8)
    bad += (0 if ok else 1)
    print('  %-10s csv %d | bin %2d | %s' % (s, n_csv, n_bin, 'OK' if ok else 'PROBLEM'))
e = os.path.join(root,'example','cdk2')
print('  cdk2 example: structures %d 个, bin %d | %s' % (
    len(os.listdir(os.path.join(e,'structures'))) if os.path.isdir(os.path.join(e,'structures')) else 0,
    len([x for x in os.listdir(os.path.join(e,'data_in','graph_collect')) if x.endswith('.bin')]) if os.path.isdir(os.path.join(e,'data_in','graph_collect')) else 0,
    'OK'))
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && FAIL=1
echo
[ $FAIL -eq 0 ] && echo "自检通过" || echo "自检未通过，请检查上面的 MISSING/PROBLEM"
exit $FAIL
