#!/usr/bin/env bash
# 用某一折保存的权重，对 test 集合做预测（第 2 阶段：整体泛化评估）
# 用法: bash scripts/03_predict_test.sh [权重文件名]
#       默认用随包发布的 rp0_2024-03-20_06_45_37.pth
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE/example/test"

export PYTHONPATH="$HERE/software_code:${PYTHONPATH:-}"
ln -sfn "$HERE/software_code/model" model_save_cross
mkdir -p output

python "$HERE/software_code/predict_koff.py" \
    --input_path  . \
    --input_csv   test.csv \
    --output_path output "$@"

echo
echo "输出: $HERE/example/test/output/prediction.csv"
