#!/usr/bin/env bash
# 用训练好的模型对 cdk2 测试集做预测（使用已预计算的特征，秒级完成）
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE/example/cdk2"

export PYTHONPATH="$HERE/software_code:${PYTHONPATH:-}"
# predict_koff.py 从**当前工作目录**读 ./model_save_cross/<name>.pth
ln -sfn "$HERE/software_code/model" model_save_cross
mkdir -p output

python "$HERE/software_code/predict_koff.py" \
    --input_path  . \
    --input_csv   cdk2.csv \
    --output_path output "$@"

echo
echo "输出: $HERE/example/cdk2/output/prediction.csv"
