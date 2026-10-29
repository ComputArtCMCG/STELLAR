#!/usr/bin/env bash
# 用随包发布的特征文件复现 5 折训练
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
# training.py 里的特征/日志路径都是相对 CWD 的硬编码（./new_train_data_5、./log_cross …）
# 因此必须在 data/ 目录下运行
cd "$HERE/data"

export PYTHONPATH="$HERE/software_code:${PYTHONPATH:-}"
mkdir -p log_cross conclusion_cross_n stats_n model_save_cross

python "$HERE/software_code/training.py" \
    --home_path       ./splits \
    --model_save_path ./model_save_cross \
    --num_process 8 "$@"

echo
echo "模型:     $HERE/data/model_save_cross/"
echo "日志:     $HERE/data/log_cross/"
echo "预测csv:  $HERE/data/conclusion_cross_n/"
echo "指标汇总: $HERE/data/stats_n/"
