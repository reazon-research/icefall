#!/bin/bash
# Training monitor script for bilingual fine-tuning
# Usage: ./monitor_training.sh

LOG_FILE="finetune_bilingual_ja_en_330h.log"

echo "====== Bilingual Training Monitor ======"
echo "Data: 134.6h Japanese + 195.9h English = 330.5h total"
echo "========================================="
echo ""

# Check if training is running
if pgrep -f "finetune.py.*bilingual" > /dev/null; then
    echo "✓ Training process is RUNNING"
else
    echo "✗ Training process NOT FOUND"
    exit 1
fi

echo ""
echo "--- Latest Training Progress ---"
tail -50 "$LOG_FILE" | grep "Epoch.*batch.*loss\[" | tail -5

echo ""
echo "--- Loss Trend (last 10 log points) ---"
tail -500 "$LOG_FILE" | grep "Epoch.*batch.*loss\[" | grep -oP "tot_loss\[loss=\K[0-9.]+" | tail -10

echo ""
echo "--- GPU Memory Usage ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk '{printf "GPU %s: %s/%s MB (%.1f%% util)\n", $1, $2, $3, $4}'

echo ""
echo "--- Training Speed ---"
tail -100 "$LOG_FILE" | grep "Epoch 1, batch" | tail -2

echo ""
echo "========================================="
echo "Monitor updates every 30 seconds. Press Ctrl+C to stop."
