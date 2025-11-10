#!/bin/bash
echo "=========================================="
echo "TRAINING STATUS CHECK"
echo "Time: $(date)"
echo "=========================================="
echo ""

# Check if training is running
if ps -p 65893 > /dev/null 2>&1; then
    echo "Status: ⏳ TRAINING STILL RUNNING"
    echo ""
    echo "Latest progress:"
    grep "Epoch.*batch.*00," finetune_ja_en_200h_FIXED.log | tail -3
    echo ""
    echo "GPU Usage:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
        awk '{printf "  GPU %s: %.1fGB, %d%% util\n", $1, $2/1024, $3}'
else
    echo "Status: ✓ TRAINING COMPLETED"
    echo ""
    
    # Check for errors
    if tail -100 finetune_ja_en_200h_FIXED.log | grep -qi "error\|oom\|traceback"; then
        echo "⚠️  ERRORS DETECTED!"
        echo ""
        tail -50 finetune_ja_en_200h_FIXED.log | grep -i "error\|oom\|traceback" | tail -10
    else
        echo "✓ No errors detected"
    fi
    
    echo ""
    echo "Final training progress:"
    grep "Epoch.*batch.*00," finetune_ja_en_200h_FIXED.log | tail -5
    
    echo ""
    echo "Saved checkpoints:"
    ls -lh zipformer/exp_finetune_ja_en_200h_v2/*.pt 2>/dev/null | tail -5
fi

echo ""
echo "=========================================="
echo "Monitor log (last 20 lines):"
echo "=========================================="
tail -20 training_monitor.log 2>/dev/null || echo "No monitor log found"
echo ""
