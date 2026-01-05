#!/bin/bash
# 测试多 Episode 采集功能

echo "🧪 测试多 Episode 采集（10 个 episodes）"
echo "每个 episode 将使用不同的随机抓取姿态（±10度随机偏移）"
echo ""

/home/di-gua/isaac-sim/python.sh scipy/collect_curobo.py \
    --script scipy/collect_pick_place_seed \
    --out ./collect_output \
    --fps 30 \
    --width 1280 \
    --height 960 \
    --rgb-format jpg \
    --depth-format npy \
    --timestamp-log both \
    --camera-dir-mode short \
    --num-episodes 10 \
    --headless

echo ""
echo "✅ 测试完成！检查 collect_output 目录查看生成的 episodes"
