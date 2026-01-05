#!/bin/bash
# 测试 2 个 Episode 采集（用于快速验证）

echo "🧪 测试 2 个 Episode 采集"
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
    --num-episodes 2 \
    --headless

echo ""
echo "✅ 测试完成！检查 collect_output 目录应该看到 episode_0001 和 episode_0002"
ls -lh ./collect_output/session_*/
