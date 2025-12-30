#!/bin/bash
# 数据采集测试脚本
# 用于快速测试 collect_curobo.py 与 pick_place_localFranka_curobo_scipy_seed.py 的集成

# 设置输出目录
OUTPUT_DIR="/home/di-gua/licheng/manipulation/collect_output"

# 测试参数
FPS=5                    # 降低 FPS 以便观察
WIDTH=1280
HEIGHT=960
RGB_FORMAT="jpg"
DEPTH_FORMAT="npy"

echo "=========================================="
echo "数据采集集成测试"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "FPS: $FPS"
echo "分辨率: ${WIDTH}x${HEIGHT}"
echo "=========================================="
echo ""

# 运行数据采集
/home/di-gua/isaac-sim/python.sh collect_curobo.py \
  --script collect_pick_place_seed \
  --out "$OUTPUT_DIR" \
  --fps $FPS \
  --width $WIDTH \
  --height $HEIGHT \
  --rgb-format $RGB_FORMAT \
  --depth-format $DEPTH_FORMAT \
  --timestamp-log both \
  --exit-on-complete \
  --gui

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "检查输出:"
echo "  ls -R $OUTPUT_DIR/session_*/episode_0001/"
echo ""
echo "预期文件:"
echo "  - colors/Camera/*.jpg"
echo "  - depths/Camera/*.npy"
echo "  - data.json"
echo "  - franka_states.csv"
echo "  - timeline.csv"
echo "=========================================="
