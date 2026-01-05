#!/usr/bin/env python3
"""
数据采集包装脚本 - 用于 collect_curobo.py

此脚本导入 pick_place_localFranka_curobo_scipy_seed 模块并配置数据采集参数。

使用方法:
    /home/di-gua/isaac-sim/python.sh collect_curobo.py \\
        --script collect_pick_place_seed \\
        --out /home/di-gua/licheng/manipulation/collect_output \\
        --fps 30 \\
        --width 1280 \\
        --height 960 \\
        --rgb-format jpg \\
        --depth-format npy \\
        --timestamp-log both

配置说明:
    可以通过修改下面的 configure_collection 调用来改变抓取/放置的物体。
    所有参数都是可选的，未指定的参数将使用模块默认值。
"""

# 导入主模块（这会初始化 IsaacSim 环境）
#from pick_place_localFranka_curobo_scipy_seed import *
from pick_place_cu_ramsci import *

# ============================================================
# 配置数据采集参数
# ============================================================

# 示例 1: 抓取盐罐放到砧板上
configure_collection(
    pick_obj="/World/SaltShaker_3",
    place_obj="/World/Bowl_0",
    auto_height_offset=True,
    use_seed_model=False,  # 不使用 Seed 模型，使用手动姿态
    render=True
)

# 示例 2: 抓取瓶子放到砧板上（注释掉）
# configure_collection(
#     pick_obj="/World/Bottle_2",
#     place_obj="/World/CuttingBoard_4",
#     auto_height_offset=True,
#     use_seed_model=False,
#     render=True
# )

# 示例 3: 抓取蔬菜放到碗里（注释掉）
# configure_collection(
#     pick_obj="/World/Vegetable_7",
#     place_obj="/World/Bowl_0",
#     auto_height_offset=True,
#     use_seed_model=False,
#     render=True
# )

# ============================================================
# 以下变量会被 collect_curobo.py 自动使用
# ============================================================
# - simulation_app: IsaacSim 应用实例
# - my_world: World 实例
# - my_franka: Franka 机器人实例
# - my_controller: CuroboPickPlaceController 实例
# - step_once: 主循环函数（会被 collector 调用）
#
# collector 会自动检测 my_controller.is_planning 标志，
# 在规划期间暂停数据采集，只在执行动作时采集图像和关节状态。
