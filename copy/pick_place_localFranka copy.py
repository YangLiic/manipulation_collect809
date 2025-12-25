# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os

from isaacsim import SimulationApp

_HEADLESS = os.environ.get("ISAACSIM_HEADLESS", os.environ.get("OMNI_ISAAC_HEADLESS", "0")).lower()
_HEADLESS_FLAG = _HEADLESS in {"1", "true", "yes", "on"}

simulation_app = SimulationApp({"headless": _HEADLESS_FLAG})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage, add_reference_to_stage
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid

# 优先使用 omni.isaac.franka 的 Franka 包装类
try:
    from omni.isaac.franka import Franka
except Exception:
    Franka = None

# 控制器导入：优先 omni 路径，失败则回退到示例路径
try:
    from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
    print("成功导入 omni.isaac.franka.controllers.pick_place_controller 中的 PickPlaceController")
except Exception:
    from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
    print("使用示例路径中的 PickPlaceController")

# 加载你的场景 USD
open_stage("/home/yons/data/Collected_World1/World_yang.usd")
# 让场景完成一次更新，确保后续能正确查询 prim
simulation_app.update()

# 创建 World（不要再添加示例任务）
my_world = World(stage_units_in_meters=1.0)

# 包装场景中现有的对象（注意：isaacsim.core.prims 的 XFormPrim 仅接受位置参数）
salt = XFormPrim("/World/Vegetable_7")
bowl = XFormPrim("/World/Bowl_0")

# 固定放置位置
fixed_spawn_pos = np.array([0.0, 0.5, -0.25], dtype=float)

FRANKA_LOCAL_USD = "/home/yons/data/Franka_usd/Franka.usd"
FRANKA_REFERENCE_PATH = "/World/Franka"
FRANKA_NESTED_PATH = "/World/Franka/franka"


def _ensure_local_franka_loaded():
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("无法获取 USD stage，无法加载本地 Franka")

    prim = stage.GetPrimAtPath(FRANKA_REFERENCE_PATH)
    if prim and prim.IsValid() and prim.GetReferences().GetAddedOrExplicitItems():
        return

    if not os.path.isfile(FRANKA_LOCAL_USD):
        raise FileNotFoundError(f"本地 Franka USD 不存在: {FRANKA_LOCAL_USD}")

    print(f"🔧 正在引用本地 Franka USD: {FRANKA_LOCAL_USD}")
    add_reference_to_stage(usd_path=FRANKA_LOCAL_USD, prim_path=FRANKA_REFERENCE_PATH)
    simulation_app.update()


def _resolve_franka_prim_path():
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("无法获取 USD stage 以定位 Franka root")

    for candidate in (FRANKA_NESTED_PATH, FRANKA_REFERENCE_PATH):
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            return candidate

    # 兜底：遍历寻找包含 panda_link0 的 prim 的上一级
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path.lower().endswith("panda_link0"):
            return path.rsplit("/", 1)[0]

    raise RuntimeError("无法在 Stage 中找到 Franka articulation root，请检查 USD 结构。")

# 包装/创建 Franka：引用本地 USD 并使用正确的 prim path
_ensure_local_franka_loaded()
franka_prim_path = _resolve_franka_prim_path()

if Franka is None:
    raise RuntimeError("未找到 Franka 包装类(omni.isaac.franka)。请在扩展中启用 omni.isaac.franka 后重试。")

if is_prim_path_valid(franka_prim_path):
    my_franka = Franka(prim_path=franka_prim_path, name="Franka")
    try:
        my_franka.set_world_pose(position=fixed_spawn_pos)
    except Exception:
        XFormPrim(franka_prim_path).set_world_pose(position=fixed_spawn_pos)
    simulation_app.update()
else:
    my_franka = Franka(prim_path=franka_prim_path, name="Franka", position=fixed_spawn_pos)

# 注册到 scene（仅注册机器人即可，XFormPrim 直接用于读姿态，无需加入 scene）
# my_world.scene.add(salt)
# my_world.scene.add(bowl)
my_world.scene.add(my_franka)

my_world.reset()

# 创建抓放控制器
my_controller = PickPlaceController(
    name="pick_place_controller",
    gripper=my_franka.gripper,
    robot_articulation=my_franka
)
articulation_controller = my_franka.get_articulation_controller()

def _force_open_gripper():
    try:
        open_action = my_franka.gripper.forward(action="open")
        articulation_controller.apply_action(open_action)
    except Exception:
        # 兼容少数实现：直接用 opened_positions
        if hasattr(my_franka.gripper, "joint_opened_positions"):
            my_franka.gripper.set_joint_positions(my_franka.gripper.joint_opened_positions)


# 初始化：强制打开夹爪一次
_force_open_gripper()

reset_needed = False

# 可调参数
# 可调参数
placing_height_offset = 0.05
eef_lateral_offset = np.array([0.0, 0.00, 0.0])

def step_once(render: bool = True) -> bool:
    """执行一次仿真和控制循环，返回 False 表示无需继续。"""
    global reset_needed

    if not simulation_app.is_running():
        return False

    my_world.step(render=render)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            _force_open_gripper()
            reset_needed = False

        # 获取盐瓶与碗的世界位姿（批量API，取第一个元素）
        salt_positions, _ = salt.get_world_poses()
        bowl_positions, _ = bowl.get_world_poses()
        salt_pos = salt_positions[0]
        bowl_pos = bowl_positions[0]

        picking_position = salt_pos + np.array([0.0, -0.06, -0.1])
        placing_position = bowl_pos + np.array([0.0, 0.0, placing_height_offset])

        current_joint_positions = my_franka.get_joint_positions()

        actions = my_controller.forward(
            picking_position=picking_position,
            placing_position=placing_position,
            current_joint_positions=current_joint_positions,
            end_effector_offset=eef_lateral_offset
        )

        articulation_controller.apply_action(actions)

        # 在状态机早期阶段(0/1/2)持续强制打开夹爪，避免靠近时碰撞
        try:
            if hasattr(my_controller, "get_current_event") and my_controller.get_current_event() < 3:
                open_action = my_franka.gripper.forward(action="open")
                articulation_controller.apply_action(open_action)
        except Exception:
            pass

    return True


# 只有直接运行时才执行主循环，被导入时跳过
if __name__ == "__main__":
    try:
        while step_once(render=True):
            pass
    finally:
        simulation_app.close()

#./python.sh /home/yons/data/isaacsim/pick_place_localFranka.py
