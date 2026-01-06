# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""
基于 CuRobo MotionGen 的抓取-放置脚本（具备避障能力）
严格遵循 pick_place_localFranka.py 的框架，替换 PickPlaceController 为 CuRobo
参考 simple_stacking.py 的 CuRobo 实现
"""

import os
import sys

from isaacsim import SimulationApp

_HEADLESS = os.environ.get("ISAACSIM_HEADLESS", os.environ.get("OMNI_ISAAC_HEADLESS", "0")).lower()
_HEADLESS_FLAG = _HEADLESS in {"1", "true", "yes", "on"}

simulation_app = SimulationApp({"headless": _HEADLESS_FLAG})

# Third Party - Import torch AFTER SimulationApp initialization
import torch
a = torch.zeros(4, device="cuda:0")  # 必须在导入 curobo 前初始化 torch

import numpy as np
np.set_printoptions(suppress=True)

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage, add_reference_to_stage
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.types import ArticulationAction

# ---------------- CuRobo 导入 ----------------
CUROBO_PATH = os.path.join(os.path.dirname(__file__), "curobo", "src")
if CUROBO_PATH not in sys.path:
    sys.path.insert(0, CUROBO_PATH)

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# Import grasp pose generator (指定姿态版本)
from grasp_pose_generator_specified import generate_grasp_pose, quaternion_to_euler

# Import Seed 模型抓取姿态估计
from estimate_grasp_pose_seed import estimate_grasp_pose

# 优先使用 omni.isaac.franka 的 Franka 包装类
try:
    from omni.isaac.franka import Franka
except Exception:
    Franka = None

# 添加 USD 相关导入用于 bounding box 计算
from pxr import UsdGeom, Gf, Usd
import omni.usd

def get_object_bounding_box(prim_path: str):
    """
    获取物体的世界坐标系 bounding box
    
    参数:
        prim_path: 物体的 USD 路径
        
    返回:
        (min_point, max_point): bounding box 的最小和最大点 (世界坐标)
        如果失败返回 None
    """
    try:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        
        if prim.IsValid():
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
            bound = bbox_cache.ComputeWorldBound(prim)
            bbox = bound.ComputeAlignedBox()
            
            min_point = bbox.GetMin()
            max_point = bbox.GetMax()
            
            # 转换为 numpy 数组
            min_array = np.array([min_point[0], min_point[1], min_point[2]])
            max_array = np.array([max_point[0], max_point[1], max_point[2]])
            
            return (min_array, max_array)
    except Exception as e:
        print(f"⚠️ 获取 bounding box 失败: {e}")
        return None
    
    return None

def check_object_overlap(obj1_path: str, obj2_path: str, safety_margin: float = 0.05) -> bool:
    """
    检查两个物体的 bounding box 是否重叠（带安全边距）
    
    参数:
        obj1_path: 第一个物体的 USD 路径
        obj2_path: 第二个物体的 USD 路径
        safety_margin: 安全边距（米），默认 5cm
        
    返回:
        True 表示重叠（碰撞），False 表示安全
    """
    bbox1 = get_object_bounding_box(obj1_path)
    bbox2 = get_object_bounding_box(obj2_path)
    
    if bbox1 is None or bbox2 is None:
        # 无法获取 bounding box，保守地认为不碰撞
        return False
    
    min1, max1 = bbox1
    min2, max2 = bbox2
    
    # 扩展 bounding box（添加安全边距）
    min1_expanded = min1 - safety_margin
    max1_expanded = max1 + safety_margin
    
    # 检查是否重叠（AABB 碰撞检测）
    overlap_x = max1_expanded[0] >= min2[0] and min1_expanded[0] <= max2[0]
    overlap_y = max1_expanded[1] >= min2[1] and min1_expanded[1] <= max2[1]
    overlap_z = max1_expanded[2] >= min2[2] and min1_expanded[2] <= max2[2]
    
    return overlap_x and overlap_y and overlap_z


def randomize_object_position(
    obj_path: str,
    original_pos: np.ndarray,
    offset_range: float = 0.1,
    max_attempts: int = 100,
    scene_objects: list = None,
    safety_margin: float = 0.05,
    table_path: str = "/World/Table_1",
    table_margin: float = 0.05
) -> np.ndarray:
    """
    随机偏移物体位置（仅 X-Y 平面），并检查碰撞
    
    参数:
        obj_path: 要移动的物体的 USD 路径
        original_pos: 物体的原始位置（世界坐标）
        offset_range: 偏移范围（米），默认 ±0.1m
        max_attempts: 最大尝试次数，默认 10 次
        scene_objects: 场景中其他物体的路径列表（用于碰撞检测）
        safety_margin: 碰撞检测安全边距（米）
        table_path: 桌子的 USD 路径
        table_margin: 距离桌子边缘的安全距离（米），默认 0.1m
        
    返回:
        新位置（如果找到安全位置），否则返回原始位置
    """
    try:
        # 获取桌子的 bounding box
        table_bbox = get_object_bounding_box(table_path)
        if table_bbox is None:
            print(f"⚠️ 无法获取桌子边界，跳过桌面检查")
            table_min, table_max = None, None
        else:
            table_min, table_max = table_bbox
            print(f"📐 桌面边界: X [{table_min[0]:.3f}, {table_max[0]:.3f}], Y [{table_min[1]:.3f}, {table_max[1]:.3f}]")
        
        # 使用 USD API 直接设置位置
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(obj_path)
        if not prim.IsValid():
            print(f"⚠️ 无效的 prim 路径: {obj_path}")
            return original_pos
        
        xform = UsdGeom.Xformable(prim)
        
        # 获取或创建 translate 操作（不重复添加）
        xform_ops = xform.GetOrderedXformOps()
        translate_op = None
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        
        if translate_op is None:
            translate_op = xform.AddTranslateOp()
        
        for attempt in range(max_attempts):
            # 生成随机偏移（仅 X-Y）
            offset_x = np.random.uniform(-offset_range, offset_range)
            offset_y = np.random.uniform(-offset_range, offset_range)
            new_pos = original_pos + np.array([offset_x, offset_y, 0.0])
            
            # 检查是否在桌面范围内（带边距）
            if table_min is not None and table_max is not None:
                if (new_pos[0] < table_min[0] + table_margin or 
                    new_pos[0] > table_max[0] - table_margin or
                    new_pos[1] < table_min[1] + table_margin or 
                    new_pos[1] > table_max[1] - table_margin):
                    print(f"   ⚠️ 尝试 {attempt + 1}/{max_attempts} 超出桌面范围，重新生成...")
                    continue
            
            # 临时设置新位置（用于碰撞检测）
            translate_op.Set(Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))
            
            # 检查是否与其他物体碰撞
            has_collision = False
            if scene_objects:
                for other_obj_path in scene_objects:
                    if other_obj_path == obj_path:
                        continue  # 跳过自己
                    
                    if check_object_overlap(obj_path, other_obj_path, safety_margin):
                        has_collision = True
                        break
            
            if not has_collision:
                print(f"✅ 随机偏移成功 (尝试 {attempt + 1}/{max_attempts}):")
                print(f"   原始位置: {original_pos}")
                print(f"   新位置: {new_pos}")
                print(f"   偏移量: [{offset_x:+.3f}, {offset_y:+.3f}, 0.000]m")
                return new_pos
            else:
                print(f"   ⚠️ 尝试 {attempt + 1}/{max_attempts} 检测到碰撞，重新生成...")
        
        # 所有尝试都失败，恢复原始位置
        print(f"❌ 随机偏移失败：{max_attempts} 次尝试均检测到碰撞或超出桌面，使用原始位置")
        translate_op.Set(Gf.Vec3d(float(original_pos[0]), float(original_pos[1]), float(original_pos[2])))
        return original_pos
        
    except Exception as e:
        print(f"⚠️ 随机偏移异常: {e}，使用原始位置")
        return original_pos


def calculate_height_offset(
    pick_obj_path: str, 
    pick_pos: np.ndarray, 
    place_obj_path: str,
    place_pos: np.ndarray,
    target_offset_from_top: float = 0.03
):
    """
    计算抓取和放置高度偏移
    
    参数:
        pick_obj_path: 抓取物体的 USD 路径
        pick_pos: 抓取物体中心位置 (世界坐标)
        place_obj_path: 放置物体的 USD 路径
        place_pos: 放置物体中心位置 (世界坐标)
        target_offset_from_top: 目标位置相对于物体顶部的偏移 (默认 0.03m)
        
    返回:
        (pick_height_offset, placing_height_offset): 抓取和放置的高度偏移
        如果计算失败返回 None
    """
    # 获取抓取物体的 bounding box
    pick_bbox = get_object_bounding_box(pick_obj_path)
    if pick_bbox is None:
        print(f"⚠️ 无法获取抓取物体 bounding box，无法自动计算高度偏移")
        return None
    
    pick_min, pick_max = pick_bbox
    pick_object_bottom_z = pick_min[2]
    pick_object_top_z = pick_max[2]
    pick_object_height = pick_object_top_z - pick_object_bottom_z
    
    # 计算抓取高度偏移
    if pick_object_height <= 0.04:
        pick_height_offset = 0.0
        print(f"🔧 自动计算抓取高度偏移:")
        print(f"   抓取物体中心 Z: {pick_pos[2]:.3f}m")
        print(f"   抓取物体高度: {pick_object_height:.3f}m")
        print(f"   ⚠️ 物体高度 ≤ {target_offset_from_top}m，使用偏移 0（抓取中心）")
    else:
        # 目标位置：顶部 - target_offset_from_top
        target_z = pick_object_top_z - target_offset_from_top
        pick_height_offset = target_z - pick_pos[2]
        
        print(f"🔧 自动计算抓取高度偏移:")
        print(f"   抓取物体中心 Z: {pick_pos[2]:.3f}m")
        print(f"   抓取物体顶部 Z: {pick_object_top_z:.3f}m")
        print(f"   抓取物体高度: {pick_object_height:.3f}m")
        print(f"   目标 Z: {target_z:.3f}m (顶部 - {target_offset_from_top}m)")
        print(f"   抓取偏移: {pick_height_offset:.3f}m")
    
    # 获取放置物体的 bounding box
    place_bbox = get_object_bounding_box(place_obj_path)
    if place_bbox is None:
        print(f"⚠️ 无法获取放置物体 bounding box，放置偏移使用抓取偏移")
        placing_height_offset = pick_height_offset
    else:
        place_min, place_max = place_bbox
        place_object_top_z = place_max[2]
        
        # 放置高度 = 放置物体顶部 + 抓取物体高度
        target_place_z = place_object_top_z + pick_object_height
        placing_height_offset = target_place_z - place_pos[2]
        
        print(f"🔧 自动计算放置高度偏移:")
        print(f"   放置物体中心 Z: {place_pos[2]:.3f}m")
        print(f"   放置物体顶部 Z: {place_object_top_z:.3f}m")
        print(f"   目标放置 Z: {target_place_z:.3f}m (放置物体顶部 + 抓取物体高度 {pick_object_height:.3f}m)")
        print(f"   放置偏移: {placing_height_offset:.3f}m")
    
    return (pick_height_offset, placing_height_offset)


# 加载场景 USD
open_stage("/home/di-gua/licheng/manipulation/Collected_World1/World0.usd")
simulation_app.update()

# 创建 World
my_world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0/60.0,  # 60 Hz 物理仿真
    rendering_dt=1.0/60.0,  # 60 Hz 渲染
)

# 对象引用已移至 step_once 函数参数中

# ============================================================
# 数据采集模式配置变量（供 collect_curobo.py 使用）
# ============================================================
_COLLECT_PICK_OBJ_PATH = "/World/Vegetable_9"
_COLLECT_PLACE_OBJ_PATH = "/World/Bowl_0"
_COLLECT_AUTO_HEIGHT_OFFSET = True
_COLLECT_PICK_HEIGHT_OFFSET = 0.23
_COLLECT_PLACING_HEIGHT_OFFSET = 0.23
_COLLECT_EEF_LATERAL_OFFSET = None
_COLLECT_APPROACH_HEIGHT = 0.15
_COLLECT_LIFT_HEIGHT = 0.05
_COLLECT_USE_SEED_MODEL = False
_COLLECT_SEED_IMAGE_PATH = "/home/di-gua/data/seed-one-errors.png"
_COLLECT_SEED_OBJECT_NAME = "bottle"
_COLLECT_RENDER = True
_COLLECT_RANDOMIZE_PICK_POSITION = True  # 是否随机化抓取物体位置
_COLLECT_POSITION_OFFSET_RANGE = 0.1  # 位置偏移范围（米）
_COLLECT_SCENE_OBJECTS = [                # 场景中其他物体（用于碰撞检测）
        "/World/Bowl_0",
        "/World/CuttingBoard_4",
        "/World/SaltShaker_3",
        # ... 添加所有可能碰撞的物体
    ] # 场景物体列表（用于碰撞检测）

# Episode 状态标志（供 collect_curobo.py 检测）
_EPISODE_FAILED = False  # 标记当前 episode 是否因规划失败而放弃
_MAX_PLAN_FAILURES = 10  # 最大规划失败次数阈值

def configure_collection(
    pick_obj: str = None,
    place_obj: str = None,
    auto_height_offset: bool = None,
    pick_height_offset: float = None,
    placing_height_offset: float = None,
    eef_lateral_offset = None,
    approach_height: float = None,
    lift_height: float = None,
    use_seed_model: bool = None,
    seed_image_path: str = None,
    seed_object_name: str = None,
    render: bool = None,
    randomize_pick_position: bool = None,
    position_offset_range: float = None,
    scene_objects: list = None,
):
    """
    配置数据采集模式的参数
    
    参数:
        pick_obj: 抓取物体路径
        place_obj: 放置物体路径
        auto_height_offset: 是否自动计算高度偏移
        pick_height_offset: 手动抓取高度偏移
        placing_height_offset: 手动放置高度偏移
        eef_lateral_offset: 末端执行器横向偏移
        approach_height: 接近高度
        lift_height: 抬升高度
        use_seed_model: 是否使用 Seed 模型
        seed_image_path: Seed 模型图片路径
        seed_object_name: Seed 模型物体名称
        render: 是否渲染
    """
    global _COLLECT_PICK_OBJ_PATH, _COLLECT_PLACE_OBJ_PATH
    global _COLLECT_AUTO_HEIGHT_OFFSET, _COLLECT_PICK_HEIGHT_OFFSET
    global _COLLECT_PLACING_HEIGHT_OFFSET, _COLLECT_EEF_LATERAL_OFFSET
    global _COLLECT_APPROACH_HEIGHT, _COLLECT_LIFT_HEIGHT
    global _COLLECT_USE_SEED_MODEL, _COLLECT_SEED_IMAGE_PATH
    global _COLLECT_SEED_OBJECT_NAME, _COLLECT_RENDER
    global _COLLECT_RANDOMIZE_PICK_POSITION, _COLLECT_POSITION_OFFSET_RANGE
    global _COLLECT_SCENE_OBJECTS
    
    if pick_obj is not None:
        _COLLECT_PICK_OBJ_PATH = pick_obj
        print(f"🔧 配置抓取物体: {pick_obj}")
    if place_obj is not None:
        _COLLECT_PLACE_OBJ_PATH = place_obj
        print(f"🔧 配置放置物体: {place_obj}")
    if auto_height_offset is not None:
        _COLLECT_AUTO_HEIGHT_OFFSET = auto_height_offset
    if pick_height_offset is not None:
        _COLLECT_PICK_HEIGHT_OFFSET = pick_height_offset
    if placing_height_offset is not None:
        _COLLECT_PLACING_HEIGHT_OFFSET = placing_height_offset
    if eef_lateral_offset is not None:
        _COLLECT_EEF_LATERAL_OFFSET = eef_lateral_offset
    if approach_height is not None:
        _COLLECT_APPROACH_HEIGHT = approach_height
    if lift_height is not None:
        _COLLECT_LIFT_HEIGHT = lift_height
    if use_seed_model is not None:
        _COLLECT_USE_SEED_MODEL = use_seed_model
    if seed_image_path is not None:
        _COLLECT_SEED_IMAGE_PATH = seed_image_path
    if seed_object_name is not None:
        _COLLECT_SEED_OBJECT_NAME = seed_object_name
    if render is not None:
        _COLLECT_RENDER = render
    if randomize_pick_position is not None:
        _COLLECT_RANDOMIZE_PICK_POSITION = randomize_pick_position
    if position_offset_range is not None:
        _COLLECT_POSITION_OFFSET_RANGE = position_offset_range
    if scene_objects is not None:
        _COLLECT_SCENE_OBJECTS = scene_objects

# 固定放置位置
fixed_spawn_pos = np.array([0.0, 0.45, -0.25], dtype=float)

# 🔄 基座旋转配置：是否通过代码旋转基座（而不是修改USD）
# 设置为 True 时，基座将顺时针旋转90度（从侧面朝向桌子变为正面朝向桌子）
ENABLE_BASE_ROTATION = True  # 设置为 False 则不旋转
BASE_ROTATION_DEGREES = -90.0  # 顺时针旋转90度（负值表示顺时针）

FRANKA_LOCAL_USD = "Franka_usd/Franka.usd"
FRANKA_REFERENCE_PATH = "/World/Franka"
FRANKA_NESTED_PATH = "/World/Franka/franka"


def _get_base_rotation_quaternion(rotation_degrees: float):
    """
    生成绕Z轴旋转的四元数（用于基座旋转）
    
    参数:
        rotation_degrees: 旋转角度（度），负值表示顺时针
        
    返回:
        四元数 (w, x, y, z) 格式
    """
    from scipy.spatial.transform import Rotation as R
    
    # 绕Z轴旋转（Z轴向上，从Z轴正方向看，负角度是顺时针）
    rotation = R.from_euler('z', rotation_degrees, degrees=True)
    quat_xyzw = rotation.as_quat()  # 返回 (x, y, z, w)
    
    # 转换为 (w, x, y, z) 格式（Isaac Sim 格式）
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    return quat_wxyz


def _ensure_local_franka_loaded():
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("无法获取 USD stage")
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
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path.lower().endswith("panda_link0"):
            return path.rsplit("/", 1)[0]
    raise RuntimeError("无法在 Stage 中找到 Franka articulation root")


# 加载 Franka
_ensure_local_franka_loaded()
franka_prim_path = _resolve_franka_prim_path()

if Franka is None:
    raise RuntimeError("未找到 Franka 包装类(omni.isaac.franka)")

if is_prim_path_valid(franka_prim_path):
    my_franka = Franka(prim_path=franka_prim_path, name="Franka")
    try:
        # 🔄 如果启用了基座旋转，同时设置位置和旋转
        if ENABLE_BASE_ROTATION:
            base_rotation_quat = _get_base_rotation_quaternion(BASE_ROTATION_DEGREES)
            print(f"🔄 通过代码旋转基座: {BASE_ROTATION_DEGREES}° (顺时针)")
            print(f"   旋转四元数 (w, x, y, z): {base_rotation_quat}")
            my_franka.set_world_pose(position=fixed_spawn_pos, orientation=base_rotation_quat)
        else:
            my_franka.set_world_pose(position=fixed_spawn_pos)
    except Exception:
        # 备用方法：使用 XFormPrim
        if ENABLE_BASE_ROTATION:
            base_rotation_quat = _get_base_rotation_quaternion(BASE_ROTATION_DEGREES)
            print(f"🔄 通过代码旋转基座（备用方法）: {BASE_ROTATION_DEGREES}°")
            XFormPrim(franka_prim_path).set_world_pose(
                position=fixed_spawn_pos, 
                orientation=base_rotation_quat
            )
        else:
            XFormPrim(franka_prim_path).set_world_pose(position=fixed_spawn_pos)
    simulation_app.update()
else:
    # 创建时设置位置（如果需要旋转，后续再设置）
    my_franka = Franka(prim_path=franka_prim_path, name="Franka", position=fixed_spawn_pos)
    if ENABLE_BASE_ROTATION:
        base_rotation_quat = _get_base_rotation_quaternion(BASE_ROTATION_DEGREES)
        print(f"🔄 通过代码旋转基座（创建后）: {BASE_ROTATION_DEGREES}°")
        try:
            my_franka.set_world_pose(position=fixed_spawn_pos, orientation=base_rotation_quat)
        except Exception:
            XFormPrim(franka_prim_path).set_world_pose(
                position=fixed_spawn_pos, 
                orientation=base_rotation_quat
            )
        simulation_app.update()

my_world.scene.add(my_franka)
my_world.reset()

# 设置初始关节位置到 home position
try:
    home_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Franka home pose
    # 只设置前7个关节（机械臂关节），不包括夹爪
    current_positions = my_franka.get_joint_positions()
    new_positions = current_positions.copy()
    new_positions[:7] = home_joints
    my_franka.set_joint_positions(new_positions)
    simulation_app.update()
    print(f"✅ 设置初始关节位置: {home_joints}")
except Exception as e:
    print(f"⚠️ 无法设置初始关节位置: {e}")


# ==========================
# CuRobo 控制器（替代 PickPlaceController）
# ==========================
class CuroboPickPlaceController:
    """
    基于 CuRobo MotionGen 的 Pick-Place 控制器
    接口与原 PickPlaceController 兼容
    """
    
    def __init__(self, name, gripper, robot_articulation, franka_prim_path):
        self.name = name
        self.gripper = gripper
        self.robot = robot_articulation
        self.franka_prim_path = franka_prim_path
        
        # 高度常量 - 用于所有事件的高度计算
        self.approach_height = 0.06  # 接近高度
        self.lift_height = 0.13  # 提升高度
        
        # TCP 偏移补偿：panda_hand 到夹爪指尖的距离
        # CuRobo 使用 panda_hand 作为 ee_link，但实际接触点在指尖
        # 这个偏移量补偿了从 panda_hand 到指尖的 Z 轴距离
        # 注意：根据实际测试调整，考虑夹爪闭合后的实际接触点
        self.tcp_z_offset = 0.11
        
        # 获取机器人基座的世界位置和姿态（包括代码设置的旋转）
        robot_base_prim = XFormPrim(franka_prim_path)
        positions, orientations = robot_base_prim.get_world_poses()
        self.robot_base_position = positions[0]  # 取第一个元素
        self.robot_base_orientation = orientations[0]
        print(f"🤖 机器人基座世界位置: {self.robot_base_position}")
        print(f"🤖 机器人基座世界姿态 (w, x, y, z): {self.robot_base_orientation}")
        
        # 🔍 验证基座旋转角度（帮助调试）
        from scipy.spatial.transform import Rotation as R
        base_quat_xyzw = np.array([
            self.robot_base_orientation[1], 
            self.robot_base_orientation[2], 
            self.robot_base_orientation[3], 
            self.robot_base_orientation[0]
        ])
        base_rotation = R.from_quat(base_quat_xyzw)
        base_euler = base_rotation.as_euler('xyz', degrees=True)
        print(f"🔍 基座旋转角度 (Euler XYZ, 度): {base_euler}")
        if np.allclose(base_euler, [0, 0, 0], atol=1.0):
            print(f"   ✅ 基座未旋转（或旋转很小），坐标系对齐")
        else:
            print(f"   ⚠️ 基座已旋转，坐标变换将自动处理")
        
        # 控制关节
        self.cmd_js_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        
        self.tensor_args = TensorDeviceType()
        
        # 加载机器人配置
        print("🚀 初始化 CuRobo MotionGen...")
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        robot_cfg["kinematics"]["base_link"] = "panda_link0"
        robot_cfg["kinematics"]["ee_link"] = "panda_hand"
        robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}
        robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_collision_mesh.yml"
        
        # UsdHelper
        self.usd_help = UsdHelper()
        self.usd_help.load_stage(my_world.stage)
        
        # 世界配置
        world_cfg_table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        self._world_cfg_table = world_cfg_table
        
        # 获取场景障碍物
        self._update_world()
        
        # MotionGen 配置
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            self._world_cfg,
            self.tensor_args,
            trajopt_tsteps=32,  # 增加优化步数，生成更平滑的轨迹
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=0.03,  # 30ms 时间步长，降低控制频率，减少抖动
            collision_cache={"obb": 50, "mesh": 30},
            collision_activation_distance=0.01,  # 🔑 放宽到 25mm，降低碰撞检测敏感度
            # 添加平滑参数
            smooth_weight=[100.0, 50.0, 10.0],  # 位置、速度、加速度平滑权重
            velocity_scale=0.75,  # 降低速度，增加稳定性
            acceleration_scale=0.75,  # 降低加速度，减少抖动
        )
        
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(parallel_finetune=True)
        self.motion_gen.update_world(self._world_cfg.get_collision_check_world())
        
        # 规划配置
        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            max_attempts=50,  # 🔑 增加到 50 次尝试
            enable_graph_attempt=25,  # 🔑 增加图搜索尝试到 25 次
            enable_finetune_trajopt=True,
            parallel_finetune=True,
            time_dilation_factor=1.0,
            timeout=15.0,  # 🔑 增加超时到 15 秒
        )
        
        # 状态
        self.current_event = 0
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.idx_list = None
        self.is_attached = False
        self.wait_counter = 0  # 等待计数器
        self.wait_steps = 50   # 增加等待步数，确保夹爪完全闭合
        self.saved_pick_position = None  # 保存抓取时的位置，避免提升时跟踪移动物体
        self.plan_fail_counter = 0  # 规划失败计数器
        self.is_planning = False  # 标记是否正在进行运动规划（用于采集器暫停采集）
        
        # 🔑 关键：夹爪闭合位置，用于在移动时保持夹持状态
        self.gripper_closed_position = 0.0  # 夹爪闭合时的位置（每个手指）
        
        # 阈值
        self.position_threshold = 0.02  # 8cm（放宽阈值，因为轨迹执行完即可认为到达）
        
        # 🎯 随机抓取姿态生成器
        self.use_random_grasp = True  # 启用随机抓取姿态
        self.current_grasp_quat = None  # 当前生成的抓取姿态
        self.current_place_quat = None  # 当前生成的放置姿态
        
        # 🎯 目标物体路径（用于动态附着）
        self.target_object_path = None  # 将由 step_once 设置
        
        print("✅ CuRobo MotionGen 初始化完成")
    
    def reached_target(self, target_position) -> bool:
        """检查是否到达目标位置（参考 simple_stacking.py）"""
        # 首先检查轨迹是否执行完毕
        if self.cmd_plan is not None:
            return False  # 还在执行轨迹，未到达
        
        # 轨迹执行完毕，检查位置精度
        try:
            # 获取末端执行器位置 - 尝试多种方法
            ee_position = None
            
            # 方法1: 通过 end_effector 属性
            if hasattr(self.robot, 'end_effector') and self.robot.end_effector is not None:
                ee_position = self.robot.end_effector.get_world_pose()[0]
            # 方法2: 通过 panda_hand prim
            else:
                from isaacsim.core.prims import XFormPrim
                hand_prim = XFormPrim(f"{self.franka_prim_path}/panda_hand")
                positions, _ = hand_prim.get_world_poses()
                ee_position = positions[0]
            
            if ee_position is None:
                print(f"   ⚠️ 无法获取末端位置，但轨迹已执行完，认为已到达")
                return True  # 轨迹执行完，即使无法获取位置也认为到达
            
            distance = np.linalg.norm(target_position - ee_position)
            
            # 到达条件：距离 < 阈值
            # 如果轨迹执行完毕，即使距离稍大也认为到达（避免因控制精度卡住）
            reached = distance < self.position_threshold
            
            # 调试输出
            print(f"   📏 距离目标: {distance:.4f}m, 阈值: {self.position_threshold}m, 到达: {reached}")
            
            # 如果距离在合理范围内（< 0.12m），也认为到达（轨迹执行完即可）
            if not reached and distance < 0.12:
                print(f"   ✅ 轨迹执行完毕，距离 {distance:.4f}m 在合理范围内，认为已到达")
                return True
            
            return reached
        except Exception as e:
            print(f"   ⚠️ reached_target 异常: {e}")
            import traceback
            traceback.print_exc()
            # 如果无法判断，但轨迹执行完，默认认为已到达（避免卡死）
            return True
    
    def _update_world(self):
        """更新世界障碍物"""
        # 暂时忽略所有物体，只保留桌子作为障碍物
        ignore_substring = [
            self.franka_prim_path, 
            "/SimpleRoom",  # 忽略 SimpleRoom 下的所有碰撞物体（地板、毛巾等）
            _COLLECT_PICK_OBJ_PATH,
            "/World/Table_1",        
        ]
        obstacles = self.usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            ignore_substring=ignore_substring,
            reference_prim_path=self.franka_prim_path,
        )
        obstacles.add_obstacle(self._world_cfg_table.cuboid[0])
        self._world_cfg = obstacles
    
    def forward(self, picking_position, placing_position, current_joint_positions, end_effector_offset=None):
        """
        主控制接口 - 使用 simple_stacking.py 的状态机逻辑
        
        Args:
            picking_position: 抓取位置
            placing_position: 放置位置  
            current_joint_positions: 当前关节位置
            end_effector_offset: 末端偏移
        
        Returns:
            ArticulationAction
        """
        if end_effector_offset is None:
            end_effector_offset = np.zeros(3)
        
        # 如果正在执行轨迹，继续执行
        if self.cmd_plan is not None:
            return self._execute_trajectory()
        
        # 如果在等待（夹爪动作），继续等待
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return None
        
        # 根据当前事件规划下一个动作
        target_pose = self._get_target_pose(picking_position, placing_position, end_effector_offset)
        
        if target_pose is not None:
            success = self._plan_to_pose(target_pose)
            if not success:
                print(f"⚠️ Event {self.current_event} 规划失败")
                return ArticulationAction(
                    current_joint_positions[:7],
                    joint_indices=list(range(7)),
                )
        
        # 执行轨迹
        return self._execute_trajectory()
    
    def _get_target_pose(self, picking_position, placing_position, offset):
        """根据当前事件获取目标位姿（支持随机抓取和放置姿态）
        
        关键修复：当夹爪倾斜时，需要反向补偿位置，确保夹爪末端（而非 panda_hand）到达目标点
        """
        from scipy.spatial.transform import Rotation as R
        
        # 🎯 根据当前事件选择使用抓取姿态还是放置姿态
        if self.current_event in [0, 1, 2]:  # Event 0-2: 使用抓取姿态
            if self.current_grasp_quat is not None:
                ee_quat = self.current_grasp_quat
            else:
                # 默认：末端朝下的四元数 [w, x, y, z]
                ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        else:  # Event 3-6: 使用放置姿态
            if self.current_place_quat is not None:
                ee_quat = self.current_place_quat
            else:
                # 默认：末端朝下的四元数 [w, x, y, z]
                ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        
        tcp_offset_local = np.array([0.0, 0.0, -self.tcp_z_offset])
        
        quat_xyzw = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
        rotation = R.from_quat(quat_xyzw)

        tcp_offset_world = rotation.apply(tcp_offset_local)

        # 调试输出（仅在事件切换时打印）
        if self.current_event in [0, 1, 3, 4, 6] and self.cmd_plan is None:
            print(f"🔧 TCP 偏移补偿 (Event {self.current_event}):")
            print(f"   局部偏移 (panda_hand坐标系): {tcp_offset_local}")
            print(f"   姿态四元数 (wxyz): {ee_quat}")
            print(f"   旋转后世界偏移: {tcp_offset_world}")
        
        if self.current_event == 0:  # 接近抓取
            # panda_hand 目标位置 = 抓取点 + 接近高度 - TCP偏移（反向补偿）
            pos = picking_position + np.array([0, 0, self.approach_height]) + tcp_offset_world + offset
            return (pos, ee_quat)
        elif self.current_event == 1:  # 下降抓取
            # panda_hand 目标位置 = 抓取点 - TCP偏移（反向补偿）
            pos = picking_position + tcp_offset_world + offset
            return (pos, ee_quat)
        elif self.current_event == 2:  # 抓取（夹爪控制移到主循环）
            return None  # 不规划，等待夹爪闭合
        elif self.current_event == 3:  # 附着物体并直接移动到接近放置位置
            if not self.is_attached and self.target_object_path:
                self._attach_object(self.target_object_path)
            # 附着后直接移动到放置位置上方
            pos = placing_position + np.array([0, 0, self.approach_height]) + tcp_offset_world + offset
            return (pos, ee_quat)
        elif self.current_event == 4:  # 下降放置
            # panda_hand 目标位置 = 放置点 - TCP偏移（反向补偿）
            pos = placing_position + tcp_offset_world + offset
            return (pos, ee_quat)
        elif self.current_event == 5:  # 放置（夹爪控制移到主循环）
            return None  # 不规划，等待夹爪打开
        elif self.current_event == 6:  # 分离物体并后退
            if self.is_attached:
                self._detach_object()
            # 后退时也需要补偿
            pos = placing_position + np.array([0, 0, self.lift_height]) + tcp_offset_world + offset
            return (pos, ee_quat)
            return (pos, ee_quat)   
        else:  # Event 7+: 完成
            return None
    
    def _plan_to_pose(self, target_pose):
        """规划到目标位姿"""
        pos_world, quat_world = target_pose
        
        # 调试输出：世界坐标
        print(f"📍 Event {self.current_event}:")
        print(f"   世界坐标目标位置（Panda hand）: {pos_world}")
        print(f"   机器人基座位置: {self.robot_base_position}")
        print(f"   机器人基座姿态 (wxyz): {self.robot_base_orientation}")
        
        # ✅ CuRobo 使用相对于机器人基座的坐标系
        # 需要进行完整的坐标变换：平移 + 旋转
        
        # 1. 平移变换：世界坐标 → 相对于基座原点的坐标
        pos_translated = pos_world - self.robot_base_position
        
        # 2. 旋转变换：将世界坐标系中的向量旋转到基座坐标系
        # 使用基座姿态的逆四元数进行旋转
        from scipy.spatial.transform import Rotation as R
        
        # 基座姿态四元数 (wxyz 格式)
        base_quat_wxyz = self.robot_base_orientation
        # 转换为 scipy 格式 (xyzw)
        base_quat_xyzw = np.array([base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]])
        
        # 创建旋转对象并求逆（从世界坐标系到基座坐标系）
        base_rotation = R.from_quat(base_quat_xyzw)
        base_rotation_inv = base_rotation.inv()
        
        # 应用逆旋转
        pos_relative = base_rotation_inv.apply(pos_translated)
        
        print(f"   平移后位置: {pos_translated}")
        print(f"   旋转后位置（相对基座）: {pos_relative}")
        
        # 🔑 关键修复：姿态也需要从世界坐标系变换到基座坐标系
        # 将世界坐标系中的姿态四元数转换为基座坐标系中的姿态
        # quat_world 是 (w, x, y, z) 格式
        quat_world_xyzw = np.array([quat_world[1], quat_world[2], quat_world[3], quat_world[0]])  # 转换为 (x, y, z, w)
        world_rotation = R.from_quat(quat_world_xyzw)
        
        # 组合旋转：基座逆旋转 * 世界姿态 = 基座坐标系中的姿态
        relative_rotation = base_rotation_inv * world_rotation
        quat_relative_xyzw = relative_rotation.as_quat()  # 返回 (x, y, z, w)
        quat_relative = np.array([quat_relative_xyzw[3], quat_relative_xyzw[0], quat_relative_xyzw[1], quat_relative_xyzw[2]])  # 转换回 (w, x, y, z)
        
        # 使用相对坐标
        pos = pos_relative
        quat = quat_relative
        
        print(f"   传入 CuRobo 的位置: {pos}")
        print(f"   传入 CuRobo 的姿态（世界坐标系）: {quat_world}")
        print(f"   传入 CuRobo 的姿态（基座坐标系）: {quat}")
        
        ik_goal = Pose(
            position=self.tensor_args.to_device(pos),
            quaternion=self.tensor_args.to_device(quat),
        )
        
        # 获取当前关节状态
        sim_js = self.robot.get_joints_state()
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=self.robot.dof_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        
        # 执行规划（设置规划状态标志）
        self.is_planning = True
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config.clone())
        self.is_planning = False
        
        if result.success.item():
            cmd_plan = result.get_interpolated_plan()
            self.idx_list = [i for i in range(len(self.cmd_js_names))]
            self.cmd_plan = cmd_plan.get_ordered_joint_state(self.cmd_js_names)
            self.cmd_idx = 0
            self._step_idx = 0
            print(f"✅ 规划成功 (Event {self.current_event}), 轨迹长度: {len(self.cmd_plan.position)}")
            self.plan_fail_counter = 0  # 重置失败计数
            return True
        else:
            print(f"❌ 规划失败 (Event {self.current_event}), 失败次数: {self.plan_fail_counter + 1}")
            self.plan_fail_counter += 1
            
            # 🔑 检查是否超过失败阈值，标记 episode 失败
            global _EPISODE_FAILED, _MAX_PLAN_FAILURES
            if self.plan_fail_counter >= _MAX_PLAN_FAILURES:
                _EPISODE_FAILED = True
                print(f"⚠️⚠️⚠️ 规划失败次数达到阈值 ({_MAX_PLAN_FAILURES})，放弃当前 episode")
            
            return False
    
    def _execute_trajectory(self):
        """执行当前轨迹"""
        if self.cmd_plan is None:
            return None  # 等待主循环切换事件
        
        # 每2步发送一次指令（与 interpolation_dt=0.03 配合，约 60Hz 控制频率）
        if self._step_idx % 2 == 0:
            cmd_state = self.cmd_plan[self.cmd_idx]
            self.cmd_idx += 1
            
            # 🔑 关键修复：在 Event 3-5 期间，在 ArticulationAction 中包含夹爪关节
            # 这样可以防止手臂运动时覆盖夹爪控制
            if self.current_event in [3, 4, 5]:  # 抬起、移动、下降放置阶段
                # 包含手臂关节 (0-6) + 夹爪关节 (7-8)
                positions = np.concatenate([
                    cmd_state.position.cpu().numpy(),  # 手臂关节位置
                    np.array([self.gripper_closed_position, self.gripper_closed_position])  # 夹爪保持闭合
                ])
                velocities = np.concatenate([
                    cmd_state.velocity.cpu().numpy() * 0.0,
                    np.array([0.0, 0.0])  # 夹爪速度为0
                ])
                
                # 🔑🔑 关键：为夹爪添加持续的闭合力矩（努力值）
                # 手臂关节不使用力控制（None），夹爪关节施加较大的闭合力
                gripper_force = 200.0  # 夹爪闭合力（牛顿），可根据物体重量调整
                efforts = np.concatenate([
                    np.zeros(7),  # 手臂关节不使用力控制
                    np.array([-gripper_force, -gripper_force])  # 夹爪施加闭合力（负值表示闭合方向）
                ])
                
                joint_indices = list(range(9))  # 0-8: 所有关节
                
                art_action = ArticulationAction(
                    positions,
                    velocities,
                    efforts,  # 添加力矩控制
                    joint_indices=joint_indices,
                )
            else:
                # Event 0-2, 5-6: 只控制手臂关节
                # 🔑 保留速度信息，使用 CuRobo 规划的速度而不是清零
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    cmd_state.velocity.cpu().numpy(),  # 保留规划速度，提升平滑度
                    joint_indices=self.idx_list,
                )
            
            if self.cmd_idx >= len(self.cmd_plan.position):
                self.cmd_plan = None
                self.cmd_idx = 0
            
            self._step_idx += 1
            return art_action
        else:
            self._step_idx += 1
            return None
    
    def _attach_object(self, target_object_path: str):
        """附着物体
        
        Args:
            target_object_path: 要附加的物体的 USD 路径，例如 "/World/Bottle_2"
        """
        # 🔑 优化：直接指定要附加的物体路径，而不是通过排除法
        print(f"🔄 直接获取目标物体: {target_object_path}")
        
        # 方法1: 尝试直接从 stage 获取单个物体
        try:
            # 使用 only_paths 直接指定物体路径
            obstacles_with_object = self.usd_help.get_obstacles_from_stage(
                only_paths=[target_object_path],
                reference_prim_path=self.franka_prim_path,
            )
            
            # 检查是否成功获取到物体
            object_name = None
            if obstacles_with_object.mesh is not None and len(obstacles_with_object.mesh) > 0:
                object_name = obstacles_with_object.mesh[0].name
                print(f"✅ 直接获取到物体: {object_name}")
            else:
                # 备用：使用路径作为名称
                object_name = target_object_path
                print(f"⚠️ 未找到 mesh，使用路径: {object_name}")
            
            # 更新世界模型（包含目标物体）
            collision_world = obstacles_with_object.get_collision_check_world()
            self.motion_gen.update_world(collision_world)
            
        except Exception as e:
            print(f"⚠️ 直接获取物体失败: {e}，使用备用方法")
            # 备用方法：使用路径
            object_name = target_object_path
        
        # 附加物体到机器人
        sim_js = self.robot.get_joints_state()
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=self.robot.dof_names,
        )
        
        try:
            print(f"📦 尝试附加物体: {object_name}")
            self.motion_gen.attach_objects_to_robot(
                cu_js,
                [object_name],
                sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_args),
            )
            self.is_attached = True
            print("✅ 物体附加成功")
            
            # ✅ 重要：附加成功后，更新世界模型，移除已附着的物体
            # 因为物体现在附着在机器人上，不应该再作为独立的障碍物
            print("🔄 更新世界模型，移除已附着的物体...")
            self._update_world()
            self.motion_gen.update_world(self._world_cfg.get_collision_check_world())
            print("✅ 世界模型已更新（物体已移除）")
        except Exception as e:
            print(f"❌ 附加物体失败: {e}")
            import traceback
            traceback.print_exc()
            # 即使失败也继续，避免程序崩溃
            self.is_attached = False
    
    def _detach_object(self):
        """分离物体"""
        try:
            if self.is_attached:
                self.motion_gen.detach_object_from_robot()
            self.is_attached = False
            self._update_world()
            self.motion_gen.update_world(self._world_cfg.get_collision_check_world())
            print("📤 已分离物体")
        except Exception as e:
            print(f"⚠️ 分离物体异常: {e}")
            self.is_attached = False  # 确保状态更新
    
    def reset(self):
        """重置控制器"""
        self.current_event = 0
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.wait_counter = 0
        self.plan_fail_counter = 0
        self.current_grasp_quat = None  # 重置抓取姿态
        self.current_place_quat = None  # 重置放置姿态
        if self.is_attached:
            self._detach_object()
    
    def get_current_event(self):
        """获取当前事件（兼容接口）"""
        return self.current_event
    
    def next_event(self):
        """切换到下一个事件"""
        if self.current_event < 8:
            self.current_event += 1
            print(f"✅ 切换到 Event {self.current_event}")


# 创建 CuRobo 控制器（替代 PickPlaceController）
my_controller = CuroboPickPlaceController(
    name="curobo_pick_place_controller",
    gripper=my_franka.gripper,
    robot_articulation=my_franka,
    franka_prim_path=franka_prim_path
)
articulation_controller = my_franka.get_articulation_controller()


def _force_open_gripper():
    try:
        open_action = my_franka.gripper.forward(action="open")
        articulation_controller.apply_action(open_action)
    except Exception:
        if hasattr(my_franka.gripper, "joint_opened_positions"):
            my_franka.gripper.set_joint_positions(my_franka.gripper.joint_opened_positions)


# 初始化：强制打开夹爪
_force_open_gripper()

reset_needed = False

# 🔄 Seed 模型结果缓存（避免重复调用）
_seed_grasp_params_cache = None  # 缓存格式: (z_rot, tilt_x, tilt_y)

# 🔄 高度偏移计算缓存（避免重复计算）
_height_offset_calculated = False
_cached_pick_height_offset = None
_cached_placing_height_offset = None

# 所有可调参数已移至 step_once 函数参数中


def step_once(
    pick_obj_path: str = None,
    place_obj_path: str = None,
    auto_height_offset: bool = None,
    pick_height_offset: float = None,
    placing_height_offset: float = None,
    eef_lateral_offset: np.ndarray = None,
    use_seed_model: bool = None,
    seed_image_path: str = None,
    seed_object_name: str = None,
    grasp_z_rotation: float = 45.0,
    grasp_tilt_x: float = 0.0,
    grasp_tilt_y: float = 0.0,
    place_z_rotation: float = 0.0,
    place_tilt_x: float = 0.0,
    place_tilt_y: float = 0.0,
    randomize_grasp_pose: bool = True,
    randomize_pick_position: bool = None,
    position_offset_range: float = 0.1,
    scene_objects: list = None,
    render: bool = None
) -> bool:
    """
    执行一次仿真和控制循环
    
    参数:
        pick_obj_path: 要抓取的物体的 USD 路径，例如 "/World/Bottle_2"
        place_obj_path: 放置目标物体的 USD 路径，例如 "/World/CuttingBoard_4"
        auto_height_offset: 是否自动计算高度偏移（基于物体 bounding box）
        pick_height_offset: 抓取时高度偏移（auto_height_offset=False 时使用）
        placing_height_offset: 放置时高度偏移（auto_height_offset=False 时使用）
        eef_lateral_offset: 夹取时末端偏移
        use_seed_model: 是否使用 Seed 模型估计抓取姿态
        seed_image_path: Seed 模型输入图像路径
        seed_object_name: 要抓取的物体名称（用于 Seed 模型 prompt）
        grasp_z_rotation: 手动指定的抓取 Z 轴旋转角度（度）
        grasp_tilt_x: 手动指定的抓取 X 轴倾斜角度（度）
        grasp_tilt_y: 手动指定的抓取 Y 轴倾斜角度（度）
        place_z_rotation: 手动指定的放置 Z 轴旋转角度（度）
        place_tilt_x: 手动指定的放置 X 轴倾斜角度（度）
        place_tilt_y: 手动指定的放置 Y 轴倾斜角度（度）
        randomize_grasp_pose: 是否对抓取姿态参数进行随机化偏移（-10~+10度）
        randomize_pick_position: 是否随机化抓取物体位置（仅 X-Y 平面）
        position_offset_range: 位置偏移范围（米），默认 ±0.1m
        scene_objects: 场景中其他物体的路径列表（用于碰撞检测）
        render: 是否渲染
        
    返回:
        bool: False 表示无需继续
    """
    global reset_needed, _height_offset_calculated, _cached_pick_height_offset, _cached_placing_height_offset
    global _EPISODE_FAILED

    # 使用配置变量作为默认值（支持数据采集模式）
    if pick_obj_path is None:
        pick_obj_path = _COLLECT_PICK_OBJ_PATH
    if place_obj_path is None:
        place_obj_path = _COLLECT_PLACE_OBJ_PATH
    if auto_height_offset is None:
        auto_height_offset = _COLLECT_AUTO_HEIGHT_OFFSET
    if pick_height_offset is None:
        pick_height_offset = _COLLECT_PICK_HEIGHT_OFFSET
    if placing_height_offset is None:
        placing_height_offset = _COLLECT_PLACING_HEIGHT_OFFSET
    if eef_lateral_offset is None and _COLLECT_EEF_LATERAL_OFFSET is not None:
        eef_lateral_offset = _COLLECT_EEF_LATERAL_OFFSET
    if use_seed_model is None:
        use_seed_model = _COLLECT_USE_SEED_MODEL
    if seed_image_path is None:
        seed_image_path = _COLLECT_SEED_IMAGE_PATH
    if seed_object_name is None:
        seed_object_name = _COLLECT_SEED_OBJECT_NAME
    if render is None:
        render = _COLLECT_RENDER
    if randomize_pick_position is None:
        randomize_pick_position = _COLLECT_RANDOMIZE_PICK_POSITION
    if position_offset_range is None:
        position_offset_range = _COLLECT_POSITION_OFFSET_RANGE
    if scene_objects is None:
        scene_objects = _COLLECT_SCENE_OBJECTS
    
    # 根据路径创建 XFormPrim 对象
    pick_obj = XFormPrim(pick_obj_path)
    place_obj = XFormPrim(place_obj_path)
    
    # 设置默认偏移
    if eef_lateral_offset is None:
        eef_lateral_offset = np.array([0.0, 0.0, 0.0])

    if not simulation_app.is_running():
        return False
    
    # 🔑 检查 episode 是否已标记为失败
    if _EPISODE_FAILED:
        print(f"⚠️ 检测到 episode 失败标志，返回 False 通知采集器停止")
        return False

    my_world.step(render=render)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed:

            my_world.reset()
            my_controller.reset()
            _force_open_gripper()
            # 重置高度偏移缓存
            _height_offset_calculated = False
            _cached_pick_height_offset = None
            _cached_placing_height_offset = None
            reset_needed = False
            
            # 🔑 重置 episode 失败标志
            _EPISODE_FAILED = False

        # 获取抓取物体与放置物体的世界位姿
        pick_positions, _ = pick_obj.get_world_poses()
        place_positions, _ = place_obj.get_world_poses()
        pick_pos = pick_positions[0]
        place_pos = place_positions[0]

        # 修正：使用物体顶部位置
        # 如果已经计算过，使用缓存的偏移值；否则使用函数参数
        if _height_offset_calculated and _cached_pick_height_offset is not None:
            current_pick_offset = _cached_pick_height_offset
            current_place_offset = _cached_placing_height_offset
        else:
            current_pick_offset = pick_height_offset
            current_place_offset = placing_height_offset
            
        picking_position = pick_pos + np.array([0.0, 0.00, current_pick_offset])
        placing_position = place_pos + np.array([0.0, 0.0, current_place_offset])
        
        # 首次执行时：生成抓取姿态并计算高度偏移
        if my_controller.get_current_event() == 0 and my_controller.cmd_plan is None:
            
            print(f"🔍 调试：进入 Event 0 初始化代码块")
            print(f"🔍 调试：randomize_pick_position = {randomize_pick_position}")
            
            # 🎲 随机化抓取物体位置（如果启用）
            if randomize_pick_position:
                print(f"\n🎲 开始随机化抓取物体位置...")
                pick_pos = randomize_object_position(
                    obj_path=pick_obj_path,
                    original_pos=pick_pos,
                    offset_range=position_offset_range,
                    max_attempts=100,
                    scene_objects=scene_objects,
                    safety_margin=0.05
                )
                # 更新 pick_positions（用于后续计算）
                pick_positions = np.array([pick_pos])
                print()
            else:
                print(f"🔍 调试：位置随机化已禁用")
            
            print(f"🎯 抓取物体位置: {pick_pos}")
            print(f"🎯 放置物体位置: {place_pos}")
            
            # 🔧 自动计算或使用手动指定的高度偏移（仅首次执行）
            if auto_height_offset and not _height_offset_calculated:
                result = calculate_height_offset(
                    pick_obj_path, pick_pos, 
                    place_obj_path, place_pos,
                    target_offset_from_top=0.03
                )
                if result is not None:
                    _cached_pick_height_offset, _cached_placing_height_offset = result
                    _height_offset_calculated = True  # 标记已计算
                    # 更新当前使用的偏移值
                    current_pick_offset = _cached_pick_height_offset
                    current_place_offset = _cached_placing_height_offset
                else:
                    print(f"⚠️ 使用手动偏移值: pick={pick_height_offset}, place={placing_height_offset}")
                    # 缓存手动值
                    _cached_pick_height_offset = pick_height_offset
                    _cached_placing_height_offset = placing_height_offset
            
            # 重新计算位置（使用更新后的偏移）
            picking_position = pick_pos + np.array([0.0, 0.00, current_pick_offset])
            placing_position = place_pos + np.array([0.0, 0.0, current_place_offset])
            
            print(f"🎯 抓取位置: {picking_position}")
            print(f"🎯 放置位置: {placing_position}")
            
            # 🔑 设置目标物体路径（用于后续附着操作）
            my_controller.target_object_path = pick_obj_path
            print(f"🎯 目标物体路径: {pick_obj_path}")
            
            # 🎯 生成抓取姿态
            if my_controller.use_random_grasp:
                global _seed_grasp_params_cache
                
                # 🎲 根据 randomize_grasp_pose 参数决定是否添加随机偏移
                if randomize_grasp_pose:
                    random_offset_range = 10.0  # 度
                    grasp_z_rotation_random = grasp_z_rotation + np.random.uniform(-random_offset_range, random_offset_range)
                    grasp_tilt_x_random = grasp_tilt_x + np.random.uniform(-random_offset_range, random_offset_range)
                    grasp_tilt_y_random = grasp_tilt_y + np.random.uniform(-random_offset_range, random_offset_range)
                    
                    place_z_rotation_random = place_z_rotation + np.random.uniform(-random_offset_range, random_offset_range)
                    place_tilt_x_random = place_tilt_x + np.random.uniform(-random_offset_range, random_offset_range)
                    place_tilt_y_random = place_tilt_y + np.random.uniform(-random_offset_range, random_offset_range)
                    
                    print(f"🎲 抓取姿态（基础 + 随机偏移）:")
                    print(f"   Z旋转: {grasp_z_rotation:.1f}° + {grasp_z_rotation_random - grasp_z_rotation:.1f}° = {grasp_z_rotation_random:.1f}°")
                    print(f"   X倾斜: {grasp_tilt_x:.1f}° + {grasp_tilt_x_random - grasp_tilt_x:.1f}° = {grasp_tilt_x_random:.1f}°")
                    print(f"   Y倾斜: {grasp_tilt_y:.1f}° + {grasp_tilt_y_random - grasp_tilt_y:.1f}° = {grasp_tilt_y_random:.1f}°")
                    print(f"🎲 放置姿态（基础 + 随机偏移）:")
                    print(f"   Z旋转: {place_z_rotation:.1f}° + {place_z_rotation_random - place_z_rotation:.1f}° = {place_z_rotation_random:.1f}°")
                    print(f"   X倾斜: {place_tilt_x:.1f}° + {place_tilt_x_random - place_tilt_x:.1f}° = {place_tilt_x_random:.1f}°")
                    print(f"   Y倾斜: {place_tilt_y:.1f}° + {place_tilt_y_random - place_tilt_y:.1f}° = {place_tilt_y_random:.1f}°")
                else:
                    # 不随机化，直接使用原参数
                    grasp_z_rotation_random = grasp_z_rotation
                    grasp_tilt_x_random = grasp_tilt_x
                    grasp_tilt_y_random = grasp_tilt_y
                    
                    place_z_rotation_random = place_z_rotation
                    place_tilt_x_random = place_tilt_x
                    place_tilt_y_random = place_tilt_y
                    
                    print(f"📝 使用原始抓取姿态参数（无随机偏移）:")
                    print(f"   Z旋转: {grasp_z_rotation:.1f}°")
                    print(f"   X倾斜: {grasp_tilt_x:.1f}°")
                    print(f"   Y倾斜: {grasp_tilt_y:.1f}°")
                    print(f"📝 使用原始放置姿态参数（无随机偏移）:")
                    print(f"   Z旋转: {place_z_rotation:.1f}°")
                    print(f"   X倾斜: {place_tilt_x:.1f}°")
                    print(f"   Y倾斜: {place_tilt_y:.1f}°")
                
                # 决定使用 Seed 模型还是手动参数
                if use_seed_model:
                    # 检查是否有缓存
                    if _seed_grasp_params_cache is not None:
                        print("\n" + "="*70)
                        print("♻️ 使用缓存的 Seed 模型抓取姿态（避免重复调用）")
                        print("="*70)
                        z_rot, tilt_x, tilt_y = _seed_grasp_params_cache
                        print(f"📦 缓存参数: Z={z_rot}°, X={tilt_x}°, Y={tilt_y}°")
                        print("="*70 + "\n")
                    else:
                        # 首次调用 Seed 模型
                        print("\n" + "="*70)
                        print("🤖 使用豆包 Seed 1.6 Vision 模型估计抓取姿态（首次调用）")
                        print("="*70)
                        try:
                            z_rot, tilt_x, tilt_y = estimate_grasp_pose(
                                image_path=seed_image_path,
                                object_name=seed_object_name
                            )
                            # 保存到缓存
                            _seed_grasp_params_cache = (z_rot, tilt_x, tilt_y)
                            print(f"💾 已缓存 Seed 模型结果，后续重置将直接使用")
                            print("="*70 + "\n")
                        except Exception as e:
                            print(f"❌ Seed 模型调用失败: {e}")
                            print("⚠️ 回退到手动指定参数")
                            z_rot, tilt_x, tilt_y = grasp_z_rotation_random, grasp_tilt_x_random, grasp_tilt_y_random
                            print("="*70 + "\n")
                else:
                    print("\n📝 使用手动指定的抓取姿态参数" + ("（带随机偏移）" if randomize_grasp_pose else "（无随机偏移）"))
                    z_rot, tilt_x, tilt_y = grasp_z_rotation_random, grasp_tilt_x_random, grasp_tilt_y_random
                
                # 生成抓取姿态四元数
                my_controller.current_grasp_quat = generate_grasp_pose(
                    z_rotation=z_rot,
                    tilt_x=tilt_x,
                    tilt_y=tilt_y
                )
                euler = quaternion_to_euler(my_controller.current_grasp_quat, degrees=True)
                print(f"🎯 最终抓取姿态:")
                print(f"   输入参数: Z={z_rot}°, X={tilt_x}°, Y={tilt_y}°")
                print(f"   四元数: {my_controller.current_grasp_quat}")
                print(f"   欧拉角 [roll, pitch, yaw]: [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]")
                
                # 🎯 生成放置姿态四元数（使用带随机偏移的参数）
                my_controller.current_place_quat = generate_grasp_pose(
                    z_rotation=place_z_rotation_random,
                    tilt_x=tilt_x,
                    tilt_y=tilt_y
                )
                place_euler = quaternion_to_euler(my_controller.current_place_quat, degrees=True)
                print(f"🎯 最终放置姿态:")
                print(f"   输入参数: Z={place_z_rotation_random}°, X={place_tilt_x_random}°, Y={place_tilt_y_random}°")
                print(f"   四元数: {my_controller.current_place_quat}")
                print(f"   欧拉角 [roll, pitch, yaw]: [{place_euler[0]:.1f}°, {place_euler[1]:.1f}°, {place_euler[2]:.1f}°]\n")

        current_joint_positions = my_franka.get_joint_positions()
        current_event = my_controller.get_current_event()

        # === 状态机逻辑（参考 simple_stacking.py）===
        
        # Event 0, 1, 3, 4, 6: 规划并执行到达目标
        if current_event in [0, 1, 3, 4, 6]:
            # ✅ Event 3-4 的夹爪控制已经整合到 _execute_trajectory() 的 ArticulationAction 中
            # 不再需要单独的夹爪命令，避免冲突
            
            actions = my_controller.forward(
                picking_position=picking_position,
                placing_position=placing_position,
                current_joint_positions=current_joint_positions,
                end_effector_offset=eef_lateral_offset
            )
            if actions is not None:
                articulation_controller.apply_action(actions)
            
            # 🔑 在状态机早期阶段(0/1)持续强制打开夹爪，避免靠近时碰撞或半闭合状态
            # 参考 pick_place.py 的实现
            try:
                if current_event < 2:  # Event 0, 1: 接近和下降抓取阶段
                    open_action = my_franka.gripper.forward(action="open")
                    articulation_controller.apply_action(open_action)
            except Exception:
                pass
            
            # Event 3 特殊处理：如果规划失败太多次，跳过移动阶段直接下降放置
            if current_event == 3 and my_controller.plan_fail_counter >= 10:
                print(f"⚠️ Event 3 规划失败 {my_controller.plan_fail_counter} 次，跳过移动阶段")
                my_controller.plan_fail_counter = 0
                my_controller.next_event()  # 跳到 Event 4
            
            # 检查是否到达目标（需要加上偏移，与规划目标保持一致）
            target_pos = None
            if current_event == 0:
                target_pos = picking_position + np.array([0, 0, my_controller.approach_height]) + eef_lateral_offset
            elif current_event == 1:
                target_pos = picking_position + eef_lateral_offset
            elif current_event == 3:
                # Event 3: 附着物体后直接移动到放置位置上方
                target_pos = placing_position + np.array([0, 0, my_controller.approach_height]) + eef_lateral_offset
            elif current_event == 4:
                target_pos = placing_position + eef_lateral_offset
            elif current_event == 6:
                target_pos = placing_position + np.array([0, 0, my_controller.lift_height]) + eef_lateral_offset
            
            if target_pos is not None:
                # 调试：打印目标位置和末端位置
                if my_controller.cmd_plan is None:  # 只在轨迹执行完后检查
                    try:
                        ee_pos = my_controller.robot.end_effector.get_world_pose()[0]
                        print(f"🔍 Event {current_event} 检查到达:")
                        print(f"   目标位置(世界)（手指末端）: {target_pos}")
                        print(f"   末端位置(世界)（手指末端）: {ee_pos}")
                    except Exception as e:
                        print(f"   ⚠️ 无法获取末端位置: {e}")
                
                if my_controller.reached_target(target_pos):
                    print(f"✅ Event {current_event} 到达目标")
                    my_controller.next_event()
        
        # Event 2: 抓取（使用力控制闭合夹爪）
        elif current_event == 2:
            # ✅ 使用力控制命令 gripper.forward(action="close")
            # 夹爪会自动感应物体并停止在接触面，无需手动设置宽度
            if my_controller.wait_counter == 0:
                print("🤏 开始闭合夹爪（力控制模式）...")
                my_controller.wait_counter = 100  # 增加等待时间，确保夹爪完全闭合并稳定
            
            # 持续发送闭合命令（力控制）
            try:
                close_action = my_franka.gripper.forward(action="close")
                articulation_controller.apply_action(close_action)
            except Exception as e:
                print(f"⚠️ 夹爪闭合命令失败: {e}")
            
            my_controller.wait_counter -= 1
            
            # 每15步打印一次进度
            if my_controller.wait_counter % 15 == 0:
                print(f"   🤏 夹爪闭合中... 剩余 {my_controller.wait_counter} 步")
            
            # 等待完成后进入下一阶段
            if my_controller.wait_counter == 0:
                print("📦 夹爪闭合完成，附加物体到 CuRobo")
                
                # 🔑 关键：读取并保存夹爪的实际闭合位置
                try:
                    gripper_positions = my_franka.gripper.get_joint_positions()
                    my_controller.gripper_closed_position = gripper_positions[0]  # 两个手指位置相同，取第一个
                    print(f"🔒 保存夹爪闭合位置: {my_controller.gripper_closed_position:.4f}")
                except Exception as e:
                    print(f"⚠️ 无法读取夹爪位置，使用默认值 0.0: {e}")
                    my_controller.gripper_closed_position = 0.0
                
                # 保存当前抓取位置
                my_controller.saved_pick_position = picking_position.copy()
                print(f"📍 保存抓取位置: {my_controller.saved_pick_position}")
                try:
                    # 使用控制器中保存的目标物体路径
                    my_controller._attach_object(my_controller.target_object_path)
                    if not my_controller.is_attached:
                        print("⚠️ 物体附加失败，但继续执行任务")
                except Exception as e:
                    print(f"❌ 附加物体异常: {e}")
                    import traceback
                    traceback.print_exc()
                my_controller.next_event()
        
        # Event 5: 放置（打开夹爪）
        elif current_event == 5:
            if my_controller.wait_counter == 0:
                print("✋ 打开夹爪...")
                my_controller.wait_counter = my_controller.wait_steps
            
            # 🔑🔑 关键：必须显式重置夹爪力矩！
            # Event 3-5 设置了 -200N 的闭合力矩，如果不重置，夹爪无法打开
            try:
                # 获取夹爪打开位置
                if hasattr(my_franka.gripper, "joint_opened_positions"):
                    gripper_open_pos = my_franka.gripper.joint_opened_positions[0]
                else:
                    gripper_open_pos = 0.04  # 默认打开位置
                
                # 使用完整的 ArticulationAction：位置 + 力矩（正值 = 打开方向）
                gripper_open_force = 50.0  # 打开力矩（正值）
                open_action = ArticulationAction(
                    joint_positions=np.array([gripper_open_pos, gripper_open_pos]),
                    joint_velocities=np.array([0.0, 0.0]),
                    joint_efforts=np.array([gripper_open_force, gripper_open_force]),  # 打开力矩
                    joint_indices=[7, 8]  # 只控制夹爪关节
                )
                articulation_controller.apply_action(open_action)
            except Exception as e:
                print(f"⚠️ 夹爪打开命令失败: {e}")
                # 备用方法
                try:
                    my_franka.gripper.open()
                except:
                    pass
            
            my_controller.wait_counter -= 1
            if my_controller.wait_counter == 0:
                # 夹爪打开完成，从 CuRobo 分离物体
                print("📤 从 CuRobo 分离物体")
                my_controller._detach_object()
                my_controller.next_event()
        
        # Event 7+: 完成
        else:
            # 🔑 区分单独运行和采集模式
            # - 单独运行（__name__ == "__main__"）：保持运行，不退出
            # - 采集模式（被 collect_curobo.py 导入）：返回 False 退出
            
            if not hasattr(my_controller, '_completion_steps'):
                my_controller._completion_steps = 0
                print("\n🎉 任务完成！\n")
            
            my_controller._completion_steps += 1
            
            # 执行 30 步让场景稳定
            if my_controller._completion_steps <= 30:
                my_world.step(render=render if render is not None else True)
                return True
            else:
                # 🔑 关键：检查是否为采集模式
                # 如果是被导入的（采集模式），返回 False 退出
                # 如果是直接运行，继续返回 True 保持运行
                if __name__ != "__main__":
                    # 采集模式：返回 False 让 collect_curobo.py 退出
                    print("📊 采集模式：任务完成，准备退出...")
                    return False
                else:
                    # 单独运行模式：保持运行
                    my_world.step(render=render if render is not None else True)
                    return True

    return True


# 只有直接运行时才执行主循环，被导入时跳过，直接运行在这里修改参数
if __name__ == "__main__":
    try:
        while step_once(
            render=True
            ):
            pass
    finally:
        simulation_app.close()

# 运行命令
# /home/di-gua/isaac-sim/python.sh scipy/pick_place_cu_ramsci.py