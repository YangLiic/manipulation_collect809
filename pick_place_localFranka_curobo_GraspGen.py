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

# Third Party
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

# 优先使用 omni.isaac.franka 的 Franka 包装类
try:
    from omni.isaac.franka import Franka
except Exception:
    Franka = None

# 加载场景 USD
open_stage("/home/yons/data/Collected_World1/World0.usd")
simulation_app.update()

# 创建 World
my_world = World(stage_units_in_meters=1.0)

# 包装场景中现有的对象
salt = XFormPrim("/World/Vegetable_7")
bowl = XFormPrim("/World/Bowl_0")

# 固定放置位置
fixed_spawn_pos = np.array([0.0, 0.6, -0.25], dtype=float)

FRANKA_LOCAL_USD = "/home/yons/data/Franka_usd/Franka.usd"
FRANKA_REFERENCE_PATH = "/World/Franka"
FRANKA_NESTED_PATH = "/World/Franka/franka"


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
        my_franka.set_world_pose(position=fixed_spawn_pos)
    except Exception:
        XFormPrim(franka_prim_path).set_world_pose(position=fixed_spawn_pos)
    simulation_app.update()
else:
    my_franka = Franka(prim_path=franka_prim_path, name="Franka", position=fixed_spawn_pos)

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
        self.approach_height = 0.10  # 接近高度
        self.lift_height = 0.10      # 抬升高度
        
        # 获取机器人基座的世界位置
        robot_base_prim = XFormPrim(franka_prim_path)
        positions, orientations = robot_base_prim.get_world_poses()
        self.robot_base_position = positions[0]  # 取第一个元素
        self.robot_base_orientation = orientations[0]
        print(f"🤖 机器人基座世界位置: {self.robot_base_position}")
        print(f"🤖 机器人基座世界姿态: {self.robot_base_orientation}")
        
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
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=0.01,
            collision_cache={"obb": 50, "mesh": 30},
            collision_activation_distance=0.10,  # 增加容忍度
        )
        
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(parallel_finetune=True)
        self.motion_gen.update_world(self._world_cfg.get_collision_check_world())
        
        # 规划配置
        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            max_attempts=30,  # 增加尝试次数
            enable_graph_attempt=15,  # 增加图搜索尝试
            enable_finetune_trajopt=True,
            parallel_finetune=True,
            time_dilation_factor=1.0,
            timeout=10.0,  # 增加超时时间
        )
        
        # 状态
        self.current_event = 0
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.idx_list = None
        self.is_attached = False
        self.wait_counter = 0  # 等待计数器
        self.stabilize_counter = 0 # 放置前稳定计数器
        self.wait_steps = 50   # 增加等待步数，确保夹爪完全闭合
        self.saved_pick_position = None  # 保存抓取时的位置，避免提升时跟踪移动物体
        self.plan_fail_counter = 0  # 规划失败计数器
        self.is_planning = False  # 标记是否正在进行运动规划（用于采集器暫停采集）
        
        # 🔑 关键：夹爪闭合位置，用于在移动时保持夹持状态
        self.gripper_closed_position = 0.0  # 夹爪闭合时的位置（每个手指）
        
        # 阈值
        self.position_threshold = 0.08  # 8cm（放宽阈值，因为轨迹执行完即可认为到达）
        
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
            #"/World/defaultGroundPlane", 
            #"/curobo",
            "/World/Vegetable_7",  # 忽略要抓取的物体
            "/World/Bowl_0",       # 忽略碗
            # 暂时忽略其他物体以简化规划
            #"/World/Bottle",
            "/World/Scissors",
            "/World/Vegetable_8",
            "/World/Vegetable_9",
            "/World/Garlic",
            "/World/Peeler",
            #"/World/SaltShaker",
            "/World/CuttingBoard",
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
        """根据当前事件获取目标位姿（6状态版本）"""
        # 末端朝下的四元数 [w, x, y, z] - 180度绕X轴旋转
        ee_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 朝下 (w, x, y, z)
        
        if self.current_event == 0:  # 靠近抓取物体 (Approach)
            pos = picking_position + np.array([0, 0, self.approach_height]) + offset
            return (pos, ee_quat)
        elif self.current_event == 1:  # 下降抓取 (Descend)
            pos = picking_position + offset
            return (pos, ee_quat)
        elif self.current_event == 2:  # 抓取 (Grasp - 等待夹爪闭合)
            return None  
        elif self.current_event == 3:  # 附着物体并运输 (Transport)
            if not self.is_attached:
                self._attach_object()
            # 直接规划到放置位置
            pos = placing_position + offset
            return (pos, ee_quat)
        elif self.current_event == 4:  # 放置 (Place - 等待稳定 + 打开夹爪)
            return None  
        elif self.current_event == 5:  # 分离物体并后退 (Retreat)
            if self.is_attached:
                self._detach_object()
            pos = placing_position + np.array([0, 0, self.approach_height]) + offset
            return (pos, ee_quat)
        else:  # 完成
            return None
    
    def _plan_to_pose(self, target_pose):
        """规划到目标位姿"""
        pos_world, quat_world = target_pose
        
        # 调试输出：世界坐标
        print(f"📍 Event {self.current_event}:")
        print(f"   世界坐标目标位置: {pos_world}")
        print(f"   机器人基座位置: {self.robot_base_position}")
        
        # ✅ CuRobo 使用相对于机器人基座的坐标系
        # 必须将世界坐标转换为相对坐标
        pos_relative = pos_world - self.robot_base_position
        print(f"   相对基座位置: {pos_relative}")
        
        # 使用相对坐标
        pos = pos_relative
        quat = quat_world
        
        print(f"   传入 CuRobo 的位置: {pos}")
        print(f"   传入 CuRobo 的姿态: {quat}")
        
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
            return False
    
    def _execute_trajectory(self):
        """执行当前轨迹"""
        if self.cmd_plan is None:
            return None  # 等待主循环切换事件
        
        # 每3步发送一次指令
        if self._step_idx % 3 == 0:
            cmd_state = self.cmd_plan[self.cmd_idx]
            self.cmd_idx += 1
            
            # 🔑 关键修复：在 Event 3 (运输) 期间，在 ArticulationAction 中包含夹爪关节
            # 这样可以防止手臂运动时覆盖夹爪控制
            if self.current_event == 3:  # 运输阶段
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
                # Event 0, 1, 5: 只控制手臂关节
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    cmd_state.velocity.cpu().numpy() * 0.0,
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
    
    def _attach_object(self):
        """附着物体"""
        # 🔑 优化：直接指定要附加的物体路径，而不是通过排除法
        target_object_path = "/World/Vegetable_7"
        
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
            self._update_world()  # 这会忽略 Vegetable_7
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
        if self.is_attached:
            self._detach_object()
    
    def get_current_event(self):
        """获取当前事件（兼容接口）"""
        return self.current_event
    
    def next_event(self):
        """切换到下一个事件（6状态版本）"""
        if self.current_event < 6:
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

# 可调参数
placing_height_offset = 0.10 #放置时高度
eef_lateral_offset = np.array([0.0, 0.0, 0.02])  # 夹取时偏移


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

        # 获取盐瓶与碗的世界位姿
        salt_positions, _ = salt.get_world_poses()
        bowl_positions, _ = bowl.get_world_poses()
        salt_pos = salt_positions[0]
        bowl_pos = bowl_positions[0]

        # 修正：使用物体顶部位置，假设物体高度约 0.05m
        picking_position = salt_pos + np.array([0.0, 0.00, 0.09])  # 盐瓶顶部
        placing_position = bowl_pos + np.array([0.0, 0.0, 0.2])  # 碗顶部
        
        # 首次执行时打印位置信息
        if my_controller.get_current_event() == 0 and my_controller.cmd_plan is None:
            print(f"🎯 盐瓶位置: {salt_pos}")
            print(f"🎯 碗位置: {bowl_pos}")
            print(f"🎯 抓取位置: {picking_position}")
            print(f"🎯 放置位置: {placing_position}")

        current_joint_positions = my_franka.get_joint_positions()
        current_event = my_controller.get_current_event()

        # === 状态机逻辑（6状态版本）===
        
        # Event 0, 1, 3, 5: 规划并执行到达目标
        if current_event in [0, 1, 3, 5]:
            # ✅ Event 3 的夹爪控制已经整合到 _execute_trajectory() 的 ArticulationAction 中
            
            actions = my_controller.forward(
                picking_position=picking_position,
                placing_position=placing_position,
                current_joint_positions=current_joint_positions,
                end_effector_offset=eef_lateral_offset
            )
            if actions is not None:
                articulation_controller.apply_action(actions)
            
            # Event 3 (运输) 特殊处理：如果规划失败太多次，跳过此阶段
            if current_event == 3 and my_controller.plan_fail_counter >= 10:
                print(f"⚠️ Event 3 规划失败 {my_controller.plan_fail_counter} 次，跳过此阶段")
                my_controller.plan_fail_counter = 0
                my_controller.next_event()  # 跳到 Event 4
            
            # 检查是否到达目标
            target_pos = None
            if current_event == 0: # Approach
                target_pos = picking_position + np.array([0, 0, my_controller.approach_height]) + eef_lateral_offset
            elif current_event == 1: # Descend
                target_pos = picking_position + eef_lateral_offset
            elif current_event == 3: # Transport
                target_pos = placing_position + eef_lateral_offset
            elif current_event == 5: # Retreat
                target_pos = placing_position + np.array([0, 0, my_controller.approach_height]) + eef_lateral_offset
            
            if target_pos is not None:
                # 调试
                if my_controller.cmd_plan is None:
                    try:
                        ee_pos = my_controller.robot.end_effector.get_world_pose()[0]
                        print(f"🔍 Event {current_event} 检查到达:")
                        print(f"   目标位置(世界): {target_pos}")
                        print(f"   末端位置(世界): {ee_pos}")
                    except Exception as e:
                        print(f"   ⚠️ 无法获取末端位置: {e}")
                
                if my_controller.reached_target(target_pos):
                    print(f"✅ Event {current_event} 到达目标")
                    my_controller.next_event()
        
        # Event 2: 抓取（使用力控制闭合夹爪）
        elif current_event == 2:
            # ✅ 使用力控制命令 gripper.forward(action="close")
            if my_controller.wait_counter == 0:
                print("🤏 开始闭合夹爪（力控制模式）...")
                my_controller.wait_counter = 100  # 增加等待时间
            
            # 持续发送闭合命令（力控制）
            try:
                close_action = my_franka.gripper.forward(action="close")
                articulation_controller.apply_action(close_action)
            except Exception as e:
                print(f"⚠️ 夹爪闭合命令失败: {e}")
            
            my_controller.wait_counter -= 1
            
            if my_controller.wait_counter % 15 == 0:
                print(f"   🤏 夹爪闭合中... 剩余 {my_controller.wait_counter} 步")
            
            if my_controller.wait_counter == 0:
                print("📦 夹爪闭合完成，附加物体到 CuRobo")
                
                # 读取并保存夹爪的实际闭合位置
                try:
                    gripper_positions = my_franka.gripper.get_joint_positions()
                    my_controller.gripper_closed_position = gripper_positions[0]
                    print(f"🔒 保存夹爪闭合位置: {my_controller.gripper_closed_position:.4f}")
                except Exception as e:
                    print(f"⚠️ 无法读取夹爪位置，使用默认值 0.0: {e}")
                    my_controller.gripper_closed_position = 0.0
                
                # 保存当前抓取位置
                my_controller.saved_pick_position = picking_position.copy()
                print(f"📍 保存抓取位置: {my_controller.saved_pick_position}")
                my_controller.next_event()
        
        # Event 4: 放置（稳定 -> 打开夹爪）
        elif current_event == 4:
            # 1. 稳定阶段
            if my_controller.stabilize_counter == 0 and my_controller.wait_counter == 0:
                print("⏳ 到达放置位置，开始稳定 500ms...")
                my_controller.stabilize_counter = 45 # 30 steps * (1/60s) approx 0.5s (assuming 60hz)
            
            if my_controller.stabilize_counter > 0:
                # 保持夹爪闭合 + 保持位置 (隐式，通过不发送新运动指令机器人会维持位置)
                # 但必须发送夹爪闭合力矩，否则会掉
                 try:
                    # 获取当前关节位置(保持手臂不动)
                    # 此处简单处理：只维持夹爪闭合力
                    gripper_force = 200.0  # 闭合力
                    hold_action = ArticulationAction(
                        joint_efforts=np.array([-gripper_force, -gripper_force]),
                        joint_indices=[7, 8]
                    )
                    articulation_controller.apply_action(hold_action)
                 except Exception as e:
                     print(f"⚠️ 稳定阶段夹爪保持失败: {e}")

                 my_controller.stabilize_counter -= 1
                 if my_controller.stabilize_counter == 0:
                     print("⏳ 稳定完成，开始打开夹爪...")
                     my_controller.wait_counter = my_controller.wait_steps # 初始化打开等待
                 return True # 本次step结束

            # 2. 打开夹爪阶段 (稳定完成后)
            if my_controller.wait_counter == 0:
                # (Should not reach here due to logic above, but safety)
                my_controller.wait_counter = my_controller.wait_steps

            # 🔑🔑 关键：必须显式重置夹爪力矩！
            try:
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
            
            my_controller.wait_counter -= 1
            if my_controller.wait_counter == 0:
                # 夹爪打开完成，从 CuRobo 分离物体
                print("📤 从 CuRobo 分离物体")
                my_controller._detach_object()
                my_controller.next_event()
        
        # Event 6+: 完成
        else:
            print("\n🎉 任务完成！\n")
            for _ in range(30):
                my_world.step(render=True)

    return True


# 只有直接运行时才执行主循环，被导入时跳过
if __name__ == "__main__":
    try:
        while step_once(render=True):
            pass
    finally:
        simulation_app.close()

# 运行命令:
# /home/di-gua/isaac-sim/python.sh /home/di-gua/licheng/manipulation/manipulation_collect/pick_place_localFranka_curobo_5states.py
# omni_python /home/yons/data/manipulation_collect/pick_place_localFranka_curobo_5states.py