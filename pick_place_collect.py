# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import time
from pathlib import Path

import numpy as np
from PIL import Image
import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid


CAPTURE_CAMERA_PATH = "/Camera"
CAPTURE_RESOLUTION = (720, 720)
CAPTURE_BASE_DIR = "/home/yons/tmp/pick_place_capture"
CAPTURE_FPS = 5.0


def _rgb_frame_to_numpy(rgb_data, resolution):
    if rgb_data is None:
        return None
    data = rgb_data.get("data") if isinstance(rgb_data, dict) else rgb_data
    if data is None:
        return None
    if hasattr(data, "__module__") and "warp" in data.__module__:
        if hasattr(data, "numpy"):
            data = data.numpy()
        else:
            try:
                import warp as wp  # type: ignore

                data = wp.to_numpy(data)
            except Exception:
                return None
    if isinstance(data, bytes):
        data = np.frombuffer(data, dtype=np.uint8)
    if not isinstance(data, np.ndarray):
        try:
            data = np.asarray(data, dtype=np.uint8)
        except Exception:
            return None
    res_w, res_h = resolution
    expected_rgba = res_w * res_h * 4
    expected_rgb = res_w * res_h * 3
    if data.ndim == 1:
        if data.size == expected_rgba:
            data = data.reshape((res_h, res_w, 4))[:, :, :3]
        elif data.size == expected_rgb:
            data = data.reshape((res_h, res_w, 3))
        else:
            return None
    elif data.ndim == 3:
        if data.shape[-1] == 4:
            data = data[:, :, :3]
        elif data.shape[-1] != 3:
            return None
    else:
        return None
    if data.dtype != np.uint8:
        data = data.astype(np.uint8)
    return data


def _normalize_joint_values(values, width):
    if width <= 0:
        return []
    if values is None:
        return [0.0] * width
    array = list(values)
    if len(array) < width:
        array.extend([0.0] * (width - len(array)))
    elif len(array) > width:
        array = array[:width]
    return array


def _discover_camera_paths(keyword="camera"):
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []
    keyword_lower = keyword.lower()
    results = []
    for prim in stage.Traverse():
        path_str = prim.GetPath().pathString
        if keyword_lower not in path_str.lower():
            continue
        if prim.GetTypeName().lower() == "camera" or UsdGeom.Camera(prim).IsValid():
            results.append(path_str)
    return results

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

# 包装/创建 Franka；不再使用碗位置，统一按固定位置放置
if Franka is not None and is_prim_path_valid("/World/Franka"):
    my_franka = Franka(prim_path="/World/Franka", name="Franka")
    # 强制移动已有 Franka 到固定位置
    try:
        my_franka.set_world_pose(position=fixed_spawn_pos)
    except Exception:
        XFormPrim("/World/Franka").set_world_pose(position=fixed_spawn_pos)
    simulation_app.update()
else:
    if Franka is None:
        raise RuntimeError("未找到 Franka 包装类(omni.isaac.franka)。请在扩展中启用 omni.isaac.franka 后重试。")
    my_franka = Franka(prim_path="/World/Franka", name="Franka", position=fixed_spawn_pos)

# 注册到 scene（仅注册机器人即可，XFormPrim 直接用于读姿态，无需加入 scene）
# my_world.scene.add(salt)
# my_world.scene.add(bowl)
my_world.scene.add(my_franka)

my_world.reset()

if not is_prim_path_valid(CAPTURE_CAMERA_PATH):
    raise RuntimeError(f"未找到采集相机 {CAPTURE_CAMERA_PATH}，请确认场景中存在该 prim。")

discovered_camera_paths = _discover_camera_paths()
camera_paths = []
for path in discovered_camera_paths:
    if not is_prim_path_valid(path):
        continue
    if path not in camera_paths:
        camera_paths.append(path)
if CAPTURE_CAMERA_PATH not in camera_paths:
    camera_paths.insert(0, CAPTURE_CAMERA_PATH)

if not camera_paths:
    raise RuntimeError("场景中未找到包含 'camera' 的有效相机 prim，无法采集图像。")

session_dir = Path(CAPTURE_BASE_DIR) / time.strftime("session_%Y%m%d_%H%M%S")
camera_rgb_root = session_dir / "camera_rgb"
camera_rgb_root.mkdir(parents=True, exist_ok=True)
joint_log_path = session_dir / "franka_joint_states.txt"

camera_capture_entries = []
for cam_path in camera_paths:
    subdir_name = cam_path.strip("/").replace("/", "_") or "root_camera"
    save_dir = camera_rgb_root / subdir_name
    save_dir.mkdir(parents=True, exist_ok=True)
    render_product = rep.create.render_product(cam_path, CAPTURE_RESOLUTION)
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    try:
        annotator.attach([render_product])
    except Exception:
        annotator.attach(render_product)
    camera_capture_entries.append({
        "path": cam_path,
        "dir": save_dir,
        "annotator": annotator,
    })

rep.orchestrator.step()
rep.orchestrator.wait_until_complete()

sample_positions = my_franka.get_joint_positions()
if hasattr(my_franka, "dof_names") and my_franka.dof_names:
    dof_names = list(my_franka.dof_names)
elif sample_positions is not None:
    dof_names = [f"dof_{idx}" for idx in range(len(sample_positions))]
else:
    fallback_count = getattr(my_franka, "num_dof", None) or getattr(my_franka, "num_dofs", None)
    if fallback_count:
        dof_names = [f"dof_{idx}" for idx in range(int(fallback_count))]
    else:
        dof_names = []
dof_count = len(dof_names)

joint_log_file = open(joint_log_path, "w", encoding="utf-8")
joint_log_file.write("Franka joint log (TXT)\n")
joint_log_file.write(f"Total DOFs: {dof_count}\n")
joint_log_file.write("Order: frame_index, sim_time_sec, joints listed with pos/vel/eff\n")
joint_log_file.write("-" * 80 + "\n")

print("📸 数据采集信息:")
print(f"   - 默认相机: {CAPTURE_CAMERA_PATH}, 分辨率: {CAPTURE_RESOLUTION[0]}x{CAPTURE_RESOLUTION[1]}, 频率: {CAPTURE_FPS} Hz")
print("   - 多相机输出目录:")
for entry in camera_capture_entries:
    print(f"       · {entry['path']} -> {entry['dir']}")
print(f"   - 关节数据日志: {joint_log_path}")

# 创建抓放控制器
my_controller = PickPlaceController(
    name="pick_place_controller",
    gripper=my_franka.gripper,
    robot_articulation=my_franka
)
articulation_controller = my_franka.get_articulation_controller()

# 初始化：强制打开夹爪一次
try:
    open_action = my_franka.gripper.forward(action="open")
    articulation_controller.apply_action(open_action)
except Exception:
    # 兼容少数实现：直接用 opened_positions
    if hasattr(my_franka.gripper, "joint_opened_positions"):
        my_franka.gripper.set_joint_positions(my_franka.gripper.joint_opened_positions)

reset_needed = False

# 可调参数
placing_height_offset = 0.05
eef_lateral_offset = np.array([0.0, 0.01, 0.0])
capture_interval_sec = 1.0 / CAPTURE_FPS if CAPTURE_FPS > 0 else 0.2
capture_time_accum = 0.0
frame_index = 0
sim_time_sec = 0.0
capturing_active = False
capture_episode_index = 0

try:
    while simulation_app.is_running():
        my_world.step(render=True)
        try:
            physics_dt = float(my_world.get_physics_dt())
        except Exception:
            physics_dt = 1.0 / 60.0
        sim_time_sec += physics_dt
        capture_time_accum += physics_dt

        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                my_controller.reset()
                capturing_active = False
                capture_time_accum = 0.0
                # reset 后再次确保夹爪打开
                try:
                    open_action = my_franka.gripper.forward(action="open")
                    articulation_controller.apply_action(open_action)
                except Exception:
                    if hasattr(my_franka.gripper, "joint_opened_positions"):
                        my_franka.gripper.set_joint_positions(my_franka.gripper.joint_opened_positions)
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

            event_id = None
            if hasattr(my_controller, "get_current_event"):
                try:
                    event_id = my_controller.get_current_event()
                except Exception:
                    event_id = None

            if not capturing_active and event_id is not None and event_id >= 0 and not my_controller.is_done():
                capturing_active = True
                capture_episode_index += 1
                capture_time_accum = 0.0
                joint_log_file.write(f"Episode {capture_episode_index} START (sim_time={sim_time_sec:.6f})\n")
                joint_log_file.write("-" * 60 + "\n")
                joint_log_file.flush()

            if capturing_active and my_controller.is_done():
                joint_log_file.write(f"Episode {capture_episode_index} END (sim_time={sim_time_sec:.6f})\n")
                joint_log_file.write("-" * 60 + "\n")
                joint_log_file.flush()
                capturing_active = False
                capture_time_accum = 0.0
                continue

            if not capturing_active:
                continue

            while CAPTURE_FPS > 0 and capture_time_accum >= capture_interval_sec:
                capture_time_accum -= capture_interval_sec
                frame_index += 1
                for entry in camera_capture_entries:
                    rgb_frame = _rgb_frame_to_numpy(entry["annotator"].get_data(), CAPTURE_RESOLUTION)
                    if rgb_frame is not None:
                        image_path = entry["dir"] / f"{frame_index:06d}.png"
                        Image.fromarray(rgb_frame).save(str(image_path))
                    else:
                        print(f"[capture] 第 {frame_index:06d} 帧未获取到 {entry['path']} 的 RGB 数据。")

                vel_callable = getattr(my_franka, "get_joint_velocities", None)
                eff_callable = getattr(my_franka, "get_joint_efforts", None)
                joint_velocities = vel_callable() if callable(vel_callable) else None
                joint_efforts = eff_callable() if callable(eff_callable) else None

                joint_positions = _normalize_joint_values(current_joint_positions, dof_count)
                joint_velocities = _normalize_joint_values(joint_velocities, dof_count)
                joint_efforts = _normalize_joint_values(joint_efforts, dof_count)

                joint_log_file.write(f"Frame: {frame_index}\n")
                joint_log_file.write(f"Sim time (s): {sim_time_sec:.6f}\n")
                for idx, name in enumerate(dof_names):
                    pos_val = joint_positions[idx] if idx < len(joint_positions) else 0.0
                    vel_val = joint_velocities[idx] if idx < len(joint_velocities) else 0.0
                    eff_val = joint_efforts[idx] if idx < len(joint_efforts) else 0.0
                    joint_log_file.write(
                        f"  {name:>12}: pos={pos_val: .6f}  vel={vel_val: .6f}  eff={eff_val: .6f}\n"
                    )
                if not dof_names:
                    joint_log_file.write("  (no DOF data exposed)\n")
                joint_log_file.write("=" * 60 + "\n")
                joint_log_file.flush()

            # if my_controller.is_done():
            #     print("完成一次抓取并放置循环")
finally:
    if joint_log_file:
        joint_log_file.close()
    simulation_app.close()

#./python.sh /home/yons/data/isaacsim/pick_place_collect.py
