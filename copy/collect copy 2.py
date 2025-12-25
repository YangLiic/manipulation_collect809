"""通用采集脚本。

职责概览：

1. 仅执行一次任务脚本的初始化，让其中的场景、控制器和 `SimulationApp` 得到复用。
2. 通过约定的 `step_once`（或其它候选名称）回调，由任务脚本驱动机器人逻辑。
3. 本文件专注于数据采集：自动发现场景中的相机，定频抓取 RGB 图像，并记录 Franka 关节状态。
4. 采集生命周期可由任务脚本提供的钩子函数控制，也可退化为基于控制器状态机的默认策略。

借助这种分层，`collect.py` 能够为不同任务脚本提供“即插即用”的采集能力，而无需复制控制代码。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image


CAPTURE_RESOLUTION = (1280, 960)
CAMERA_KEYWORD = "Camera"
DEFAULT_FPS = 5.0
DEPTH_MAX_METERS = 10.0
DEPTH_SCALE_MM = 1000.0


def _resolve_script_path(script: str) -> Path:
    """Resolve user input to an actual Python file path.

    Accepts either `foo.py` or `foo` and searches relative to当前工作目录。
    Raises `FileNotFoundError` 以便调用方给出明确报错。
    """
    path = Path(script)
    if path.is_file():
        return path
    candidate = Path(f"{script}.py")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"找不到脚本 {script} 或 {script}.py")


def _load_module(script: str) -> ModuleType:
    """Dynamically import the task module without executing its main loop."""
    module_path = _resolve_script_path(script)
    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _rgb_frame_to_numpy(rgb_data, resolution):
    """Normalize replicator RGB输出为 HxWx3 uint8 numpy 数组。"""
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


def _depth_frame_to_uint16(depth_data, resolution):
    """Convert replicator distance_to_camera数据为 uint16 (mm) 图像。"""
    if depth_data is None:
        return None
    data = depth_data.get("data") if isinstance(depth_data, dict) else depth_data
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
        data = np.frombuffer(data, dtype=np.float32)
    if not isinstance(data, np.ndarray):
        try:
            data = np.asarray(data, dtype=np.float32)
        except Exception:
            return None

    res_w, res_h = resolution
    expected = res_w * res_h
    if data.ndim == 1:
        if data.size == expected:
            data = data.reshape((res_h, res_w))
        else:
            return None
    elif data.ndim == 3:
        data = data.reshape((res_h, res_w)) if data.shape[-1] == 1 else data[:, :, 0]
    elif data.ndim != 2:
        return None

    if data.dtype != np.float32:
        data = data.astype(np.float32)

    clipped = np.clip(data, 0.0, DEPTH_MAX_METERS)
    depth_mm = (clipped * DEPTH_SCALE_MM).astype(np.uint16)
    return depth_mm


def _normalize_joint_values(values, width):
    """Pad/trim 序列到固定长度，方便日志格式化。"""
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


def _discover_camera_paths(keyword: str = CAMERA_KEYWORD):
    """在当前 stage 中搜索包含 keyword 的相机 prim。"""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []
    keyword_lower = keyword.lower()
    results = []
    for prim in stage.Traverse():
        path_str = prim.GetPath().pathString
        if keyword_lower not in path_str.lower():
            continue
        # 修复: UsdGeom.Camera(prim) 没有 IsValid() 方法，改用 prim.IsA(UsdGeom.Camera)
        if prim.GetTypeName() == "Camera" or prim.IsA(UsdGeom.Camera):
            results.append(path_str)
    return results


def _resolve_step_function(module: ModuleType) -> Callable[..., object]:
    """从任务模块中选择一个可调用的 step 函数。"""
    candidate_names = ("step_once", "task_step", "collect_step", "step")
    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(
        "模块需要提供可调用的 step_once/task_step/collect_step/step (返回 False 可提前结束)"
    )


def _call_step_function(step_fn: Callable[..., object]) -> object:
    """调用任务提供的 step 函数，兼容是否需要 render 参数。"""
    try:
        return step_fn(render=True)
    except TypeError:
        return step_fn()


def _safe_get_physics_dt(world) -> float:
    """Safely query physics dt, falling back to 60 Hz。"""
    try:
        return float(world.get_physics_dt())
    except Exception:
        return 1.0 / 60.0


def _query_controller_state(controller) -> Tuple[Optional[int], bool]:
    """读取控制器状态机事件编号与完成标记。"""
    event_id = None
    controller_done = False
    if controller is None:
        return event_id, controller_done
    if hasattr(controller, "get_current_event"):
        try:
            event_id = controller.get_current_event()
        except Exception:
            event_id = None
    if hasattr(controller, "is_done"):
        try:
            controller_done = bool(controller.is_done())
        except Exception:
            controller_done = False
    return event_id, controller_done


def _ensure_timeline_playing():
    """确保 Omni timeline 处于播放状态，避免手动点 Play。"""
    try:
        import omni.timeline  # type: ignore

        timeline = omni.timeline.get_timeline_interface()
    except Exception:
        timeline = None
    if timeline is not None and not timeline.is_playing():
        timeline.play()


def _auto_start_world(world):
    """在 timeline 已播放的前提下，确保 World 也进入播放/复位状态。"""
    if world is None:
        return
    try:
        is_playing = world.is_playing() if hasattr(world, "is_playing") else True
    except Exception:
        is_playing = True

    if not is_playing and hasattr(world, "play"):
        try:
            world.play()
        except Exception as exc:
            print(f"[collect] 无法自动播放 World: {exc}")
            return

    if hasattr(world, "reset"):
        try:
            world.reset()
        except Exception as exc:
            print(f"[collect] 无法在自动播放后复位 World: {exc}")


class CaptureSession:
    """负责处理 replicator 相机与 Franka 关节日志的一次采集会话。"""

    def __init__(self, world, franka, output_root: str, fps: float):
        """初始化输出目录、相机 render product 以及关节日志文件。"""
        import omni.replicator.core as rep

        self.world = world
        self.franka = franka
        self.rep = rep
        self.fps = fps if fps > 0 else DEFAULT_FPS
        self.capture_interval = 1.0 / self.fps if self.fps > 0 else 0.2

        root_dir = Path(output_root).expanduser().absolute()
        timestamp = time.strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = root_dir / timestamp
        self.camera_rgb_root = self.session_dir / "camera_rgb"
        self.camera_depth_root = self.session_dir / "camera_depth"
        self.camera_rgb_root.mkdir(parents=True, exist_ok=True)
        self.camera_depth_root.mkdir(parents=True, exist_ok=True)

        self.camera_capture_entries = []
        for cam_path in _discover_camera_paths():
            subdir_name = cam_path.strip("/").replace("/", "_") or "root_camera"
            rgb_dir = self.camera_rgb_root / subdir_name
            depth_dir = self.camera_depth_root / subdir_name
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            render_product = rep.create.render_product(cam_path, CAPTURE_RESOLUTION)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            try:
                annotator.attach([render_product])
            except Exception:
                annotator.attach(render_product)
            depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            try:
                depth_annotator.attach([render_product])
            except Exception:
                depth_annotator.attach(render_product)
            self.camera_capture_entries.append({
                "path": cam_path,
                "rgb_dir": rgb_dir,
                "depth_dir": depth_dir,
                "rgb_annotator": annotator,
                "depth_annotator": depth_annotator,
            })

        rep.orchestrator.step()
        rep.orchestrator.wait_until_complete()

        self.joint_log_file = None
        self.dof_names = []
        self.dof_count = 0
        if self.franka is not None:
            sample_positions = getattr(self.franka, "get_joint_positions", lambda: None)()
            if hasattr(self.franka, "dof_names") and self.franka.dof_names:
                self.dof_names = list(self.franka.dof_names)
            elif sample_positions is not None:
                self.dof_names = [f"dof_{idx}" for idx in range(len(sample_positions))]
            else:
                fallback_count = getattr(self.franka, "num_dof", None) or getattr(self.franka, "num_dofs", None)
                if fallback_count:
                    self.dof_names = [f"dof_{idx}" for idx in range(int(fallback_count))]
            self.dof_count = len(self.dof_names)

            joint_log_path = self.session_dir / "franka_joint_states.txt"
            self.joint_log_file = open(joint_log_path, "w", encoding="utf-8")
            self.joint_log_file.write("Franka joint log (TXT)\n")
            self.joint_log_file.write(f"Total DOFs: {self.dof_count}\n")
            self.joint_log_file.write("Order: frame_index, sim_time_sec, joints listed with pos/vel/eff\n")
            self.joint_log_file.write("-" * 80 + "\n")
            self.joint_log_file.flush()

        self.sim_time_sec = 0.0
        self.capture_time_accum = 0.0
        self.frame_index = 0

        print("📸 数据采集信息:")
        if self.camera_capture_entries:
            for entry in self.camera_capture_entries:
                print(f"   · {entry['path']} -> RGB: {entry['rgb_dir']} | Depth: {entry['depth_dir']}")
        else:
            print("   · 未发现包含 camera 关键字的相机，将无法输出 RGB/Depth 桢。")
        if self.joint_log_file is not None:
            print(f"   · 关节日志写入: {self.session_dir / 'franka_joint_states.txt'}")

    def advance_time(self, physics_dt: float):
        """推进内部时间累计值（仿真时间步长来自任务世界）。"""
        self.sim_time_sec += physics_dt
        self.capture_time_accum += physics_dt

    def reset_capture_timer(self):
        """清零捕获计时器，用于 episode 切换或暂停阶段。"""
        self.capture_time_accum = 0.0

    def start_episode(self, episode_index: int):
        """记录 episode 起点并重置计时器。"""
        if self.joint_log_file is not None:
            self.joint_log_file.write(
                f"Episode {episode_index} START (sim_time={self.sim_time_sec:.6f})\n"
            )
            self.joint_log_file.write("-" * 60 + "\n")
            self.joint_log_file.flush()
        self.reset_capture_timer()

    def end_episode(self, episode_index: int):
        """记录 episode 终点并重置计时器。"""
        if self.joint_log_file is not None:
            self.joint_log_file.write(
                f"Episode {episode_index} END (sim_time={self.sim_time_sec:.6f})\n"
            )
            self.joint_log_file.write("-" * 60 + "\n")
            self.joint_log_file.flush()
        self.reset_capture_timer()

    def capture_frames_if_needed(self):
        """按设定频率写出 RGB 帧，并附带一次关节日志。"""
        if not self.camera_capture_entries:
            self.capture_time_accum = 0.0
            return
        while self.capture_time_accum >= self.capture_interval:
            self.capture_time_accum -= self.capture_interval
            self.frame_index += 1
            for entry in self.camera_capture_entries:
                rgb_saved = False
                rgb_frame = _rgb_frame_to_numpy(entry["rgb_annotator"].get_data(), CAPTURE_RESOLUTION)
                if rgb_frame is not None:
                    rgb_path = entry["rgb_dir"] / f"{self.frame_index:06d}.png"
                    Image.fromarray(rgb_frame).save(str(rgb_path))
                    rgb_saved = True
                else:
                    print(f"[capture] 帧 {self.frame_index:06d} 未获取到 {entry['path']} 的 RGB 数据。")

                depth_saved = False
                depth_frame = _depth_frame_to_uint16(entry["depth_annotator"].get_data(), CAPTURE_RESOLUTION)
                if depth_frame is not None:
                    depth_path = entry["depth_dir"] / f"{self.frame_index:06d}.png"
                    Image.fromarray(depth_frame, mode="I;16").save(str(depth_path))
                    depth_saved = True
                else:
                    print(f"[capture] 帧 {self.frame_index:06d} 未获取到 {entry['path']} 的 Depth 数据。")

                print(
                    f"[capture] 帧 {self.frame_index:06d} {entry['path']} -> RGB:{'✔' if rgb_saved else '✘'} Depth:{'✔' if depth_saved else '✘'}"
                )
            self._log_franka_state()

    def _log_franka_state(self):
        """将当前位置/速度/力矩写入 joint log。"""
        if self.franka is None or self.joint_log_file is None or self.dof_count == 0:
            return
        vel_callable = getattr(self.franka, "get_joint_velocities", None)
        eff_callable = getattr(self.franka, "get_joint_efforts", None)
        joint_velocities = vel_callable() if callable(vel_callable) else None
        joint_efforts = eff_callable() if callable(eff_callable) else None

        joint_positions = _normalize_joint_values(
            getattr(self.franka, "get_joint_positions", lambda: [])(), self.dof_count
        )
        joint_velocities = _normalize_joint_values(joint_velocities, self.dof_count)
        joint_efforts = _normalize_joint_values(joint_efforts, self.dof_count)

        self.joint_log_file.write(f"Frame: {self.frame_index}\n")
        self.joint_log_file.write(f"Sim time (s): {self.sim_time_sec:.6f}\n")
        for idx, name in enumerate(self.dof_names):
            pos_val = joint_positions[idx] if idx < len(joint_positions) else 0.0
            vel_val = joint_velocities[idx] if idx < len(joint_velocities) else 0.0
            eff_val = joint_efforts[idx] if idx < len(joint_efforts) else 0.0
            self.joint_log_file.write(
                f"  {name:>12}: pos={pos_val: .6f}  vel={vel_val: .6f}  eff={eff_val: .6f}\n"
            )
        if not self.dof_names:
            self.joint_log_file.write("  (no DOF data exposed)\n")
        self.joint_log_file.write("=" * 60 + "\n")
        self.joint_log_file.flush()

    def close(self):
        """关闭关节日志文件。"""
        if self.joint_log_file is not None:
            try:
                self.joint_log_file.close()
            except Exception:
                pass


def collect_from_module(
    script: str,
    out_dir: str,
    fps: float = DEFAULT_FPS,
    headless: bool = True,
    exit_on_complete: bool = True,
):
    """导入任务脚本并运行采集循环。

    - 任务脚本负责控制逻辑，只需暴露 `step_once`（或候选名称）和必要对象。
    - 本函数负责：定位脚本、构建采集会话、根据控制器状态或钩子函数决定何时采集。
    - `exit_on_complete=True` 时，一旦任务结束（stop hook 或控制器完成）将自动退出仿真。
    """
    os.environ["ISAACSIM_HEADLESS"] = "1" if headless else "0"

    module = _load_module(script)

    simulation_app = getattr(module, "simulation_app", None)
    my_world = getattr(module, "my_world", None)
    if simulation_app is None or my_world is None:
        raise RuntimeError("模块需要暴露 simulation_app 和 my_world")

    step_fn = _resolve_step_function(module)
    my_franka = getattr(module, "my_franka", None)
    collector = CaptureSession(my_world, my_franka, out_dir, fps)

    _ensure_timeline_playing()
    _auto_start_world(my_world)

    controller = getattr(module, "my_controller", None)
    start_hook = getattr(module, "collect_should_start_capture", None)
    stop_hook = getattr(module, "collect_should_stop_capture", None)

    capturing_active = False
    episode_index = 0
    terminate_after_cycle = False

    try:
        while simulation_app.is_running():
            # 让任务脚本推进一次控制（含 my_world.step/render 等）
            step_result = _call_step_function(step_fn)
            if step_result is False:
                break

            physics_dt = _safe_get_physics_dt(my_world)
            collector.advance_time(physics_dt)

            should_start = False
            should_stop = False

            if callable(start_hook) or callable(stop_hook):
                # 用户自定义采集启动/停止判定，适合复杂任务
                if callable(start_hook) and not capturing_active:
                    should_start = bool(start_hook())
                if callable(stop_hook) and capturing_active:
                    should_stop = bool(stop_hook())
            else:
                # 默认：基于控制器的事件 ID 与 is_done 状态
                event_id, controller_done = _query_controller_state(controller)
                if not capturing_active and event_id is not None and event_id >= 0 and not controller_done:
                    should_start = True
                if capturing_active and controller_done:
                    should_stop = True

            if should_start:
                capturing_active = True
                episode_index += 1
                collector.start_episode(episode_index)

            if should_stop:
                collector.end_episode(episode_index)
                capturing_active = False
                collector.reset_capture_timer()
                if exit_on_complete:
                    terminate_after_cycle = True
                    break
                continue

            if not capturing_active:
                collector.reset_capture_timer()
                continue

            collector.capture_frames_if_needed()

            if terminate_after_cycle:
                break

        if terminate_after_cycle:
            print("[collect] 任务完成，按 exit_on_complete 设置自动退出。")
    finally:
        collector.close()
        simulation_app.close()

    print("采集完成。")


def parse_args():
    """CLI 参数解析，支持脚本、输出目录、采样频率及 headless 控制。"""
    p = argparse.ArgumentParser(description="导入任务脚本并执行数据采集")
    p.add_argument("--script", "-s", default="pick_place", help="任务脚本(不含 .py)")
    p.add_argument("--out", "-o", default="./collect_output", help="输出根目录")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="采样频率 (Hz)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--headless", dest="headless", action="store_true", help="以无头模式运行（默认）")
    group.add_argument("--gui", dest="headless", action="store_false", help="强制显示 GUI")
    p.set_defaults(headless=True)
    exit_group = p.add_mutually_exclusive_group()
    exit_group.add_argument(
        "--exit-on-complete",
        dest="exit_on_complete",
        action="store_true",
        help="任务完成后自动退出仿真（默认）",
    )
    exit_group.add_argument(
        "--keep-alive",
        dest="exit_on_complete",
        action="store_false",
        help="任务完成后保持仿真运行，等待下一次采集",
    )
    p.set_defaults(exit_on_complete=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_from_module(
        args.script,
        args.out,
        args.fps,
        headless=args.headless,
        exit_on_complete=args.exit_on_complete,
    )

    #./python.sh /home/yons/data/isaacsim/collect.py --script pick_place_localFranka --out /home/yons/tmp/collect_output --fps 5
