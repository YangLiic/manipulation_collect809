#用unitreerobotics收集数据集的主脚本（版本1）并不是多进程，但是支持异步写盘以提高吞吐量。
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from types import ModuleType
from typing import Callable, Dict, Optional, Tuple

import numpy as np


CAPTURE_RESOLUTION = (1280, 960)
CAMERA_KEYWORD = "Camera"
DEFAULT_FPS = 30.0
DEPTH_MAX_METERS = 10.0
DEPTH_SCALE_MM = 1000.0


def _resolve_script_path(script: str) -> Path:
    # 1. 尝试绝对路径或相对于当前工作目录的路径
    path = Path(script)
    if path.is_file():
        return path
    candidate = Path(f"{script}.py")
    if candidate.is_file():
        return candidate
        
    # 2. 尝试相对于 collect_curobo.py 脚本所在目录的路径
    collector_dir = Path(__file__).parent
    path = collector_dir / script
    if path.is_file():
        return path
    candidate = collector_dir / f"{script}.py"
    if candidate.is_file():
        return candidate
        
    raise FileNotFoundError(f"找不到脚本 {script} 或 {script}.py (搜索范围: CWD 和 {collector_dir})")


def _load_module(script: str) -> ModuleType:
    module_path = _resolve_script_path(script)
    module_name = module_path.stem
    
    # 🔑 自动添加脚本所在目录到 sys.path，解决子脚本内部的相对导入问题
    script_dir = str(module_path.parent.absolute())
    if script_dir not in sys.path:
        print(f"➕ 添加脚本目录到 sys.path: {script_dir}")
        sys.path.insert(0, script_dir)
        
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_step_function(module: ModuleType) -> Callable[..., object]:
    candidate_names = ("step_once", "task_step", "collect_step", "step")
    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise RuntimeError("模块需要提供 step_once/task_step/collect_step/step")


def _call_step_function(step_fn: Callable[..., object]) -> object:
    try:
        return step_fn(render=True)
    except TypeError:
        return step_fn()


def _safe_get_physics_dt(world) -> float:
    try:
        return float(world.get_physics_dt())
    except Exception:
        return 1.0 / 60.0


def _query_controller_state(controller) -> Tuple[Optional[int], bool]:
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
    try:
        import omni.timeline  # type: ignore

        timeline = omni.timeline.get_timeline_interface()
    except Exception:
        timeline = None
    if timeline is not None and not timeline.is_playing():
        timeline.play()


def _auto_start_world(world):
    if world is None:
        return
    try:
        is_playing = world.is_playing() if hasattr(world, "is_playing") else True
    except Exception:
        is_playing = True

    if not is_playing and hasattr(world, "play"):
        try:
            world.play()
        except Exception:
            return

    if hasattr(world, "reset"):
        try:
            world.reset()
        except Exception:
            pass


def _discover_camera_paths(keyword: str = CAMERA_KEYWORD):
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
        if prim.GetTypeName() == "Camera" or prim.IsA(UsdGeom.Camera):
            results.append(path_str)
    return results


def _sanitize_camera_key(camera_path: str) -> str:
    key = camera_path.strip("/").replace("/", "_")
    return key or "camera"


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


def _depth_frame_to_uint16(depth_data, resolution):
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


class AsyncEpisodeDatasetWriter:
    """高吞吐异步写盘：主线程只 enqueue；写盘线程/线程池落盘。"""

    @staticmethod
    def _sanitize_path_component(name: str) -> str:
        name = str(name or "")
        name = name.strip().strip("/")
        # Keep it filesystem-friendly.
        name = re.sub(r"[^0-9A-Za-z_\-]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name

    def _prim_path_to_rel_dir(self, prim_path: str) -> Path:
        """Convert a USD prim path (e.g. '/World/Camera_01') to a relative directory Path.

        Modes:
        - hierarchy: keep USD hierarchy as nested folders.
        - flat: join all path components with underscores.
        - short: drop common prefixes/noise + de-dup, then join with underscores.
        """
        prim_path = (prim_path or "").strip()
        parts = [p for p in prim_path.split("/") if p]
        if not parts:
            return Path("camera")

        mode = getattr(self, "camera_dir_mode", "short")
        drop_tokens = set(getattr(self, "camera_dir_drop_tokens", []) or [])

        if mode == "hierarchy":
            safe_parts = [self._sanitize_path_component(p) for p in parts]
            safe_parts = [p for p in safe_parts if p]
            return Path(*safe_parts) if safe_parts else Path("camera")

        if mode == "flat":
            joined = "_".join([self._sanitize_path_component(p) for p in parts if p])
            joined = self._sanitize_path_component(joined)
            return Path(joined or "camera")

        # short
        filtered = [p for p in parts if p not in drop_tokens]
        safe = [self._sanitize_path_component(p) for p in filtered]
        safe = [p for p in safe if p]
        # de-dup consecutive duplicates (common in camera rigs)
        dedup = []
        for p in safe:
            if not dedup or dedup[-1] != p:
                dedup.append(p)
        joined = "_".join(dedup)
        joined = self._sanitize_path_component(joined)
        return Path(joined or "camera")

    def __init__(
        self,
        root_dir: Path,
        fps: float,
        resolution: Tuple[int, int],
        rgb_format: str = "jpg",
        jpg_quality: int = 95,
        depth_format: str = "npy",
        writer_workers: int = 8,
        queue_size: int = 256,
        drop_when_full: bool = True,
        timestamp_log: str = "both",
        camera_dir_mode: str = "short",
        camera_dir_drop_tokens: Optional[str] = None,
    ):
        self.root_dir = Path(root_dir)
        self.fps = float(fps)
        self.resolution = tuple(resolution)
        self.rgb_format = rgb_format
        self.jpg_quality = int(jpg_quality)
        self.depth_format = depth_format
        self.drop_when_full = bool(drop_when_full)
        self.timestamp_log = str(timestamp_log or "off")
        self.camera_dir_mode = str(camera_dir_mode or "short")
        drop_str = str(camera_dir_drop_tokens or "World,Franka,base_link")
        self.camera_dir_drop_tokens = [t.strip() for t in drop_str.split(",") if t.strip()]

        self._queue: Queue = Queue(maxsize=max(1, int(queue_size)))
        self._stop = False
        self._need_save = False
        self._episode_open = False
        self._first_item = True
        self._item_id = -1
        self._episode_id = 0
        self._dropped = 0

        # Stats for diagnosing throughput bottlenecks.
        self._enqueued = 0
        self._processed = 0
        self._queue_peak = 0
        self._episode_start_wall = None
        self._episode_start_t = None
        self._episode_end_t = None

        self._executor = ThreadPoolExecutor(max_workers=max(1, int(writer_workers)))
        self._worker = Thread(target=self._process_queue, daemon=True)
        self._worker.start()

    def is_ready(self) -> bool:
        return not self._episode_open

    def create_episode(self, episode_id: int, camera_keys: Dict[str, str], dof_names: Optional[list] = None):
        if self._episode_open:
            return False

        self._episode_id = int(episode_id)
        self._item_id = -1
        self._first_item = True
        self._dropped = 0
        self._enqueued = 0
        self._processed = 0
        self._queue_peak = 0
        self._episode_start_wall = time.time()
        self._episode_start_t = None
        self._episode_end_t = None

        self.episode_dir = self.root_dir / f"episode_{self._episode_id:04d}"
        self.colors_dir = self.episode_dir / "colors"
        self.depths_dir = self.episode_dir / "depths"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.colors_dir.mkdir(parents=True, exist_ok=True)
        self.depths_dir.mkdir(parents=True, exist_ok=True)

        # Per-camera subdirs: use prim path as folder hierarchy.
        self._color_subdirs: Dict[str, Path] = {}
        self._depth_subdirs: Dict[str, Path] = {}
        for cam_key, prim_path in (camera_keys or {}).items():
            rel_dir = self._prim_path_to_rel_dir(str(prim_path))
            color_dir = self.colors_dir / rel_dir
            depth_dir = self.depths_dir / rel_dir
            color_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            self._color_subdirs[str(cam_key)] = color_dir
            self._depth_subdirs[str(cam_key)] = depth_dir

        info = {
            "version": "1.0.0",
            "date": _dt.date.today().strftime("%Y-%m-%d"),
            "image": {"width": self.resolution[0], "height": self.resolution[1], "fps": self.fps, "format": self.rgb_format},
            "depth": {"width": self.resolution[0], "height": self.resolution[1], "fps": self.fps, "format": self.depth_format},
            "camera_keys": camera_keys,
            "joint_names": {"franka": dof_names or []},
        }

        self._dof_names = list(dof_names or [])
        self._warned_no_franka_state = False

        self.json_path = self.episode_dir / "data.json"
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write("{\n")
            f.write('"info": ' + json.dumps(info, ensure_ascii=False, indent=2) + ",\n")
            f.write('"data": [\n')

        # Timeline log (idx,t) for visualization/debug.
        self.timeline_path = self.episode_dir / "timeline.csv"
        with open(self.timeline_path, "w", encoding="utf-8") as f:
            f.write("idx,t\n")

        # Robot joint states (one row per frame). Keep this separate from JSON for easy analysis.
        self.franka_states_path = self.episode_dir / "franka_states.csv"
        with open(self.franka_states_path, "w", encoding="utf-8") as f:
            cols = ["idx", "t"]
            cols += [f"qpos.{name}" for name in self._dof_names]
            cols += [f"qvel.{name}" for name in self._dof_names]
            cols += [f"eff.{name}" for name in self._dof_names]
            f.write(",".join(cols) + "\n")

        self._episode_open = True
        return True

    def add_item(self, item: dict) -> bool:
        if not self._episode_open:
            return False

        if self._queue.full():
            if self.drop_when_full:
                self._dropped += 1
                return False
            self._queue.put(item)
            self._enqueued += 1
            try:
                self._queue_peak = max(self._queue_peak, int(self._queue.qsize()))
            except Exception:
                pass
            return True

        self._queue.put(item)
        self._enqueued += 1
        try:
            self._queue_peak = max(self._queue_peak, int(self._queue.qsize()))
        except Exception:
            pass
        return True

    def save_episode(self):
        self._need_save = True

    def close(self):
        if self._episode_open:
            self.save_episode()
        while self._episode_open:
            time.sleep(0.01)
        self._stop = True
        self._worker.join()
        self._executor.shutdown(wait=True)

    def _process_queue(self):
        while not self._stop or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                item = None

            if item is not None:
                try:
                    self._process_item(item)
                except Exception:
                    pass
                self._queue.task_done()

            if self._need_save and self._queue.empty() and self._episode_open:
                self._finalize_episode()

    def _submit_save_rgb(self, path: Path, rgb: np.ndarray):
        import cv2

        def _write():
            if self.rgb_format == "png":
                cv2.imwrite(str(path), rgb[:, :, ::-1])  # RGB->BGR
            else:
                params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpg_quality)]
                cv2.imwrite(str(path), rgb[:, :, ::-1], params)  # RGB->BGR

        self._executor.submit(_write)

    def _submit_save_depth(self, path: Path, depth: np.ndarray):
        import cv2

        def _write():
            if self.depth_format == "png":
                cv2.imwrite(str(path), depth)
            else:
                np.save(str(path), depth)

        self._executor.submit(_write)

    def _process_item(self, item: dict):
        self._item_id += 1
        idx = int(item.get("idx", self._item_id))

        # Timestamp visualization/logging.
        t = item.get("t", None)
        if t is not None:
            try:
                t_val = float(t)
            except Exception:
                t_val = None

            if t_val is not None:
                if self._episode_start_t is None:
                    self._episode_start_t = t_val
                self._episode_end_t = t_val

                if self.timestamp_log in ("file", "both"):
                    try:
                        with open(self.timeline_path, "a", encoding="utf-8") as f:
                            f.write(f"{idx},{t_val:.6f}\n")
                    except Exception:
                        pass

                if self.timestamp_log in ("print", "both"):
                    try:
                        # One line per timestamp (can be noisy for long episodes).
                        print(f"[episode {self._episode_id:04d}] idx={idx:06d} t={t_val:.6f}")
                    except Exception:
                        pass

        # Robot state logging.
        try:
            states = item.get("states", {}) or {}
            franka_state = states.get("franka", None) if isinstance(states, dict) else None
            if isinstance(franka_state, dict):
                qpos = franka_state.get("qpos", []) or []
                qvel = franka_state.get("qvel", []) or []
                eff = franka_state.get("eff", []) or []

                def _to_float_list(x):
                    out = []
                    for v in (x or []):
                        try:
                            out.append(float(v))
                        except Exception:
                            out.append(float("nan"))
                    return out

                qpos_f = _to_float_list(qpos)
                qvel_f = _to_float_list(qvel)
                eff_f = _to_float_list(eff)

                # Pad/truncate to match declared DOF list for stable CSV shape.
                n = len(getattr(self, "_dof_names", []) or [])
                if n > 0:
                    qpos_f = (qpos_f + [float("nan")] * n)[:n]
                    qvel_f = (qvel_f + [float("nan")] * n)[:n]
                    eff_f = (eff_f + [float("nan")] * n)[:n]

                t_for_csv = None
                try:
                    t_for_csv = float(item.get("t", float("nan")))
                except Exception:
                    t_for_csv = float("nan")

                with open(self.franka_states_path, "a", encoding="utf-8") as f:
                    row = [str(int(idx)), f"{t_for_csv:.6f}"]
                    row += [f"{v:.8f}" for v in qpos_f]
                    row += [f"{v:.8f}" for v in qvel_f]
                    row += [f"{v:.8f}" for v in eff_f]
                    f.write(",".join(row) + "\n")
            else:
                # If we expected joint names but never saw states, warn once.
                if (getattr(self, "_dof_names", None) and not getattr(self, "_warned_no_franka_state", False)):
                    print(
                        f"[episode {self._episode_id:04d}] warning: 未收到 franka 关节状态(states['franka']). "
                        f"请确认脚本暴露 my_franka 且支持 get_joint_positions/get_joint_velocities/get_joint_efforts。"
                    )
                    self._warned_no_franka_state = True
        except Exception:
            pass

        colors: Dict[str, np.ndarray] = item.get("colors", {}) or {}
        depths: Dict[str, np.ndarray] = item.get("depths", {}) or {}

        # 先生成路径，写到 metadata；实际写盘交给线程池。
        rel_colors = {}
        for cam_key, rgb in colors.items():
            subdir = getattr(self, "_color_subdirs", {}).get(cam_key)
            if subdir is None:
                # Backward compatible fallback: put in root.
                subdir = self.colors_dir
                rel_dir = Path("")
            else:
                rel_dir = subdir.relative_to(self.episode_dir)  # e.g. colors/World/Camera

            name = f"{idx:06d}.{self.rgb_format}"
            rel = str(rel_dir / name)
            rel_colors[cam_key] = rel
            self._submit_save_rgb(subdir / name, rgb)

        rel_depths = {}
        for cam_key, depth in depths.items():
            ext = "png" if self.depth_format == "png" else "npy"
            subdir = getattr(self, "_depth_subdirs", {}).get(cam_key)
            if subdir is None:
                subdir = self.depths_dir
                rel_dir = Path("")
            else:
                rel_dir = subdir.relative_to(self.episode_dir)  # e.g. depths/World/Camera

            name = f"{idx:06d}.{ext}"
            rel = str(rel_dir / name)
            rel_depths[cam_key] = rel
            self._submit_save_depth(subdir / name, depth)

        item_to_write = dict(item)
        item_to_write["idx"] = idx
        item_to_write["colors"] = rel_colors
        item_to_write["depths"] = rel_depths

        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self._first_item:
                f.write(",\n")
            f.write(json.dumps(item_to_write, ensure_ascii=False))
            self._first_item = False

        self._processed += 1

    def _finalize_episode(self):
        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write("\n]\n}")

        # Print episode stats for throughput diagnosis.
        try:
            wall_now = time.time()
            wall_start = float(self._episode_start_wall or wall_now)
            wall_dur = max(0.0, wall_now - wall_start)

            t0 = self._episode_start_t
            t1 = self._episode_end_t
            sim_dur = None
            if t0 is not None and t1 is not None:
                sim_dur = max(0.0, float(t1) - float(t0))

            effective_fps = None
            if sim_dur is not None and sim_dur > 1e-6:
                effective_fps = float(self._processed) / sim_dur

            msg = (
                f"[episode {self._episode_id:04d}] done: "
                f"written_frames={self._processed} "
                f"dropped_frames={self._dropped} "
                f"enqueued_frames={self._enqueued} "
                f"queue_peak={self._queue_peak} "
                f"wall_sec={wall_dur:.2f}"
            )
            if effective_fps is not None:
                msg += f" sim_fps~{effective_fps:.2f}"
            print(msg)
            print(f"[episode {self._episode_id:04d}] timeline: {self.timeline_path}")
            try:
                print(f"[episode {self._episode_id:04d}] franka_states: {self.franka_states_path}")
            except Exception:
                pass
        except Exception:
            pass

        self._need_save = False
        self._episode_open = False


class CaptureSessionV1:
    def __init__(
        self,
        world,
        franka,
        output_root: str,
        fps: float,
        resolution: Tuple[int, int],
        rgb_format: str,
        jpg_quality: int,
        depth_format: str,
        writer_workers: int,
        queue_size: int,
        drop_when_full: bool,
        timestamp_log: str,
        camera_dir_mode: str,
        camera_dir_drop_tokens: str,
    ):
        import omni.replicator.core as rep

        self.world = world
        self.franka = franka
        self.rep = rep
        self.fps = fps if fps > 0 else DEFAULT_FPS
        self.capture_interval = 1.0 / self.fps
        self.resolution = tuple(resolution)

        root_dir = Path(output_root).expanduser().absolute()
        timestamp = time.strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = root_dir / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.sim_time_sec = 0.0
        self.capture_time_accum = 0.0
        self.capture_time_sec = 0.0  # 独立的采集时间戳（不包含规划时间）
        self.frame_index = 0

        self.camera_entries = []
        camera_paths = _discover_camera_paths()
        self.camera_keys = {}
        for cam_path in camera_paths:
            cam_key = _sanitize_camera_key(cam_path)
            self.camera_keys[cam_key] = cam_path
            render_product = rep.create.render_product(cam_path, self.resolution)

            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            try:
                rgb_annotator.attach([render_product])
            except Exception:
                rgb_annotator.attach(render_product)

            depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera", device="cpu")
            try:
                depth_annotator.attach([render_product])
            except Exception:
                depth_annotator.attach(render_product)

            self.camera_entries.append(
                {
                    "path": cam_path,
                    "key": cam_key,
                    "rgb": rgb_annotator,
                    "depth": depth_annotator,
                }
            )

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

        self.writer = AsyncEpisodeDatasetWriter(
            root_dir=self.session_dir,
            fps=self.fps,
            resolution=self.resolution,
            rgb_format=rgb_format,
            jpg_quality=jpg_quality,
            depth_format=depth_format,
            writer_workers=writer_workers,
            queue_size=queue_size,
            drop_when_full=drop_when_full,
            timestamp_log=timestamp_log,
            camera_dir_mode=camera_dir_mode,
            camera_dir_drop_tokens=camera_dir_drop_tokens,
        )

        self.episode_index = 0

    def advance_time(self, physics_dt: float, is_planning: bool = False):
        """Advance time. When planning, only advance sim time, not capture time."""
        self.sim_time_sec += physics_dt
        if not is_planning:
            self.capture_time_accum += physics_dt
            self.capture_time_sec += physics_dt

    def reset_capture_timer(self):
        self.capture_time_accum = 0.0

    def start_episode(self):
        self.episode_index += 1
        self.frame_index = 0
        self.capture_time_accum = 0.0
        self.capture_time_sec = 0.0  # 重置采集时间戳
        self.writer.create_episode(self.episode_index, camera_keys=self.camera_keys, dof_names=self.dof_names)

    def end_episode(self):
        self.writer.save_episode()

    def close(self):
        self.writer.close()

    def _read_franka_state(self) -> dict:
        if self.franka is None or self.dof_count == 0:
            return {}

        pos = getattr(self.franka, "get_joint_positions", lambda: None)()
        vel = getattr(self.franka, "get_joint_velocities", lambda: None)()
        eff = getattr(self.franka, "get_joint_efforts", lambda: None)()

        pos = list(pos) if pos is not None else []
        vel = list(vel) if vel is not None else []
        eff = list(eff) if eff is not None else []

        return {
            "qpos": pos,
            "qvel": vel,
            "eff": eff,
        }

    def capture_if_needed(self):
        if not self.camera_entries:
            self.capture_time_accum = 0.0
            return

        while self.capture_time_accum >= self.capture_interval:
            self.capture_time_accum -= self.capture_interval
            self.frame_index += 1

            colors = {}
            depths = {}
            for entry in self.camera_entries:
                cam_key = entry["key"]
                rgb = _rgb_frame_to_numpy(entry["rgb"].get_data(), self.resolution)
                if rgb is not None:
                    colors[cam_key] = np.array(rgb, copy=True)

                depth = _depth_frame_to_uint16(entry["depth"].get_data(), self.resolution)
                if depth is not None:
                    depths[cam_key] = np.array(depth, copy=True)

            states = {}
            franka_state = self._read_franka_state()
            if franka_state:
                states["franka"] = franka_state

            item = {
                "idx": self.frame_index,
                "t": float(self.capture_time_sec),  # 使用采集时间戳（不包含规划时间）
                "colors": colors,
                "depths": depths,
                "states": states,
            }
            self.writer.add_item(item)


def collect_from_module(
    script: str,
    out_dir: str,
    fps: float,
    headless: bool,
    exit_on_complete: bool,
    resolution: Tuple[int, int],
    rgb_format: str,
    jpg_quality: int,
    depth_format: str,
    writer_workers: int,
    queue_size: int,
    drop_when_full: bool,
    timestamp_log: str,
    camera_dir_mode: str,
    camera_dir_drop_tokens: str,
):
    os.environ["ISAACSIM_HEADLESS"] = "1" if headless else "0"

    module = _load_module(script)
    simulation_app = getattr(module, "simulation_app", None)
    my_world = getattr(module, "my_world", None)
    if simulation_app is None or my_world is None:
        raise RuntimeError("模块需要暴露 simulation_app 和 my_world")

    step_fn = _resolve_step_function(module)
    my_franka = getattr(module, "my_franka", None)

    collector = CaptureSessionV1(
        my_world,
        my_franka,
        out_dir,
        fps=fps,
        resolution=resolution,
        rgb_format=rgb_format,
        jpg_quality=jpg_quality,
        depth_format=depth_format,
        writer_workers=writer_workers,
        queue_size=queue_size,
        drop_when_full=drop_when_full,
        timestamp_log=timestamp_log,
        camera_dir_mode=camera_dir_mode,
        camera_dir_drop_tokens=camera_dir_drop_tokens,
    )

    _ensure_timeline_playing()
    _auto_start_world(my_world)

    controller = getattr(module, "my_controller", None)
    start_hook = getattr(module, "collect_should_start_capture", None)
    stop_hook = getattr(module, "collect_should_stop_capture", None)

    capturing_active = False
    terminate_after_cycle = False

    try:
        while simulation_app.is_running():
            step_result = _call_step_function(step_fn)
            if step_result is False:
                break

            physics_dt = _safe_get_physics_dt(my_world)
            
            # 检测是否正在进行 CuRobo 规划（规划期间不采集）
            is_planning = False
            if controller is not None and hasattr(controller, "is_planning"):
                try:
                    is_planning = bool(controller.is_planning)
                except Exception:
                    is_planning = False
            
            # 传递规划状态：规划时只更新仿真时间，不更新采集时间
            collector.advance_time(physics_dt, is_planning=is_planning)

            should_start = False
            should_stop = False

            if callable(start_hook) or callable(stop_hook):
                if callable(start_hook) and not capturing_active:
                    should_start = bool(start_hook())
                if callable(stop_hook) and capturing_active:
                    should_stop = bool(stop_hook())
            else:
                event_id, controller_done = _query_controller_state(controller)
                if not capturing_active and event_id is not None and event_id >= 0 and not controller_done:
                    should_start = True
                if capturing_active and controller_done:
                    should_stop = True

            if should_start:
                capturing_active = True
                collector.start_episode()

            if should_stop:
                collector.end_episode()
                capturing_active = False
                collector.reset_capture_timer()
                if exit_on_complete:
                    terminate_after_cycle = True
                    break
                continue

            if not capturing_active:
                collector.reset_capture_timer()
                continue

            # 规划期间不采集（跳过 capture_if_needed）
            if is_planning:
                continue

            collector.capture_if_needed()

            if terminate_after_cycle:
                break

    finally:
        collector.close()
        simulation_app.close()


def parse_args():
    p = argparse.ArgumentParser(description="异步 episode 数据采集（高频连拍）")
    p.add_argument("--script", "-s", default="scipy/collect_pick_place_seed", help="任务脚本(不含 .py)")
    p.add_argument("--out", "-o", default="./collect_output", help="输出根目录")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="采样频率 (Hz)")

    p.add_argument("--width", type=int, default=CAPTURE_RESOLUTION[0], help="图像宽")
    p.add_argument("--height", type=int, default=CAPTURE_RESOLUTION[1], help="图像高")

    p.add_argument("--rgb-format", choices=("jpg", "png"), default="jpg")
    p.add_argument("--jpg-quality", type=int, default=95)
    p.add_argument("--depth-format", choices=("npy", "png"), default="npy")

    p.add_argument("--writer-workers", type=int, default=8)
    p.add_argument("--queue-size", type=int, default=256)
    p.add_argument("--drop-when-full", action="store_true", help="队列满时丢帧以保证仿真不阻塞")
    p.add_argument("--block-when-full", dest="drop_when_full", action="store_false", help="队列满时阻塞（更完整但更慢）")
    p.set_defaults(drop_when_full=True)

    p.add_argument(
        "--timestamp-log",
        choices=("off", "print", "file", "both"),
        default="both",
        help="时间戳可视化输出：off=关闭，print=终端逐帧打印，file=写 timeline.csv，both=两者都做",
    )

    p.add_argument(
        "--camera-dir-mode",
        choices=("short", "flat", "hierarchy"),
        default="short",
        help="相机目录命名：short=短名(默认)，flat=全路径下划线拼接，hierarchy=按 prim path 分层目录",
    )
    p.add_argument(
        "--camera-dir-drop-tokens",
        default="World,Franka,base_link",
        help="camera-dir-mode=short 时要丢弃的 prim path 组件，逗号分隔",
    )

    group = p.add_mutually_exclusive_group()
    group.add_argument("--headless", dest="headless", action="store_true", help="无头模式")
    group.add_argument("--gui", dest="headless", action="store_false", help="强制 GUI")
    p.set_defaults(headless=True)

    exit_group = p.add_mutually_exclusive_group()
    exit_group.add_argument("--exit-on-complete", dest="exit_on_complete", action="store_true")
    exit_group.add_argument("--keep-alive", dest="exit_on_complete", action="store_false")
    p.set_defaults(exit_on_complete=True)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 运行入口：把命令行参数传给采集主函数 collect_from_module()
    #
    # 参数含义（与 parse_args() 一一对应）：
    # - args.script: 任务脚本名或路径（默认 pick_place）。可写不带 .py 的模块名，或直接给文件路径。
    # - args.out: 输出根目录（默认 ./collect_output）。会在其下创建 session_xxx/episode_xxxx/... 结构。
    # - args.fps: 采集频率 (Hz)。例如 30 代表每秒采 30 帧（理想情况；写盘跟不上时可能丢帧/堆积）。
    # - args.headless: 是否无头运行。可选：--headless / --gui。

    # - args.exit_on_complete: episode 结束后是否退出。可选：--exit-on-complete / --keep-alive。
    # - resolution=(args.width, args.height): 相机输出分辨率。

    # - args.rgb_format: RGB 图片格式。可选：jpg / png。
    # - args.jpg_quality: JPG 质量（仅 rgb-format=jpg 生效）。范围建议 1~100，数值越大质量越好/更慢。
    # - args.depth_format: 深度保存格式。可选：npy / png（npy 通常更快且无损）。
    # - args.writer_workers: 后台写盘线程池并发数（ThreadPoolExecutor max_workers）。
    # - args.queue_size: 主线程->写盘线程的队列长度。越大越能“缓冲突发”，但会占更多内存。

    # - args.drop_when_full: 队列满时策略。
    #   - True: 丢帧（--drop-when-full，默认）保证仿真/主循环尽量不被写盘拖慢
    #   - False: 阻塞等待（--block-when-full）保证更完整但可能明显拖慢仿真

    # - args.timestamp_log: 时间戳可视化输出方式。可选：off / print / file / both。
    #   - print: 逐帧在终端打印 idx,t（可能刷屏）
    #   - file: 写 episode_xxxx/timeline.csv

    # - args.camera_dir_mode: 相机目录命名方式。可选：short / flat / hierarchy。
    #   - short: 默认，生成更短的单层目录名（例如 franka_panda_hand_ZED_X_CameraLeft）
    #   - flat: 用 prim path 全部组件下划线拼接
    #   - hierarchy: 按 prim path 分层目录（最“像原始路径”，但层级最深）

    # - args.camera_dir_drop_tokens: 仅在 camera-dir-mode=short 时生效。
    #   逗号分隔的 prim path 组件黑名单（例如默认丢弃 World,Franka,base_link）。
    collect_from_module(
        args.script,
        args.out,
        fps=args.fps,
        headless=args.headless,
        exit_on_complete=args.exit_on_complete,
        resolution=(args.width, args.height),
        rgb_format=args.rgb_format,
        jpg_quality=args.jpg_quality,
        depth_format=args.depth_format,
        writer_workers=args.writer_workers,
        queue_size=args.queue_size,
        drop_when_full=args.drop_when_full,
        timestamp_log=args.timestamp_log,
        camera_dir_mode=args.camera_dir_mode,
        camera_dir_drop_tokens=args.camera_dir_drop_tokens,
    )


#/home/di-gua/isaac-sim/python.sh collect_curobo.py --script pick_place_localFranka_curobo --out /home/di-gua/licheng/manipulation/manipulation_collect/collect_output --fps 30 --width 1280 --height 960 --rgb-format png --depth-format png --timestamp-log both --camera-dir-mode short