#!/usr/bin/env python3
"""Debug Realman gripper joints"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage, add_reference_to_stage
from isaacsim.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation as ArticulationCore

REALMAN_USD = "/home/yons/data/realman_usd/realman.usd"
REALMAN_PATH = "/World/RmJa"

open_stage("/home/yons/data/Collected_World1/World0.usd")
simulation_app.update()

my_world = World(stage_units_in_meters=1.0)

# Load Realman
print("Loading Realman...")
add_reference_to_stage(usd_path=REALMAN_USD, prim_path=REALMAN_PATH)
simulation_app.update()

realman_base = XFormPrim(REALMAN_PATH)
realman_base.set_world_poses(positions=np.array([[0.0, 0.6, -0.25]]))
simulation_app.update()

my_world.reset()

realman_art = ArticulationCore(REALMAN_PATH, "Realman")
realman_art.initialize()

print(f"\n{'='*60}")
print(f"REALMAN JOINT ANALYSIS")
print(f"{'='*60}")
print(f"Total DOF: {realman_art.num_dof}")
print(f"\nAll joint names:")
for i, name in enumerate(realman_art.dof_names):
    print(f"  [{i}] {name}")

print(f"\n{'='*60}")
print("Testing gripper control...")
print(f"{'='*60}\n")

# Get articulation controller
controller = realman_art.get_articulation_controller()

# Wait for physics
my_world.step(render=True)
my_world.step(render=True)

# Test: Set gripper joints to different values
print("Initial positions:")
positions = realman_art.get_joint_positions()
if positions is not None:
    for i in range(6, min(10, len(positions))):
        print(f"  Joint {i} ({realman_art.dof_names[i]}): {positions[i]:.4f}")

print("\n🔧 Setting gripper joints to 0.04 (open)...")
from isaacsim.core.utils.types import ArticulationAction

# Try using ArticulationAction with controller
gripper_indices = list(range(6, min(10, realman_art.num_dof)))
open_positions = [0.04] * len(gripper_indices)

action = ArticulationAction(
    joint_positions=open_positions,
    joint_indices=gripper_indices
)
controller.apply_action(action)

# Step simulation
for _ in range(50):
    my_world.step(render=True)

print("After opening:")
positions = realman_art.get_joint_positions()
if positions is not None:
    for i in gripper_indices:
        print(f"  Joint {i} ({realman_art.dof_names[i]}): {positions[i]:.4f}")

print("\n🔧 Setting gripper joints to 0.0 (close)...")
close_positions = [0.0] * len(gripper_indices)
action = ArticulationAction(
    joint_positions=close_positions,
    joint_indices=gripper_indices
)
controller.apply_action(action)

for _ in range(50):
    my_world.step(render=True)

print("After closing:")
positions = realman_art.get_joint_positions()
if positions is not None:
    for i in gripper_indices:
        print(f"  Joint {i} ({realman_art.dof_names[i]}): {positions[i]:.4f}")

print("\n✅ Debug complete. Check if gripper moved visually!")
print("Press Ctrl+C to exit")

while simulation_app.is_running():
    my_world.step(render=True)

simulation_app.close()
