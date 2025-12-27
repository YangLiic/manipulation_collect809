# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
"""
Realman Robot Pick-Place Script (Simplified Version)
Direct joint control without Articulation wrapper
"""

import os
import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage, add_reference_to_stage
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.types import ArticulationAction
import omni.usd

# Realman USD configuration
REALMAN_USD = "/home/yons/data/realman_usd/realman.usd"
REALMAN_PATH = "/World/RmJa"

# Load scene
open_stage("/home/yons/data/Collected_World1/World0.usd")
simulation_app.update()

# Create World
my_world = World(stage_units_in_meters=1.0)

# Wrap scene objects
salt = XFormPrim("/World/Vegetable_7")
bowl = XFormPrim("/World/Bowl_0")

# Fixed spawn position
fixed_spawn_pos = np.array([0.0, 0.6, -0.25], dtype=float)

# Load Realman robot
print(f"🤖 Loading Realman from: {REALMAN_USD}")
add_reference_to_stage(usd_path=REALMAN_USD, prim_path=REALMAN_PATH)
simulation_app.update()

# Set robot position
realman_base = XFormPrim(REALMAN_PATH)
realman_base.set_world_poses(positions=np.array([fixed_spawn_pos]))
simulation_app.update()

# Reset world to initialize physics
my_world.reset()

# Get articulation handle using PhysX API
from pxr import PhysxSchema
stage = omni.usd.get_context().get_stage()
realman_prim = stage.GetPrimAtPath(REALMAN_PATH)

# Get articulation API
from omni.isaac.core.articulations import Articulation as ArticulationCore
realman_art = ArticulationCore(REALMAN_PATH, "Realman")
realman_art.initialize()

print(f"📊 Realman DOF: {realman_art.num_dof}")
print(f"📊 Joint names: {realman_art.dof_names}")

# Define joint indices
ARM_JOINT_INDICES = list(range(6))
GRIPPER_JOINT_INDICES = list(range(6, min(10, realman_art.num_dof)))

print(f"🔧 Arm joints: {ARM_JOINT_INDICES}")
print(f"🤏 Gripper joints: {GRIPPER_JOINT_INDICES}")

# Home position
HOME_JOINTS = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0])

# Set initial position
print("🏠 Moving to home position...")
current_positions = realman_art.get_joint_positions()
if len(current_positions) >= 6:
    current_positions[ARM_JOINT_INDICES] = HOME_JOINTS
    realman_art.set_joint_positions(current_positions)
    my_world.step(render=True)
    print(f"✅ Home position set")

# Simple controller
# Simple controller
class SimpleController:
    def __init__(self, robot):
        self.robot = robot
        self.controller = robot.get_articulation_controller()
        self.current_event = 0
        self.wait_counter = 0
        self.gripper_open_pos = 0.04
        self.gripper_closed_pos = 0.0
        
    def open_gripper(self):
        """Open gripper using ArticulationAction"""
        from isaacsim.core.utils.types import ArticulationAction
        
        positions = self.robot.get_joint_positions()
        if positions is None:
            return
        
        # Set gripper joints to open position
        action = ArticulationAction(
            joint_positions=[self.gripper_open_pos] * len(GRIPPER_JOINT_INDICES),
            joint_indices=GRIPPER_JOINT_INDICES
        )
        self.controller.apply_action(action)
        print("🤏 Opening gripper")
        
    def close_gripper(self):
        """Close gripper using ArticulationAction"""
        from isaacsim.core.utils.types import ArticulationAction
        
        positions = self.robot.get_joint_positions()
        if positions is None:
            return
        
        # Set gripper joints to closed position
        action = ArticulationAction(
            joint_positions=[self.gripper_closed_pos] * len(GRIPPER_JOINT_INDICES),
            joint_indices=GRIPPER_JOINT_INDICES
        )
        self.controller.apply_action(action)
        print("🤏 Closing gripper")
    
    def get_current_event(self):
        return self.current_event
    
    def next_event(self):
        self.current_event += 1
        print(f"✅ Event {self.current_event}")
    
    def reset(self):
        self.current_event = 0
        self.wait_counter = 0


controller = SimpleController(realman_art)

reset_needed = False
first_run = True

def step_once(render=True):
    global reset_needed, first_run
    
    if not simulation_app.is_running():
        return False
    
    my_world.step(render=render)
    
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            controller.reset()
            first_run = True
            reset_needed = False
        
        if first_run:
            controller.open_gripper()
            first_run = False
        
        event = controller.get_current_event()
        
        # Simple test sequence
        if event == 0:
            if controller.wait_counter == 0:
                controller.wait_counter = 100
                print("⏳ Waiting...")
            else:
                controller.wait_counter -= 1
                if controller.wait_counter == 0:
                    controller.next_event()
        elif event == 1:
            controller.close_gripper()
            controller.wait_counter = 100
            controller.next_event()
        elif event == 2:
            if controller.wait_counter > 0:
                controller.wait_counter -= 1
            else:
                controller.open_gripper()
                controller.next_event()
        elif event == 3:
            print("🎉 Gripper test complete!")
            controller.next_event()
    
    return True

# Main loop
print("\n" + "=" * 60)
print("🚀 Starting Realman Simulation")
print("=" * 60)
print("Press PLAY to test gripper control")
print("=" * 60 + "\n")

while simulation_app.is_running():
    step_once()

simulation_app.close()
