"""
Test script for grasp sampler
Tests mesh loading and grasp pose generation
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.utils.stage import open_stage
from grasp_sampler import AntipodalGraspSampler

# Load stage
open_stage("/home/yons/data/Collected_World1/World0.usd")
simulation_app.update()

# Create sampler
print("🎯 Creating grasp sampler...")
sampler = AntipodalGraspSampler(
    gripper_width=0.08,
    standoff_distance=0.02,
    min_grasp_width=0.01
)

# Load mesh
print("\n📦 Loading object mesh...")
object_path = "/World/Vegetable_7"
success = sampler.load_mesh_from_usd(object_path)

if success:
    print("\n🔍 Sampling grasp poses...")
    grasps = sampler.sample_antipodal_grasps(
        num_samples=20,
        num_orientations=4,
        random_seed=42
    )
    
    if grasps:
        print(f"\n✅ Generated {len(grasps)} grasps")
        print("\nTop 5 grasps:")
        for i, grasp in enumerate(grasps[:5]):
            print(f"  {i+1}. {grasp}")
        
        best_grasp = sampler.get_best_grasp()
        print(f"\n🏆 Best grasp: {best_grasp}")
    else:
        print("\n❌ No grasps generated")
else:
    print("\n❌ Failed to load mesh")

print("\n✅ Test complete!")
simulation_app.close()
