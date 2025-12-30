#!/usr/bin/env python3
"""Quick test of bounding box sampling logic"""
import numpy as np

# Simulate object with extents [2.0, 0.4, 0.4] (20cm x 4cm x 4cm)
bbox_extents = np.array([2.0, 0.4, 0.4])
gripper_width = 0.08  # 8cm

directions = [
    (np.array([1, 0, 0]), "X+"),
    (np.array([-1, 0, 0]), "X-"),
    (np.array([0, 1, 0]), "Y+"),
    (np.array([0, -1, 0]), "Y-"),
    (np.array([0, 0, 1]), "Z+"),
    (np.array([0, 0, -1]), "Z-"),
]

print(f"Object extents: {bbox_extents}")
print(f"Gripper width: {gripper_width}m\n")

valid_grasps = 0
for approach_dir, label in directions:
    perp_mask = np.abs(approach_dir) < 0.5
    perp_extents = bbox_extents[perp_mask]
    
    if len(perp_extents) > 0:
        min_perp = np.min(perp_extents)
        max_perp = np.max(perp_extents)
        
        if min_perp > gripper_width:
            print(f"❌ {label}: min_perp={min_perp:.3f}m > {gripper_width}m - SKIP")
        else:
            print(f"✅ {label}: min_perp={min_perp:.3f}m ≤ {gripper_width}m - VALID (4 orientations)")
            valid_grasps += 4

print(f"\nTotal valid grasps: {valid_grasps}")
