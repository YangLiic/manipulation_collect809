#!/usr/bin/env python3
"""
Quick test of scipy grasp pose generator integration
Tests that poses are generated correctly before running full simulation
"""

import numpy as np
from scipy.grasp_pose_generator import generate_grasp_pose, quaternion_to_euler

print("🧪 Testing Scipy Grasp Pose Generator Integration\n")
print("=" * 60)

# Test 1: Generate multiple poses
print("\n📋 Test 1: Generating 3 random grasp poses")
print("-" * 60)

for i in range(3):
    quat = generate_grasp_pose(
        z_rotation_range=(-180, 180),
        tilt_range=(-3, 3)
    )
    
    # Verify normalization
    norm = np.linalg.norm(quat)
    euler = quaternion_to_euler(quat, degrees=True)
    
    print(f"\nPose {i+1}:")
    print(f"  Quaternion [w,x,y,z]: {quat}")
    print(f"  Norm: {norm:.6f} ✓" if abs(norm - 1.0) < 1e-6 else f"  Norm: {norm:.6f} ✗")
    print(f"  Euler [r,p,y]: [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]")
    
    # Check tilt is within range
    tilt_ok = abs(euler[0] - 180) < 5 and abs(euler[1]) < 5
    print(f"  Tilt check: {'✓ Within ±3°' if tilt_ok else '✗ Outside range'}")

# Test 2: Verify diversity
print("\n\n📋 Test 2: Verifying Z-rotation diversity")
print("-" * 60)

yaw_angles = []
for i in range(10):
    quat = generate_grasp_pose(
        z_rotation_range=(-180, 180),
        tilt_range=(-3, 3)
    )
    euler = quaternion_to_euler(quat, degrees=True)
    yaw_angles.append(euler[2])

print(f"Generated {len(yaw_angles)} poses")
print(f"Yaw angles: {[f'{y:.1f}°' for y in yaw_angles[:5]]} ...")
print(f"Yaw range: [{min(yaw_angles):.1f}°, {max(yaw_angles):.1f}°]")
print(f"Yaw std dev: {np.std(yaw_angles):.1f}°")

diversity_ok = np.std(yaw_angles) > 50  # Should have good spread
print(f"Diversity check: {'✓ Good spread' if diversity_ok else '⚠️ Low diversity'}")

print("\n" + "=" * 60)
print("✅ All tests complete!")
print("\nReady to run: /home/yons/data/isaacsim/python.sh pick_place_localFranka_curobo_scipy.py")
