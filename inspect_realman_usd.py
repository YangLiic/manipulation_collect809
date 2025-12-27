#!/usr/bin/env python3
"""Inspect Realman USD structure"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': True})

from pxr import Usd, UsdGeom, UsdPhysics
import omni.usd

# Load USD
stage = Usd.Stage.Open('/home/yons/data/realman_usd/realman.usd')

print("=" * 60)
print("REALMAN USD STRUCTURE ANALYSIS")
print("=" * 60)

# Find all prims
joints = []
links = []
grippers = []

for prim in stage.Traverse():
    path = str(prim.GetPath())
    type_name = prim.GetTypeName()
    
    # Collect joints
    if 'Joint' in type_name:
        joints.append((path, type_name))
    
    # Collect links
    if 'link' in path.lower() or prim.IsA(UsdGeom.Xform):
        if any(x in path.lower() for x in ['link', 'base', 'ee', 'end_effector', 'tool']):
            links.append(path)
    
    # Collect gripper info
    if any(x in path.lower() for x in ['gripper', 'finger', 'hand']):
        grippers.append((path, type_name))

print(f"\n📋 Found {len(joints)} joints:")
for path, type_name in joints[:20]:  # Limit output
    print(f"  {path} ({type_name})")

print(f"\n🔗 Found {len(links)} potential links:")
for path in links[:20]:
    print(f"  {path}")

print(f"\n🤏 Found {len(grippers)} gripper components:")
for path, type_name in grippers[:10]:
    print(f"  {path} ({type_name})")

# Get root prim
root = stage.GetDefaultPrim()
if root:
    print(f"\n🌳 Root prim: {root.GetPath()}")
    print(f"   Type: {root.GetTypeName()}")

simulation_app.close()
