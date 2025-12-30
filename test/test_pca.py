#!/usr/bin/env python3
"""Test PCA analysis on the vegetable mesh"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': True})

import numpy as np
from isaacsim.core.utils.stage import open_stage
from grasp_sampler import AntipodalGraspSampler

open_stage('/home/yons/data/Collected_World1/World0.usd')
simulation_app.update()

sampler = AntipodalGraspSampler(gripper_width=0.08)
success = sampler.load_mesh_from_usd('/World/Vegetable_7')

if success:
    # Run PCA analysis
    xy_vertices = sampler.mesh.vertices[:, :2]
    xy_centered = xy_vertices - np.mean(xy_vertices, axis=0)
    cov_matrix = np.cov(xy_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    principal_axis = eigenvectors[:, 0]
    secondary_axis = eigenvectors[:, 1]
    
    object_angle = np.arctan2(principal_axis[1], principal_axis[0])
    
    print(f"\n=== PCA Analysis ===")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Ratio: {eigenvalues[0]/eigenvalues[1]:.2f}:1")
    print(f"Principal axis: [{principal_axis[0]:.4f}, {principal_axis[1]:.4f}]")
    print(f"Secondary axis: [{secondary_axis[0]:.4f}, {secondary_axis[1]:.4f}]")
    print(f"Object angle: {np.degrees(object_angle):.2f}°")
    print(f"Perpendicular angle: {np.degrees(object_angle + np.pi/2):.2f}°")
    print(f"\nUser says 45° is correct")
    print(f"Difference: {abs(45 - np.degrees(object_angle)):.2f}°")

simulation_app.close()
