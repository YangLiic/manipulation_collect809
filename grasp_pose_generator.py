"""
Lightweight Grasp Pose Generator using scipy
Generates random grasp orientations with Z-axis rotation and slight tilts
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_grasp_pose(
    base_orientation=None,
    z_rotation_range=(-180, 180),
    tilt_range=(-5, 5),
    random_seed=None
):
    """
    Generate a random grasp orientation (quaternion) with Z-rotation and slight tilt
    
    Args:
        base_orientation: Base quaternion [w, x, y, z]. Default is downward [0, 1, 0, 0]
        z_rotation_range: Tuple (min, max) in degrees for Z-axis rotation
        tilt_range: Tuple (min, max) in degrees for X/Y tilt
        random_seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Quaternion [w, x, y, z] representing the grasp orientation
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Default base orientation: downward (180° around X-axis)
    if base_orientation is None:
        base_orientation = np.array([0.0, 1.0, 0.0, 0.0])  # [w, x, y, z]
    
    # Convert base orientation to scipy Rotation
    base_rot = R.from_quat([
        base_orientation[1],  # x
        base_orientation[2],  # y
        base_orientation[3],  # z
        base_orientation[0]   # w
    ])  # scipy uses [x, y, z, w] format
    
    # Generate random Z-rotation
    z_angle = np.random.uniform(z_rotation_range[0], z_rotation_range[1])
    z_rot = R.from_euler('z', z_angle, degrees=True)
    
    # Generate random tilt (small rotations around X and Y)
    tilt_x = np.random.uniform(tilt_range[0], tilt_range[1])
    tilt_y = np.random.uniform(tilt_range[0], tilt_range[1])
    tilt_rot = R.from_euler('xy', [tilt_x, tilt_y], degrees=True)
    
    # Compose rotations: base -> tilt -> z_rotation
    final_rot = z_rot * tilt_rot * base_rot
    
    # Convert back to quaternion [w, x, y, z]
    quat_scipy = final_rot.as_quat()  # [x, y, z, w]
    quat_output = np.array([
        quat_scipy[3],  # w
        quat_scipy[0],  # x
        quat_scipy[1],  # y
        quat_scipy[2]   # z
    ])
    
    return quat_output


def quaternion_to_euler(quat, degrees=True):
    """
    Convert quaternion to Euler angles for debugging
    
    Args:
        quat: Quaternion [w, x, y, z]
        degrees: Return in degrees if True, radians if False
        
    Returns:
        np.ndarray: Euler angles [roll, pitch, yaw]
    """
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x, y, z, w]
    euler = rot.as_euler('xyz', degrees=degrees)
    return euler


if __name__ == "__main__":
    """Test the grasp pose generator"""
    print("🧪 Testing Grasp Pose Generator\n")
    
    # Generate 5 random poses
    print("Generating 5 random grasp poses:")
    print("-" * 60)
    
    for i in range(5):
        quat = generate_grasp_pose(
            z_rotation_range=(-180, 180),
            tilt_range=(-3, 3)
        )
        
        # Verify quaternion is normalized
        norm = np.linalg.norm(quat)
        
        # Convert to Euler for readability
        euler = quaternion_to_euler(quat, degrees=True)
        
        print(f"Pose {i+1}:")
        print(f"  Quaternion [w,x,y,z]: {quat}")
        print(f"  Norm: {norm:.6f} (should be 1.0)")
        print(f"  Euler [r,p,y]: [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]")
        print()
    
    print("✅ Test complete!")
