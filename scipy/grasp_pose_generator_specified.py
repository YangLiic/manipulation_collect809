"""
指定姿态的抓取位姿生成器（使用 scipy）
根据用户指定的 Z 轴旋转角度和 X/Y 轴倾斜角度生成抓取姿态
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_grasp_pose(
    base_orientation=None,
    z_rotation=0.0,
    tilt_x=0.0,
    tilt_y=0.0
):
    """
    生成指定的抓取姿态（四元数）
    
    参数:
        base_orientation: 基础四元数 [w, x, y, z]。默认为向下姿态 [0, 1, 0, 0]
        z_rotation: Z 轴旋转角度（度），范围 -90 到 +90，正值为顺时针
        tilt_x: X 轴方向的倾斜角度（度），范围 -90 到 +90
        tilt_y: Y 轴方向的倾斜角度（度），范围 -90 到 +90
        
    返回:
        np.ndarray: 表示抓取姿态的四元数 [w, x, y, z]
    """
    # 参数范围检查
    if not -90 <= z_rotation <= 90:
        raise ValueError(f"z_rotation 必须在 -90 到 90 之间，当前值: {z_rotation}")
    if not -90 <= tilt_x <= 90:
        raise ValueError(f"tilt_x 必须在 -90 到 90 之间，当前值: {tilt_x}")
    if not -90 <= tilt_y <= 90:
        raise ValueError(f"tilt_y 必须在 -90 到 90 之间，当前值: {tilt_y}")
    
    # 默认基础姿态：向下（绕 X 轴旋转 180°）
    if base_orientation is None:
        base_orientation = np.array([0.0, 1.0, 0.0, 0.0])  # [w, x, y, z]
    
    # 将基础姿态转换为 scipy Rotation 对象
    base_rot = R.from_quat([
        base_orientation[1],  # x
        base_orientation[2],  # y
        base_orientation[3],  # z
        base_orientation[0]   # w
    ])  # scipy 使用 [x, y, z, w] 格式
    
    # 生成指定的 Z 轴旋转
    z_rot = R.from_euler('z', z_rotation, degrees=True)
    
    # 生成指定的 X/Y 轴倾斜
    tilt_rot = R.from_euler('xy', [tilt_x, tilt_y], degrees=True)
    
    # 组合旋转：基础姿态 -> 倾斜 -> Z 轴旋转
    final_rot = z_rot * tilt_rot * base_rot
    
    # 转换回四元数 [w, x, y, z]
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
    将四元数转换为欧拉角（用于调试）
    
    参数:
        quat: 四元数 [w, x, y, z]
        degrees: 如果为 True 返回角度，False 返回弧度
        
    返回:
        np.ndarray: 欧拉角 [roll, pitch, yaw]
    """
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x, y, z, w]
    euler = rot.as_euler('xyz', degrees=degrees)
    return euler


if __name__ == "__main__":
    """测试指定姿态生成器"""
    print("🧪 测试指定姿态生成器 (已修正方向，范围 -90°~+90°)\n")
    
    # 测试几个指定的姿态
    test_cases = [
        {"z_rotation": 0, "tilt_x": 0, "tilt_y": 0, "desc": "垂直向下，无旋转"},
        {"z_rotation": 30, "tilt_x": 0, "tilt_y": 0, "desc": "Z 轴顺时针旋转 30° (实际传入 -30° 给 scipy)"},
        {"z_rotation": -30, "tilt_x": 0, "tilt_y": 0, "desc": "Z 轴逆时针旋转 30° (实际传入 +30° 给 scipy)"},
        {"z_rotation": 90, "tilt_x": 0, "tilt_y": 0, "desc": "Z 轴顺时针最大旋转 90°"},
    ]
    
    print("生成指定的抓取姿态:")
    print("-" * 75)
    
    for i, test in enumerate(test_cases):
        try:
            quat = generate_grasp_pose(
                z_rotation=test["z_rotation"],
                tilt_x=test["tilt_x"],
                tilt_y=test["tilt_y"]
            )
            
            norm = np.linalg.norm(quat)
            euler = quaternion_to_euler(quat, degrees=True)
            
            print(f"姿态 {i+1}: {test['desc']}")
            print(f"  输入参数: Z={test['z_rotation']}°, X={test['tilt_x']}°, Y={test['tilt_y']}°")
            print(f"  四元数 [w,x,y,z]: {quat}")
            print(f"  欧拉角 [r,p,y]: [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]")
            print()
        except ValueError as e:
            print(f"❌ 捕获到预期错误: {e}")
            print()
    
    print("✅ 测试完成!")
