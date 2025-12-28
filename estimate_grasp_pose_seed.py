"""
使用豆包 Seed 1.6 Vision 模型估计抓取姿态
通过视觉分析图像中物体的姿态，输出三个抓取姿态参数
"""

import os
import re
from openai import OpenAI


def estimate_grasp_pose(image_path: str, object_name: str = "胡萝卜") -> tuple[float, float, float]:
    """
    使用豆包 Seed 1.6 Vision 模型分析图像并估计抓取姿态
    
    参数:
        image_path: 图像文件路径（本地路径或 URL）
        object_name: 要抓取的物体名称，默认为"胡萝卜"
        
    返回:
        tuple: (z_rotation, tilt_x, tilt_y) 三个姿态参数（单位：度）
               - z_rotation: 绕 Z 轴旋转角度，范围 -90 到 +90，正值为顺时针
               - tilt_x: 沿 X 轴倾斜角度，范围 -90 到 +90
               - tilt_y: 沿 Y 轴倾斜角度，范围 -90 到 +90
    """
    
    # 从环境变量中获取 API Key，如果没有则使用默认值
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        # 使用默认 API Key
        api_key = "fbf76dcd-f23f-4e53-bbef-a17ecaf9388a"
        print("⚠️ 未设置环境变量 ARK_API_KEY，使用默认 API Key")
    
    # 初始化 OpenAI 客户端（豆包使用 OpenAI 兼容接口）
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )
    
    # 构建 prompt
    prompt = f"""你现在是Franka机械臂，正视角如图所示。z轴垂直桌面向上，桌面为xy平面，x轴水平向左，y轴竖直向下。夹爪手心默认朝向桌面（z轴负方向），手指连线沿y轴方向。

现在需要你决策在抓取{object_name}时的夹爪姿态

请先完成两步操作：

1. 观察图像中**{object_name}的实际摆放姿态**：如果是物品的长轴位于xy平面，描述其长轴相对于x轴的偏转方向（顺时针还是逆时针你一定要反复确认，确保正确，这关乎到你最终的决策）及大致角度；
如果是物品的长轴垂直于xy平面，你需要分析手心向下抓取和手心侧向抓取哪种更优。

2. 结合抓取稳定性（如夹爪手指连线需垂直于物品长轴以避免打滑）与安全性（避障）要求，给出夹爪的三个姿态参数：
-- 绕z轴旋转角度（顺时针为+，范围-90~+90）
-- 沿x轴倾斜角度（向x正方向为+，范围-90~+90）
-- 沿y轴倾斜角度（向y正方向为+，范围-90~+90）

并说明每个参数的选择理由（需关联抓取物品实际姿态）。

**重要：请在回答的最后一行，以如下格式输出三个参数（仅数字，用逗号分隔）：**
GRASP_PARAMS: z_rotation, tilt_x, tilt_y

例如：GRASP_PARAMS: 30, 0, 0"""
    
    # 判断是本地文件还是 URL
    if image_path.startswith("http://") or image_path.startswith("https://"):
        image_url = image_path
    else:
        # 本地文件需要转换为 base64 编码的 data URL
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        # 根据文件扩展名确定 MIME 类型
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
        image_url = f"data:{mime_type};base64,{image_data}"
    
    # 调用 Seed 模型
    print(f"🤖 正在调用豆包 Seed 1.6 Vision 模型分析图像...")
    print(f"📷 图像路径: {image_path}")
    print(f"🎯 目标物体: {object_name}")
    
    try:
        response = client.chat.completions.create(
            model="doubao-seed-1-6-vision-250815",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        
        # 提取模型回复
        answer = response.choices[0].message.content
        print(f"\n📝 模型回复:\n{answer}\n")
        
        # 从回复中提取参数
        # 查找格式为 "GRASP_PARAMS: x, y, z" 的行
        match = re.search(r'GRASP_PARAMS:\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)', answer)
        
        if match:
            z_rotation = float(match.group(1))
            tilt_x = float(match.group(2))
            tilt_y = float(match.group(3))
            
            print(f"✅ 成功提取抓取姿态参数:")
            print(f"   Z 轴旋转: {z_rotation}°")
            print(f"   X 轴倾斜: {tilt_x}°")
            print(f"   Y 轴倾斜: {tilt_y}°")
            
            # 参数范围检查
            if not -90 <= z_rotation <= 90:
                print(f"⚠️ 警告: z_rotation={z_rotation}° 超出范围 [-90, 90]，将限制在范围内")
                z_rotation = max(-90, min(90, z_rotation))
            if not -90 <= tilt_x <= 90:
                print(f"⚠️ 警告: tilt_x={tilt_x}° 超出范围 [-90, 90]，将限制在范围内")
                tilt_x = max(-90, min(90, tilt_x))
            if not -90 <= tilt_y <= 90:
                print(f"⚠️ 警告: tilt_y={tilt_y}° 超出范围 [-90, 90]，将限制在范围内")
                tilt_y = max(-90, min(90, tilt_y))
            
            return (z_rotation, tilt_x, tilt_y)
        else:
            print("❌ 错误: 无法从模型回复中提取参数")
            print("   模型可能没有按照要求的格式输出参数")
            print("   使用默认值: (0, 0, 0)")
            return (0.0, 0.0, 0.0)
            
    except Exception as e:
        print(f"❌ 调用 Seed 模型时发生错误: {e}")
        print("   使用默认值: (0, 0, 0)")
        return (0.0, 0.0, 0.0)


if __name__ == "__main__":
    """测试 Seed 模型抓取姿态估计"""
    
    # 测试用例
    test_image = "test_image.jpg"  # 替换为实际图像路径
    test_object = "胡萝卜"
    
    print("🧪 测试豆包 Seed 1.6 Vision 抓取姿态估计\n")
    print(f"图像路径: {test_image}")
    print(f"目标物体: {test_object}\n")
    
    try:
        z_rot, tilt_x, tilt_y = estimate_grasp_pose(test_image, test_object)
        print(f"\n✅ 测试完成!")
        print(f"最终参数: Z={z_rot}°, X={tilt_x}°, Y={tilt_y}°")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
