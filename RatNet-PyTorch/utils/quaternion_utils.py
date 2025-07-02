import torch
import numpy as np

def quaternion_multiply(q1, q2):
    """
    四元数乘法
    
    Args:
        q1: (B, 4) 第一个四元数 [w, x, y, z]
        q2: (B, 4) 第二个四元数 [w, x, y, z]
    
    Returns:
        (B, 4) 乘法结果
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_conjugate(q):
    """
    四元数共轭
    
    Args:
        q: (B, 4) 四元数 [w, x, y, z]
    
    Returns:
        (B, 4) 共轭四元数 [w, -x, -y, -z]
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quaternion_inverse(q):
    """
    四元数逆
    
    Args:
        q: (B, 4) 四元数 [w, x, y, z]
    
    Returns:
        (B, 4) 逆四元数
    """
    return quaternion_conjugate(q) / torch.sum(q * q, dim=-1, keepdim=True)

def euler_to_quaternion(euler_angles):
    """
    欧拉角转四元数
    
    Args:
        euler_angles: (B, 3) 欧拉角 [roll, pitch, yaw] (弧度)
    
    Returns:
        (B, 4) 四元数 [w, x, y, z]
    """
    roll, pitch, yaw = euler_angles.unbind(-1)
    
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_to_euler(quaternion):
    """
    四元数转欧拉角
    
    Args:
        quaternion: (B, 4) 四元数 [w, x, y, z]
    
    Returns:
        (B, 3) 欧拉角 [roll, pitch, yaw] (弧度)
    """
    w, x, y, z = quaternion.unbind(-1)
    
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(np.pi / 2),
        torch.asin(sinp)
    )
    
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw], dim=-1)

def quaternion_error(q1, q2):
    """
    计算两个四元数之间的角度误差
    
    Args:
        q1: (B, 4) 第一个四元数
        q2: (B, 4) 第二个四元数
    
    Returns:
        (B,) 角度误差（弧度）
    """
    q_diff = quaternion_multiply(q1, quaternion_inverse(q2))
    return 2 * torch.acos(torch.clamp(torch.abs(q_diff[..., 0]), -1.0, 1.0)) 