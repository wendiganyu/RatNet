import torch
import numpy as np


def normalize_quaternions(quaternions):
    """
    归一化四元数
    
    Args:
        quaternions: 形状为 [batch_size, 4] 的四元数张量
    
    Returns:
        归一化后的四元数
    """
    norm = torch.norm(quaternions, dim=1, keepdim=True)
    return quaternions / (norm + 1e-8)  # 添加小值避免除零错误


def conjugate_quaternions(quaternions):
    """
    计算四元数的共轭
    
    Args:
        quaternions: 形状为 [batch_size, 4] 的四元数张量
    
    Returns:
        四元数的共轭
    """
    # 四元数格式: [w, x, y, z]
    # 共轭: [w, -x, -y, -z]
    conj = quaternions.clone()
    conj[:, 1:] = -conj[:, 1:]
    return conj


def multiply_quaternions(q1, q2):
    """
    四元数乘法
    
    Args:
        q1: 形状为 [batch_size, 4] 的四元数张量
        q2: 形状为 [batch_size, 4] 的四元数张量
    
    Returns:
        q1 和 q2 相乘的结果
    """
    # 四元数格式: [w, x, y, z]
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    return torch.stack([w, x, y, z], dim=1)


def rotation_matrix_from_quaternion(quaternion):
    """
    从四元数计算旋转矩阵
    
    Args:
        quaternion: 形状为 [batch_size, 4] 的四元数张量 (w, x, y, z)
    
    Returns:
        形状为 [batch_size, 3, 3] 的旋转矩阵
    """
    # 确保四元数已归一化
    quaternion = normalize_quaternions(quaternion)
    
    batch_size = quaternion.size(0)
    
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    
    matrix = torch.zeros((batch_size, 3, 3), device=quaternion.device)
    
    matrix[:, 0, 0] = 1.0 - (tyy + tzz)
    matrix[:, 0, 1] = txy - twz
    matrix[:, 0, 2] = txz + twy
    matrix[:, 1, 0] = txy + twz
    matrix[:, 1, 1] = 1.0 - (txx + tzz)
    matrix[:, 1, 2] = tyz - twx
    matrix[:, 2, 0] = txz - twy
    matrix[:, 2, 1] = tyz + twx
    matrix[:, 2, 2] = 1.0 - (txx + tyy)
    
    return matrix


def transform_from_quaternion_and_translation(quaternion, translation):
    """
    从四元数和平移向量创建变换矩阵
    
    Args:
        quaternion: 形状为 [batch_size, 4] 的四元数张量
        translation: 形状为 [batch_size, 3] 的平移向量张量
    
    Returns:
        形状为 [batch_size, 4, 4] 的变换矩阵
    """
    batch_size = quaternion.size(0)
    
    # 获取旋转矩阵
    rotation_matrix = rotation_matrix_from_quaternion(quaternion)
    
    # 创建变换矩阵
    transform = torch.zeros((batch_size, 4, 4), device=quaternion.device)
    transform[:, :3, :3] = rotation_matrix
    transform[:, :3, 3] = translation.squeeze(-1) if translation.dim() > 2 else translation
    transform[:, 3, 3] = 1.0
    
    return transform


def quaternion_angular_error(q1, q2):
    """
    计算两个四元数之间的角度误差（以度为单位）
    
    Args:
        q1: 形状为 [batch_size, 4] 的四元数张量
        q2: 形状为 [batch_size, 4] 的四元数张量
    
    Returns:
        角度误差（度）
    """
    # 归一化四元数
    q1 = normalize_quaternions(q1)
    q2 = normalize_quaternions(q2)
    
    # 计算内积
    dot_product = torch.sum(q1 * q2, dim=1)
    
    # 确保dot_product在[-1, 1]范围内
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # 计算角度（弧度）
    angle_rad = 2.0 * torch.acos(torch.abs(dot_product))
    
    # 转换为度
    angle_deg = angle_rad * (180.0 / np.pi)
    
    return angle_deg


def euler_from_quaternion(quaternion):
    """
    从四元数计算欧拉角（以度为单位）
    
    Args:
        quaternion: 形状为 [batch_size, 4] 的四元数张量 (w, x, y, z)
    
    Returns:
        形状为 [batch_size, 3] 的欧拉角张量 (roll, pitch, yaw)
    """
    # 确保四元数已归一化
    quaternion = normalize_quaternions(quaternion)
    
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    # 计算欧拉角
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # 使用条件操作处理奇异点
    pitch = torch.where(
        torch.abs(sinp) >= 1.0,
        torch.sign(sinp) * (torch.pi / 2.0),  # use 90 degrees if out of range
        torch.asin(sinp)
    )
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    # 转换为度
    roll_deg = roll * (180.0 / np.pi)
    pitch_deg = pitch * (180.0 / np.pi)
    yaw_deg = yaw * (180.0 / np.pi)
    
    return torch.stack([roll_deg, pitch_deg, yaw_deg], dim=1)


def quaternion_from_euler(euler_angles):
    """
    从欧拉角（以度为单位）计算四元数
    
    Args:
        euler_angles: 形状为 [batch_size, 3] 的欧拉角张量 (roll, pitch, yaw)，单位为度
    
    Returns:
        形状为 [batch_size, 4] 的四元数张量 (w, x, y, z)
    """
    # 转换为弧度
    roll = euler_angles[:, 0] * (np.pi / 180.0)
    pitch = euler_angles[:, 1] * (np.pi / 180.0)
    yaw = euler_angles[:, 2] * (np.pi / 180.0)
    
    # 预计算角度的一半
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    # 四元数计算
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return torch.stack([w, x, y, z], dim=1) 