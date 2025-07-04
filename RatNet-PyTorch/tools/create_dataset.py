#!/usr/bin/env python3
"""
生成雷达-相机校准数据集的脚本

此脚本加载nuScenes数据集，并对雷达与相机前向的校准数据进行随机去校准，
创建具有以下结构的训练数据集：
每个样本包含一个npz文件：
    - rgb_image: numpy数组，形状为 (150, 240, 3)
    - radar_detections: numpy数组，形状为 (number_of_detections, 4)，包含雷达坐标系中所有检测的位置向量 [x,y,z,1]
    - projections_groundtruth: 使用地面真值h_gt投影的雷达检测，以csr_matrix形式存储在ndarray中
    - projections_decalib: 使用h_gt和去校准投影的雷达检测，以csr_matrix形式存储在ndarray中
    - K: 内参相机校准矩阵K (numpy数组，形状为 (3,4))
    - H_gt: 从雷达到相机帧的齐次变换矩阵 (numpy数组，形状为 (4,4))
    - decalib: 相机和雷达旋转之间反转的变换，表示为四元数(idx 0-3)和平移(4-6) (numpy数组，形状为 (7,))
    - rgb_image_orig_dim : ndarray (2,1)，原始图像的高度([0])和宽度([1])（用于使用P进行变换）
"""

import os
import sys
import numpy as np
import cv2
import random
import argparse
import yaml
import math
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from tqdm import tqdm
from typing import List, Tuple, Dict

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='生成雷达-相机校准数据集')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--nuscenes_path', type=str, default='nuscenesData/v1.0-mini', help='nuScenes数据集路径')
    parser.add_argument('--output_path', type=str, default='data/calibration_dataset', help='输出数据集路径')
    parser.add_argument('--num_samples', type=int, default=-1, help='要处理的样本数，-1表示全部')
    parser.add_argument('--debug', action='store_true', help='是否保存调试图像')
    return parser.parse_args()


def load_config(config_path):
    """
    加载配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_decalib_transformation(num_transforms, rotation_std, translation_std, static_decalib=False):
    """
    创建随机去校准变换

    Args:
        num_transforms: 要生成的变换数量
        rotation_std: 旋转标准差（度）
        translation_std: 平移标准差（米）
        static_decalib: 是否使用相同的去校准

    Returns:
        decalibs: 去校准变换列表，每个变换是一个形状为(7,)的数组，
                 其中前4个元素是四元数，后3个元素是平移向量
    """
    if static_decalib:
        # 为所有变换使用相同的去校准
        roll = np.random.normal(0, rotation_std)
        pitch = np.random.normal(0, rotation_std)
        yaw = np.random.normal(0, rotation_std)
        
        # 转换为弧度
        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        
        # 创建四元数
        quat = Quaternion(axis=[1, 0, 0], angle=roll_rad) * \
               Quaternion(axis=[0, 1, 0], angle=pitch_rad) * \
               Quaternion(axis=[0, 0, 1], angle=yaw_rad)
        
        # 创建随机平移
        tx = np.random.normal(0, translation_std)
        ty = np.random.normal(0, translation_std)
        tz = np.random.normal(0, translation_std)
        
        # 组合四元数和平移
        decalib = np.array([quat.w, quat.x, quat.y, quat.z, tx, ty, tz])
        
        # 为所有变换复制相同的去校准
        decalibs = [decalib] * num_transforms
    else:
        # 为每个变换生成不同的去校准
        decalibs = []
        
        for _ in range(num_transforms):
            # 生成随机旋转角度（度）
            roll = np.random.normal(0, rotation_std)
            pitch = np.random.normal(0, rotation_std)
            yaw = np.random.normal(0, rotation_std)
            
            # 转换为弧度
            roll_rad = np.deg2rad(roll)
            pitch_rad = np.deg2rad(pitch)
            yaw_rad = np.deg2rad(yaw)
            
            # 创建四元数
            quat = Quaternion(axis=[1, 0, 0], angle=roll_rad) * \
                   Quaternion(axis=[0, 1, 0], angle=pitch_rad) * \
                   Quaternion(axis=[0, 0, 1], angle=yaw_rad)
            
            # 创建随机平移
            tx = np.random.normal(0, translation_std)
            ty = np.random.normal(0, translation_std)
            tz = np.random.normal(0, translation_std)
            
            # 组合四元数和平移
            decalib = np.array([quat.w, quat.x, quat.y, quat.z, tx, ty, tz])
            decalibs.append(decalib)
    
    return decalibs


def load_keyframe_rad_cam_data(nusc: NuScenes) -> Tuple[List[str], List[str], List[str]]:
    """
    获取所有CAM_FRONT和RADAR_FRONT样本的tokens，这些样本对应于关键帧（is_key_frame = True）
    
    Args:
        nusc: NuScenes实例
    
    Returns:
        cam_sd_tokens: 相机样本数据tokens列表
        rad_sd_tokens: 雷达样本数据tokens列表
        sample_names: 样本名称列表
    """
    # 存储相机和雷达样本数据tokens的列表
    cam_sd_tokens = []
    rad_sd_tokens = []
    sample_names = []
    
    for scene_rec in nusc.scene:
        print(f'加载场景 {scene_rec["name"]} 的样本...', end='')
        
        # 获取场景的第一个样本
        start_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        
        # 获取第一个样本的相机和雷达数据
        cam_front_sd_rec = nusc.get('sample_data', start_sample_rec['data']['CAM_FRONT'])
        rad_front_sd_rec = nusc.get('sample_data', start_sample_rec['data']['RADAR_FRONT'])
        
        # 获取当前样本数据记录
        cur_cam_front_sd_rec = cam_front_sd_rec
        cur_rad_front_sd_rec = rad_front_sd_rec
        
        # 提取样本名称（从文件名中）
        sample_name = cur_cam_front_sd_rec["filename"].replace('samples/CAM_FRONT/', '').replace('.jpg', '')
        
        # 添加第一个样本到列表中
        sample_names.append(sample_name)
        cam_sd_tokens.append(cur_cam_front_sd_rec['token'])
        rad_sd_tokens.append(cur_rad_front_sd_rec['token'])
        
        # 添加所有关键帧相机样本到列表中
        while cur_cam_front_sd_rec['next'] != '':
            cur_cam_front_sd_rec = nusc.get('sample_data', cur_cam_front_sd_rec['next'])
            sample_name = cur_cam_front_sd_rec["filename"].replace('samples/CAM_FRONT/', '').replace('.jpg', '')
            if cur_cam_front_sd_rec['is_key_frame']:
                sample_names.append(sample_name)
                cam_sd_tokens.append(cur_cam_front_sd_rec['token'])
        
        # 添加所有关键帧雷达样本到列表中
        while cur_rad_front_sd_rec['next'] != '':
            cur_rad_front_sd_rec = nusc.get('sample_data', cur_rad_front_sd_rec['next'])
            if cur_rad_front_sd_rec['is_key_frame']:
                rad_sd_tokens.append(cur_rad_front_sd_rec['token'])
        
        print("完成！")
    
    # 确保所有列表长度相同
    assert(len(cam_sd_tokens) == len(rad_sd_tokens) == len(sample_names))
    
    return cam_sd_tokens, rad_sd_tokens, sample_names


def tokens_to_data_pairs(nusc: NuScenes, cam_sd_tokens: List[str], rad_sd_tokens: List[str], 
                         image_width: int, image_height: int) -> List:
    """
    将相机和雷达样本数据tokens转换为图像和雷达点云对
    
    Args:
        nusc: NuScenes实例
        cam_sd_tokens: 相机样本数据tokens列表
        rad_sd_tokens: 雷达样本数据tokens列表
        image_width: 目标图像宽度
        image_height: 目标图像高度
        
    Returns:
        image_radar_pairs: 图像和雷达点云对的列表
    """
    rgb_images_list = []
    for i in range(len(cam_sd_tokens)):
        cam_sd_path = nusc.get_sample_data_path(cam_sd_tokens[i])
        if not os.path.isfile(cam_sd_path):
            continue
        
        # 读取图像并调整大小
        img = cv2.imread(cam_sd_path)
        img = cv2.resize(img, (image_width, image_height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_images_list.append(img_rgb)
    
    radar_pcl_list = []
    for i in range(len(rad_sd_tokens)):
        rad_sd_path = nusc.get_sample_data_path(rad_sd_tokens[i])
        if not os.path.isfile(rad_sd_path):
            continue
        
        # 加载雷达点云
        radar_pcl = RadarPointCloud.from_file(rad_sd_path)
        
        # nuScenes RadarPointCloud形状为 (18, num_points)
        # 我们需要形状为 (num_points, 4)
        radar_pcl.points = radar_pcl.points.transpose()
        radar_pcl.points = radar_pcl.points[:, :3]
        radar_pcl.points = np.hstack((radar_pcl.points, np.ones((radar_pcl.points.shape[0], 1), dtype=radar_pcl.points.dtype)))
        
        radar_pcl_list.append(radar_pcl)
    
    # 确保图像和雷达点云列表长度相同
    assert(len(rgb_images_list) == len(radar_pcl_list))
    
    # 创建图像和雷达点云对
    image_radar_pairs = list(zip(rgb_images_list, radar_pcl_list))
    
    del rgb_images_list
    del radar_pcl_list
    
    return image_radar_pairs


def get_rad_to_cam(nusc: NuScenes, cam_sd_token: str, rad_sd_token: str):
    """
    获取从radar_front到camera_front的外部校准矩阵
    
    Args:
        nusc: NuScenes实例
        cam_sd_token: 特定camera_front样本数据的token
        rad_sd_token: 特定radar_front样本数据的token
        
    Returns:
        rad_to_cam: 从雷达到相机的齐次变换矩阵，形状为 (4, 4)
    """
    # 获取相机校准传感器记录
    cam_cs_token = nusc.get('sample_data', cam_sd_token)["calibrated_sensor_token"]
    cam_cs_rec = nusc.get('calibrated_sensor', cam_cs_token)
    
    # 获取雷达校准传感器记录
    rad_cs_token = nusc.get('sample_data', rad_sd_token)["calibrated_sensor_token"]
    rad_cs_rec = nusc.get('calibrated_sensor', rad_cs_token)
    
    # 根据nuScenes脚本（如scripts/export_kitti.py）中处理变换的方式
    rad_to_ego = transform_matrix(rad_cs_rec['translation'], Quaternion(rad_cs_rec['rotation']), inverse=False)
    ego_to_cam = transform_matrix(cam_cs_rec['translation'], Quaternion(cam_cs_rec['rotation']), inverse=True)
    rad_to_cam = np.dot(ego_to_cam, rad_to_ego)
    
    return rad_to_cam


def comp_uv_depth(K, h_gt, decalib, point):
    """
    计算像素坐标和雷达深度
    
    Args:
        K: 相机内参矩阵 (3x3或3x4) - 包含相机的内部参数（焦距、主点等）
        h_gt: 地面真值变换矩阵 (4x4) - 从雷达到相机坐标系的变换矩阵
        decalib: 去校准变换矩阵 (4x4) - 模拟校准误差的额外变换
        point: 3D点坐标 [x,y,z,1] - 雷达坐标系中的点（齐次坐标）
        
    Returns:
        像素坐标和深度 [u, v, depth] - 投影后的像素坐标和深度值
    """
    # 使用投影公式: z * (u, v, 1)^T = K * H * x
    # 其中:
    # x 是雷达坐标系中的3D点 [X, Y, Z, 1]^T
    # H 是从雷达坐标系到相机坐标系的变换矩阵 (decalib * h_gt)
    # K 是相机内参矩阵
    # (u, v) 是投影后的像素坐标
    # z 是点在相机坐标系中的深度
    # 步骤1: 将点从雷达坐标系转换到相机坐标系
    # 组合去校准和地面真值变换矩阵：decalib * h_gt
    transform = np.matmul(decalib, h_gt)
    # 应用变换矩阵到点坐标：[X_cam, Y_cam, Z_cam, 1]^T = transform * [X_radar, Y_radar, Z_radar, 1]^T
    point_cam = np.matmul(transform, point.transpose())
    
    # 步骤2: 应用相机内参矩阵进行透视投影
    if K.shape[0] == 3 and K.shape[1] == 3:
        # 如果K是3x3，需要取点的前三个坐标进行投影
        # [z*u, z*v, z]^T = K * [X_cam, Y_cam, Z_cam]^T
        point_proj = np.matmul(K, point_cam[:3])
    else:
        # 如果K已经是3x4，直接进行投影
        # [z*u, z*v, z]^T = K * [X_cam, Y_cam, Z_cam, 1]^T
        point_proj = np.matmul(K, point_cam)
    
    # 步骤3: 归一化坐标 - 除以深度值z得到最终像素坐标
    if point_proj[2] != 0:  # 确保深度值不为零，避免除零错误
        # 计算像素坐标 u = (z*u)/z, v = (z*v)/z
        return [point_proj[0] / point_proj[2], point_proj[1] / point_proj[2], point_proj[2]]
    else:
        # 如果深度为零，则返回None表示投影无效
        return None


def valid_pixel_coordinates(u, v, image_height, image_width):
    """
    检查像素坐标是否有效
    
    Args:
        u: 水平像素坐标
        v: 垂直像素坐标
        image_height: 图像高度
        image_width: 图像宽度
        
    Returns:
        布尔值，表示坐标是否有效
    """
    return (u >= 0 and v >= 0 and v < image_height and u < image_width)


def create_and_store_samples(image_radar_pairs, sample_names, rad_to_cam_calibration_matrices, 
                             cam_intrinsics, output_path, config, debug=False):
    """
    为每对图像和雷达数据创建训练样本，并将雷达检测投影到RGB图像上
    
    Args:
        image_radar_pairs: 图像和雷达点云对的列表
        sample_names: 样本名称列表
        rad_to_cam_calibration_matrices: 雷达到相机变换矩阵列表
        cam_intrinsics: 相机内参矩阵列表
        output_path: 输出路径
        config: 配置字典
        debug: 是否保存调试图像
    """
    image_width = config['dataset']['image_width']
    image_height = config['dataset']['image_height']
    min_points = config['dataset']['min_points']
    rotation_std = config['preprocessing']['rotation_std']
    translation_std = config['preprocessing']['translation_std']
    static_decalib = config['preprocessing']['static_decalib']
    
    print(f"图像和雷达对的数量: {len(image_radar_pairs)}")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    if debug:
        debug_dir = os.path.join(output_path, 'debug_images')
        os.makedirs(debug_dir, exist_ok=True)
    
    # 获取去校准变换
    if static_decalib:
        decalib_list = create_decalib_transformation(1, rotation_std, translation_std, static_decalib=True)
        decalib = decalib_list[0]
        print(f"使用静态去校准: {decalib}")
    
    # 处理每对图像和雷达数据
    for i, (img, radar_pcl) in tqdm(enumerate(image_radar_pairs), total=len(image_radar_pairs), desc="处理样本"):
        sample_name = sample_names[i]
        h_gt = rad_to_cam_calibration_matrices[i]
        K = cam_intrinsics[i]
        
        # 如果不使用静态去校准，则为每个样本创建新的去校准
        if not static_decalib:
            decalib_list = create_decalib_transformation(1, rotation_std, translation_std)
            decalib = decalib_list[0]
        
        # 为去校准创建变换矩阵
        w, x, y, z = decalib[0:4]
        tx, ty, tz = decalib[4:7]
        
        quat = Quaternion(w, x, y, z)
        decalib_transform = np.eye(4)
        decalib_transform[:3, :3] = quat.rotation_matrix
        decalib_transform[:3, 3] = [tx, ty, tz]
        
        # 初始化稀疏矩阵的数据
        # 为每一行创建列表存储信息
        rows_data = [[] for _ in range(image_height)]
        rows_indices = [[] for _ in range(image_height)]
        rows_decalib_data = [[] for _ in range(image_height)]
        rows_decalib_indices = [[] for _ in range(image_height)]
        
        # 遍历雷达点
        radar_points = radar_pcl.points
        
        # 跳过具有太少点的样本
        if radar_points.shape[0] < min_points:
            continue
        
        # 投影到图像上
        for point in radar_points:
            # 地面真值投影
            proj_result = comp_uv_depth(K, h_gt, np.eye(4), point)
            
            if proj_result is not None:
                u, v, depth = proj_result
                u = int(round(u))
                v = int(round(v))
                
                if valid_pixel_coordinates(u, v, image_height, image_width):
                    # 将点添加到对应行的列表中
                    rows_indices[v].append(u)
                    rows_data[v].append(depth)
                    
                    # 去校准投影
                    proj_decalib_result = comp_uv_depth(K, h_gt, decalib_transform, point)
                    
                    if proj_decalib_result is not None:
                        u_decalib, v_decalib, depth_decalib = proj_decalib_result
                        u_decalib = int(round(u_decalib))
                        v_decalib = int(round(v_decalib))
                        
                        if valid_pixel_coordinates(u_decalib, v_decalib, image_height, image_width):
                            # 将去校准点添加到对应行的列表中
                            rows_decalib_indices[v_decalib].append(u_decalib)
                            rows_decalib_data[v_decalib].append(depth_decalib)
        
        # 构建CSR矩阵
        # 将行数据合并成一维数组
        csr_data = []
        csr_indices = []
        csr_indptr = [0]
        
        csr_decalib_data = []
        csr_decalib_indices = []
        csr_decalib_indptr = [0]
        
        # 构建地面真值投影的CSR矩阵
        for row_data, row_indices in zip(rows_data, rows_indices):
            csr_data.extend(row_data)
            csr_indices.extend(row_indices)
            csr_indptr.append(len(csr_data))
        
        # 构建去校准投影的CSR矩阵
        for row_data, row_indices in zip(rows_decalib_data, rows_decalib_indices):
            csr_decalib_data.extend(row_data)
            csr_decalib_indices.extend(row_indices)
            csr_decalib_indptr.append(len(csr_decalib_data))
        
        # 创建稀疏矩阵
        projections = csr_matrix((csr_data, csr_indices, csr_indptr), shape=(image_height, image_width))
        projections_decalib = csr_matrix((csr_decalib_data, csr_decalib_indices, csr_decalib_indptr), shape=(image_height, image_width))
        
        # 保存样本
        store_sample(
            sample_name=sample_name,
            image=img,
            radar_detections=radar_points,
            projection=projections,
            projection_decalib=projections_decalib,
            decalib=decalib,
            h_gt=h_gt,
            K=K,
            output_path=output_path,
            debug=debug
        )


def store_sample(sample_name, image, radar_detections, projection, projection_decalib, 
                decalib, h_gt, K, output_path, debug=False):
    """
    将样本保存到磁盘
    
    Args:
        sample_name: 样本名称
        image: RGB图像
        radar_detections: 雷达检测点
        projection: 地面真值投影
        projection_decalib: 去校准投影
        decalib: 去校准变换
        h_gt: 地面真值变换矩阵
        K: 相机内参矩阵
        output_path: 输出路径
        debug: 是否保存调试图像
    """
    # 原始图像尺寸
    rgb_image_orig_dim = np.array([image.shape[0], image.shape[1]])
    
    # 保存样本到npz文件
    sample_path = os.path.join(output_path, f"{sample_name}.npz")
    np.savez_compressed(
        sample_path,
        rgb_image=image,
        radar_detections=radar_detections,
        projections_groundtruth=projection,
        projections_decalib=projection_decalib,
        decalib=decalib,
        H_gt=h_gt,
        K=K,
        rgb_image_orig_dim=rgb_image_orig_dim
    )
    
    # 保存调试图像
    if debug:
        debug_image = image.copy()
        
        # 绘制地面真值投影
        for i, j in zip(*projection.nonzero()):
            cv2.circle(debug_image, (j, i), 2, (0, 255, 0), -1)  # 绿色表示地面真值
        
        # 绘制去校准投影
        for i, j in zip(*projection_decalib.nonzero()):
            cv2.circle(debug_image, (j, i), 2, (0, 0, 255), -1)  # 红色表示去校准
        
        # 保存调试图像
        debug_path = os.path.join(output_path, 'debug_images', f"{sample_name}_debug.jpg")
        cv2.imwrite(debug_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))


def create_dataset_splits(output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    创建训练、验证和测试集分割
    
    Args:
        output_path: 数据集输出路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    # 确保比例总和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例总和必须为1"
    
    # 获取所有npz文件
    all_files = [f for f in os.listdir(output_path) if f.endswith('.npz')]
    
    # 打乱文件顺序
    random.seed(42)  # 设置种子以便可重复
    random.shuffle(all_files)
    
    # 计算分割索引
    n_samples = len(all_files)
    train_idx = int(n_samples * train_ratio)
    val_idx = int(n_samples * (train_ratio + val_ratio))
    
    # 分割数据集
    train_files = all_files[:train_idx]
    val_files = all_files[train_idx:val_idx]
    test_files = all_files[val_idx:]
    
    # 保存分割列表
    with open(os.path.join(output_path, 'train_files.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_files))
    
    with open(os.path.join(output_path, 'val_files.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_files))
    
    with open(os.path.join(output_path, 'test_files.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_files))
    
    print(f"数据集分割完成: 训练集 {len(train_files)} 样本, 验证集 {len(val_files)} 样本, 测试集 {len(test_files)} 样本")


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置参数
    nuscenes_path = args.nuscenes_path
    output_path = args.output_path
    debug = args.debug
    num_samples = args.num_samples
    
    # 图像尺寸
    image_width = config['dataset']['image_width']
    image_height = config['dataset']['image_height']
    
    # 初始化nuScenes
    print(f"加载nuScenes数据集从 {nuscenes_path}")
    nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_path, verbose=True)
    
    # 获取所有关键帧的相机和雷达数据tokens
    print("获取关键帧相机和雷达数据tokens...")
    cam_sd_tokens, rad_sd_tokens, sample_names = load_keyframe_rad_cam_data(nusc)
    
    # 如果指定了样本数量，则截取相应的数量
    if num_samples > 0 and num_samples < len(cam_sd_tokens):
        cam_sd_tokens = cam_sd_tokens[:num_samples]
        rad_sd_tokens = rad_sd_tokens[:num_samples]
        sample_names = sample_names[:num_samples]
    
    # 加载图像和雷达数据
    print("加载图像和雷达数据...")
    image_radar_pairs = tokens_to_data_pairs(nusc, cam_sd_tokens, rad_sd_tokens, image_width, image_height)
    
    # 获取相机内参和雷达到相机的变换矩阵
    print("计算校准矩阵...")
    rad_to_cam_matrices = []
    cam_intrinsics = []
    
    for i in range(len(cam_sd_tokens)):
        # 获取雷达到相机的变换矩阵
        rad_to_cam = get_rad_to_cam(nusc, cam_sd_tokens[i], rad_sd_tokens[i])
        rad_to_cam_matrices.append(rad_to_cam)
        
        # 获取相机内参
        cam_cs_token = nusc.get('sample_data', cam_sd_tokens[i])["calibrated_sensor_token"]
        cam_cs_rec = nusc.get('calibrated_sensor', cam_cs_token)
        K = np.array(cam_cs_rec['camera_intrinsic'])
        cam_intrinsics.append(K)
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 创建和保存样本
    print("创建和保存样本...")
    create_and_store_samples(
        image_radar_pairs=image_radar_pairs,
        sample_names=sample_names,
        rad_to_cam_calibration_matrices=rad_to_cam_matrices,
        cam_intrinsics=cam_intrinsics,
        output_path=output_path,
        config=config,
        debug=debug
    )
    
    # 创建数据集分割
    print("创建数据集分割...")
    create_dataset_splits(
        output_path=output_path,
        train_ratio=config['dataset']['train_split'],
        val_ratio=config['dataset']['val_split'],
        test_ratio=config['dataset']['test_split']
    )
    
    print(f"数据集创建完成! 数据保存在 {output_path}")


if __name__ == "__main__":
    main()
