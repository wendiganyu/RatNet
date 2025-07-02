import os
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
import torchvision.transforms as T

class RadarCameraDataset(Dataset):
    """RadNet数据集类，用于加载和预处理雷达和相机数据"""
    
    def __init__(self, 
                 nusc: NuScenes,
                 split: str = 'train',
                 image_size: tuple = (150, 240),
                 transform=None):
        """
        初始化数据集
        
        Args:
            nusc: NuScenes数据集实例
            split: 数据集分割（'train' 或 'val'）
            image_size: 图像大小 (H, W)
            transform: 数据增强转换
        """
        self.nusc = nusc
        self.split = split
        self.image_size = image_size
        self.transform = transform
        
        # 获取场景列表
        self.scene_tokens = [s['token'] for s in nusc.scene]
        
        # 基础图像转换
        self.image_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.scene_tokens)
    
    def __getitem__(self, idx):
        scene_token = self.scene_tokens[idx]
        scene = self.nusc.get('scene', scene_token)
        
        # 获取相机和雷达数据
        sample = self.nusc.get('sample', scene['first_sample_token'])
        cam_token = sample['data']['CAM_FRONT']
        radar_token = sample['data']['RADAR_FRONT']
        
        # 加载相机图像
        cam_data = self.nusc.get('sample_data', cam_token)
        image_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        
        # 加载雷达点云
        radar_data = self.nusc.get('sample_data', radar_token)
        radar_path = os.path.join(self.nusc.dataroot, radar_data['filename'])
        radar_pc = LidarPointCloud.from_file(radar_path)
        radar_points = radar_pc.points.T  # (N, 4) [x, y, z, intensity]
        
        # 获取标定参数
        calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        k_matrix = torch.tensor(calib['camera_intrinsic'], dtype=torch.float32)
        
        # 生成随机解标定变换
        decalib_quat = self._generate_random_quaternion()
        decalib_trans = self._generate_random_translation()
        
        # 转换为张量
        radar_points = torch.tensor(radar_points, dtype=torch.float32)
        decalib_quat = torch.tensor(decalib_quat, dtype=torch.float32)
        decalib_trans = torch.tensor(decalib_trans, dtype=torch.float32)
        
        return {
            'image': image,  # (3, H, W)
            'radar_points': radar_points,  # (N, 4)
            'k_matrix': k_matrix,  # (3, 3)
            'decalib_quat': decalib_quat,  # (4,)
            'decalib_trans': decalib_trans  # (3,)
        }
    
    def _generate_random_quaternion(self):
        """生成随机四元数作为解标定旋转"""
        # 使用欧拉角生成四元数
        yaw = np.random.uniform(-np.pi/6, np.pi/6)  # ±30度
        pitch = np.random.uniform(-np.pi/18, np.pi/18)  # ±10度
        roll = np.random.uniform(-np.pi/18, np.pi/18)  # ±10度
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        
        return np.array([w, x, y, z], dtype=np.float32)
    
    def _generate_random_translation(self):
        """生成随机平移向量作为解标定平移"""
        # 在±0.1米范围内生成随机平移
        return np.random.uniform(-0.1, 0.1, size=3).astype(np.float32) 