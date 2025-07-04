import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse


class RadarCameraCalibDataset(Dataset):
    """
    雷达-相机校准数据集类
    """
    
    def __init__(self, dataset_path, transform=None, split='train'):
        """
        初始化数据集
        
        Args:
            dataset_path: 数据集根路径
            transform: 数据增强转换
            split: 数据集分割 ('train', 'val', 或 'test')
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.split = split
        
        # 加载样本列表
        self.samples = self._load_sample_list()
    
    def _load_sample_list(self):
        """
        加载指定分割的样本列表
        
        Returns:
            样本文件路径列表
        """
        # 根据分割加载相应的文件列表
        split_file = os.path.join(self.dataset_path, f"{self.split}_files.txt")
        
        if os.path.exists(split_file):
            with open(split_file, 'r', encoding='utf-8') as f:
                sample_list = [line.strip() for line in f.readlines()]
            return sample_list
        else:
            # 如果没有分割文件，则扫描整个目录
            all_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.npz')]
            
            # 根据split进行随机分割
            np.random.seed(42)  # 固定随机种子
            np.random.shuffle(all_files)
            
            # 根据不同的分割比例划分数据集
            if self.split == 'train':
                return all_files[:int(len(all_files) * 0.8)]
            elif self.split == 'val':
                return all_files[int(len(all_files) * 0.8):int(len(all_files) * 0.9)]
            else:  # test
                return all_files[int(len(all_files) * 0.9):]
    
    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            样本数据字典
        """
        sample_path = os.path.join(self.dataset_path, self.samples[idx])
        
        with np.load(sample_path, encoding='latin1', allow_pickle=True) as data:
            rgb_image = data["rgb_image"]
            
            # 获取投影数据并转换为稠密矩阵
            projections_decalib = self._get_projections(data, "projections_decalib")
            projections_groundtruth = self._get_projections(data, "projections_groundtruth")
            
            # 标签是逆去校准的四元数和平移向量
            decalib = data["decalib"]
            quaternion = decalib[:4]  # 前4个是四元数
            translation = decalib[4:7]  # 接下来的3个是平移
            
            # 相机内参
            K = data["K"]
            
            # 变换矩阵
            H_gt = data["H_gt"]
            
            # 雷达检测
            radar_detections = data["radar_detections"]
            
            # 原始图像尺寸
            rgb_image_orig_dim = data["rgb_image_orig_dim"]
        
        # 转换为张量
        rgb_image = torch.FloatTensor(rgb_image) / 255.0  # 归一化到 [0, 1]
        rgb_image = rgb_image.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        projections_decalib = torch.FloatTensor(projections_decalib)
        projections_groundtruth = torch.FloatTensor(projections_groundtruth)
        
        quaternion = torch.FloatTensor(quaternion)
        translation = torch.FloatTensor(translation)
        
        K = torch.FloatTensor(K)
        H_gt = torch.FloatTensor(H_gt)
        
        radar_detections = torch.FloatTensor(radar_detections)
        
        # 应用数据增强
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        # 返回样本数据字典
        return {
            'rgb_image': rgb_image,
            'radar_input': projections_decalib,
            'projections_groundtruth': projections_groundtruth,
            'quaternion': quaternion,
            'translation': translation,
            'K': K,
            'H_gt': H_gt,
            'radar_detections': radar_detections,
            'rgb_image_orig_dim': rgb_image_orig_dim
        }
    
    def _get_projections(self, data, key):
        """
        从稀疏CSR矩阵获取稠密投影矩阵
        
        Args:
            data: 数据字典
            key: 投影数据的键名
            
        Returns:
            投影的稠密矩阵
        """
        # 获取CSR矩阵
        csr = np.expand_dims(data[key], axis=-1)[0]
        
        # 转换为稠密矩阵并添加通道维度
        dense_matrix = sparse.csr_matrix.todense(csr)
        return np.expand_dims(dense_matrix, axis=-1)


def collate_fn(batch):
    """
    自定义整理函数，处理批次数据
    
    Args:
        batch: 样本列表
        
    Returns:
        批次数据字典
    """
    rgb_images = torch.stack([item['rgb_image'] for item in batch])
    radar_inputs = torch.stack([item['radar_input'] for item in batch])
    projections_groundtruth = torch.stack([item['projections_groundtruth'] for item in batch])
    quaternions = torch.stack([item['quaternion'] for item in batch])
    translations = torch.stack([item['translation'] for item in batch])
    K_matrices = torch.stack([item['K'] for item in batch])
    H_gt_matrices = torch.stack([item['H_gt'] for item in batch])
    
    # 雷达检测可能形状不一，需要特殊处理
    radar_detections = [item['radar_detections'] for item in batch]
    
    # 原始图像尺寸
    rgb_image_orig_dims = [item['rgb_image_orig_dim'] for item in batch]
    
    return {
        'rgb_image': rgb_images,
        'radar_input': radar_inputs,
        'projections_groundtruth': projections_groundtruth,
        'quaternion': quaternions,
        'translation': translations,
        'K': K_matrices,
        'H_gt': H_gt_matrices,
        'radar_detections': radar_detections,
        'rgb_image_orig_dim': rgb_image_orig_dims
    } 