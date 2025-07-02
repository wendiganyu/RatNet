import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MLPConv(nn.Module):
    """多层感知机卷积层"""
    def __init__(self, out_channels, kernel_size=5):
        super(MLPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class SpatialTransformer3D(nn.Module):
    """3D空间变换层"""
    def __init__(self):
        super(SpatialTransformer3D, self).__init__()
    
    def forward(self, depth_map, transform_matrix, k_matrix):
        """
        应用3D空间变换
        
        Args:
            depth_map: 深度图 (B, H, W)
            transform_matrix: 变换矩阵 (B, 4, 4)
            k_matrix: 相机内参矩阵 (B, 3, 3)
            
        Returns:
            transformed_depth: 变换后的深度图 (B, H, W)
            transformed_points: 变换后的点云 (B, N, 3)
        """
        batch_size, height, width = depth_map.shape
        
        # 1. 生成像素网格
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
        pixels = torch.stack([x, y, torch.ones_like(x)], dim=2)  # (H, W, 3)
        pixels = pixels.to(depth_map.device)
        
        # 2. 反投影到3D空间
        k_inv = torch.inverse(k_matrix)  # (B, 3, 3)
        rays = torch.matmul(k_inv[:, None, None], pixels[None, ..., :, None])  # (B, H, W, 3, 1)
        points_3d = rays.squeeze(-1) * depth_map[..., None]  # (B, H, W, 3)
        
        # 3. 应用变换
        points_3d_homo = torch.cat([points_3d, torch.ones_like(points_3d[..., :1])], dim=-1)  # (B, H, W, 4)
        transformed_points = torch.matmul(transform_matrix[:, None, None], points_3d_homo[..., None])  # (B, H, W, 4, 1)
        transformed_points = transformed_points.squeeze(-1)[..., :3]  # (B, H, W, 3)
        
        # 4. 投影回2D
        projected_points = torch.matmul(k_matrix[:, None, None], transformed_points[..., None])  # (B, H, W, 3, 1)
        projected_points = projected_points.squeeze(-1)  # (B, H, W, 3)
        
        # 归一化坐标
        projected_points = projected_points / (projected_points[..., 2:3] + 1e-6)
        pixel_coords = projected_points[..., :2]  # (B, H, W, 2)
        
        # 5. 使用双线性插值采样新的深度图
        pixel_coords = 2.0 * pixel_coords / torch.tensor([width - 1, height - 1]).to(depth_map.device) - 1.0
        transformed_depth = F.grid_sample(depth_map[:, None], pixel_coords, align_corners=True)
        
        return transformed_depth.squeeze(1), transformed_points.reshape(batch_size, -1, 3)

class RadNet(nn.Module):
    """RadNet++模型"""
    def __init__(self, input_shape, dropout_rate=0.0):
        super(RadNet, self).__init__()
        
        # RGB流
        self.mobilenet = self._load_mobilenet()
        self.rgb_stream = nn.Sequential(
            MLPConv(16),
            MLPConv(16)
        )
        
        # 雷达流
        self.radar_pool = nn.MaxPool2d(4)
        
        # 特征压缩
        self.rgb_compress = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * (input_shape[0]//4) * (input_shape[1]//4), 50),
            nn.ReLU()
        )
        
        self.radar_compress = nn.Sequential(
            nn.Flatten(),
            nn.Linear((input_shape[0]//4) * (input_shape[1]//4), 50),
            nn.ReLU()
        )
        
        # 标定块
        self.calibration_block = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        
        # 空间变换层
        self.spatial_transformer = SpatialTransformer3D()
    
    def _load_mobilenet(self):
        """加载预训练的MobileNet并截断"""
        model = models.mobilenet_v2(pretrained=True)
        return nn.Sequential(*list(model.features.children())[:7])
    
    def forward(self, rgb, radar, k_matrix, decalib_trans):
        """
        前向传播
        
        Args:
            rgb: RGB图像 (B, 3, H, W)
            radar: 雷达深度图 (B, 1, H, W)
            k_matrix: 相机内参矩阵 (B, 3, 3)
            decalib_trans: 解标定平移向量 (B, 3)
            
        Returns:
            predicted_quat: 预测的四元数 (B, 4)
            depth_maps_pred: 预测的深度图 (B, H, W)
            cloud_pred: 预测的点云 (B, N, 3)
        """
        # RGB流
        rgb_feat = self.mobilenet(rgb)
        rgb_feat = self.rgb_stream(rgb_feat)
        rgb_feat = self.rgb_compress(rgb_feat)
        
        # 雷达流
        radar_feat = self.radar_pool(radar)
        radar_feat = self.radar_compress(radar_feat)
        
        # 特征融合和四元数预测
        combined_feat = torch.cat([rgb_feat, radar_feat], dim=1)
        predicted_quat = self.calibration_block(combined_feat)
        
        # 标准化四元数
        predicted_quat = F.normalize(predicted_quat, p=2, dim=1)
        
        # 构建变换矩阵
        transform_matrix = self._quaternion_to_matrix(predicted_quat, decalib_trans)
        
        # 应用空间变换
        depth_maps_pred, cloud_pred = self.spatial_transformer(
            radar.squeeze(1),  # (B, H, W)
            transform_matrix,  # (B, 4, 4)
            k_matrix  # (B, 3, 3)
        )
        
        return predicted_quat, depth_maps_pred, cloud_pred
    
    def _quaternion_to_matrix(self, quaternion, translation):
        """
        将四元数和平移向量转换为4x4变换矩阵
        
        Args:
            quaternion: (B, 4) [w, x, y, z]
            translation: (B, 3) [x, y, z]
            
        Returns:
            transform_matrix: (B, 4, 4)
        """
        batch_size = quaternion.shape[0]
        
        # 四元数到旋转矩阵的转换
        w, x, y, z = quaternion.unbind(1)
        
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
        
        rot_matrix = torch.stack([
            1.0 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1.0 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1.0 - (txx + tyy)
        ], dim=1).reshape(batch_size, 3, 3)
        
        # 构建4x4变换矩阵
        transform_matrix = torch.eye(4).to(quaternion.device).repeat(batch_size, 1, 1)
        transform_matrix[:, :3, :3] = rot_matrix
        transform_matrix[:, :3, 3] = translation
        
        return transform_matrix 