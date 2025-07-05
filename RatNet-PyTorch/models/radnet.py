import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import numpy as np


class MLPConv(nn.Module):
    """
    多层感知机卷积模块，原TF模型中使用的mlpconv_layer的PyTorch实现
    """
    def __init__(self, in_channels, filter_maps, kernel_size=5, dropout_rate=0.0):
        super(MLPConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, filter_maps, kernel_size, padding=kernel_size//2)
        self.mlp1 = nn.Conv2d(filter_maps, filter_maps, kernel_size=1)
        self.mlp2 = nn.Conv2d(filter_maps, filter_maps, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class SpatialTransformerLayer(nn.Module):
    """
    空间变换器层，用于将雷达投影转换为点云
    """
    def __init__(self):
        super(SpatialTransformerLayer, self).__init__()

    def forward(self, predicted_quaternion, radar_input, k_matrix, translation):
        """
        将预测的四元数和平移应用于雷达输入
        
        Args:
            predicted_quaternion: 形状为 [batch_size, 4] 的四元数张量
            radar_input: 形状为 [batch_size, 1, H, W] 的雷达投影
            k_matrix: 形状为 [batch_size, 3, 3] 的相机内参矩阵
            translation: 形状为 [batch_size, 3] 的平移向量
            
        Returns:
            depth_maps_predicted: 预测的深度图
            cloud_pred: 预测的点云
        """
        batch_size = radar_input.size(0)
        height = radar_input.size(2)
        width = radar_input.size(3)
        
        # 确保四元数已归一化
        predicted_quaternion = F.normalize(predicted_quaternion, p=2, dim=1)
        
        # 创建从四元数和平移的变换矩阵
        # 使用绝对导入
        from utils.quaternion_utils import transform_from_quaternion_and_translation
        transform_matrices = transform_from_quaternion_and_translation(predicted_quaternion, translation)
        
        # 初始化输出张量
        depth_maps_predicted = torch.zeros((batch_size, height, width), device=radar_input.device)
        cloud_points_list = []
        
        # 对每个批次单独处理
        for b in range(batch_size):
            # 获取当前批次的雷达深度图
            radar_depth = radar_input[b, 0]  # [H, W]
            
            # 创建点云（将有效深度点转换为3D坐标）
            mask = radar_depth > 0  # 只考虑深度值 > 0 的点
            y_indices, x_indices = torch.nonzero(mask, as_tuple=True)
            
            if len(y_indices) == 0:
                # 如果没有有效点，添加一个空的点云
                cloud_points_list.append(torch.zeros((0, 3), device=radar_input.device))
                continue
            
            # 获取这些点的深度值
            z_values = radar_depth[mask]
            
            # 从像素坐标创建归一化相机坐标
            k_inv = torch.inverse(k_matrix[b])
            
            # 组合像素坐标
            pixels = torch.stack([x_indices.float(), y_indices.float(), torch.ones_like(x_indices, dtype=torch.float)], dim=1)  # [N, 3]
            
            # 将像素投影到相机坐标
            cam_points = torch.matmul(k_inv, pixels.transpose(0, 1)).transpose(0, 1)  # [N, 3]
            
            # 将相机坐标乘以深度
            cam_points = cam_points * z_values.unsqueeze(1)  # [N, 3]
            
            # 将相机点转换为齐次坐标
            cam_points_homogeneous = torch.cat([cam_points, torch.ones((cam_points.size(0), 1), device=cam_points.device)], dim=1)  # [N, 4]
            
            # 应用变换矩阵
            transform = transform_matrices[b]  # [4, 4]
            transformed_points = torch.matmul(transform, cam_points_homogeneous.transpose(0, 1)).transpose(0, 1)  # [N, 4]
            
            # 将齐次坐标转换回3D坐标
            transformed_points = transformed_points[:, :3]  # [N, 3]
            
            # 将点投影回图像平面
            projected_points = torch.matmul(k_matrix[b], transformed_points.transpose(0, 1))  # [3, N]
            projected_points = projected_points.transpose(0, 1)  # [N, 3]
            
            # 计算像素坐标和深度
            pixel_x = projected_points[:, 0] / projected_points[:, 2]
            pixel_y = projected_points[:, 1] / projected_points[:, 2]
            depth = projected_points[:, 2]
            
            # 将像素坐标四舍五入到最近的整数
            pixel_x = torch.round(pixel_x).long()
            pixel_y = torch.round(pixel_y).long()
            
            # 筛选出有效的像素坐标（在图像范围内）
            valid_mask = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
            pixel_x = pixel_x[valid_mask]
            pixel_y = pixel_y[valid_mask]
            depth = depth[valid_mask]
            transformed_points = transformed_points[valid_mask]
            
            # 创建深度图
            depth_map = torch.zeros((height, width), device=radar_input.device)
            depth_map[pixel_y, pixel_x] = depth
            depth_maps_predicted[b] = depth_map
            
            # 存储点云
            cloud_points_list.append(transformed_points)
        
        # 将点云列表打包为批次
        # 注意：点云可能有不同数量的点，因此返回列表而不是张量
        return depth_maps_predicted, cloud_points_list


class RadNet(nn.Module):
    """
    RadNet模型：用于雷达-相机旋转校准的深度学习模型
    """
    def __init__(self, input_shape=(150, 240, 3), dropout_rate=0.0, l2_reg=0.0):
        super(RadNet, self).__init__()
        
        self.rgb_shape = input_shape
        self.radar_shape = (input_shape[0], input_shape[1], 1)  # 雷达通道为1
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        # RGB流
        self.rgb_stream = self._build_rgb_stream()
        
        # 雷达流 - 使用卷积替代池化，确保输出尺寸合理
        self.radar_conv = nn.Conv2d(
            in_channels=1,  # 雷达输入为1通道
            out_channels=8,  # 增加特征通道数
            kernel_size=3,   # 3x3卷积核
            stride=2,       # 下采样
            padding=1       # 保持输出大小正确
        )
        
        # 使用虚拟输入创建特征，以获取确切的特征大小
        rgb_dummy = torch.zeros((1, 3, input_shape[0], input_shape[1]))
        radar_dummy = torch.zeros((1, 1, input_shape[0], input_shape[1]))
        
        # 获取实际的特征大小
        with torch.no_grad():
            rgb_features = self.rgb_stream(rgb_dummy)
            radar_features = self.radar_conv(radar_dummy)
            radar_features = F.relu(radar_features)
            
            rgb_flat = rgb_features.flatten(start_dim=1)
            radar_flat = radar_features.flatten(start_dim=1)
            combined_features = torch.cat([rgb_flat, radar_flat], dim=1)
            
            self.feature_size = combined_features.shape[1]
            print(f"计算得到的特征维度: {self.feature_size}")
        
        # 校准块
        self.calib_block = self._build_calibration_block()
        
        # 空间变换器层
        self.spatial_transformer = SpatialTransformerLayer()
    
    def _build_rgb_stream(self):
        """
        构建RGB流部分
        """
        # 加载预训练的MobileNetV2并截断
        mobilenet = mobilenet_v2(pretrained=True)
        
        # 使用模型的前20层（相当于TensorFlow模型中的前20层）
        pretrained_layers = list(mobilenet.features[:7])  # 约等于TensorFlow的层20
        
        rgb_stream = nn.Sequential(
            *pretrained_layers,
            MLPConv(in_channels=32, filter_maps=16, kernel_size=5),
            MLPConv(in_channels=16, filter_maps=16, kernel_size=5)
        )
        
        return rgb_stream
    
    def _build_calibration_block(self):
        """
        构建校准块
        """
        return nn.Sequential(
            # 输入层
            nn.Flatten(),  # 确保输入被展平
            nn.Linear(self.feature_size, 100),  # 使用动态计算的特征大小
            nn.ReLU(inplace=True),
            
            # 全连接层
            nn.Linear(100, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            
            # 四元数回归
            nn.Linear(256, 4)
        )
    
    def forward(self, rgb_image, radar_input, k_matrix, decalib_gt_trans):
        """
        前向传播
        
        Args:
            rgb_image: RGB图像，形状为 [batch_size, 3, H, W]
            radar_input: 雷达投影，形状为 [batch_size, 1, H, W]
            k_matrix: 相机内参矩阵，形状为 [batch_size, 3, 3]
            decalib_gt_trans: 地面真值平移向量，形状为 [batch_size, 3]
            
        Returns:
            predicted_quat: 预测的四元数，形状为 [batch_size, 4]
            depth_maps_predicted: 预测的深度图
            cloud_pred: 预测的点云
        """
        # RGB流
        rgb_features = self.rgb_stream(rgb_image)
        
        # 雷达流 - 首先确保输入维度正确 [batch, channels, height, width]
        # 检查输入维度并调整
        if radar_input.dim() == 4 and radar_input.size(1) != 1 and radar_input.size(3) == 1:
            # 如果是 [batch, height, width, channels]，转换为 [batch, channels, height, width]
            radar_input = radar_input.permute(0, 3, 1, 2)
        
        radar_features = self.radar_conv(radar_input)
        radar_features = F.relu(radar_features)
        
        # 连接特征
        rgb_flat = rgb_features.flatten(start_dim=1)
        radar_flat = radar_features.flatten(start_dim=1)
        combined_features = torch.cat([rgb_flat, radar_flat], dim=1)
        
        # 校准块 - 预测四元数
        predicted_quat = self.calib_block(combined_features)
        
        # 归一化四元数
        predicted_quat = F.normalize(predicted_quat, p=2, dim=1)
        
        # 处理四元数w分量的符号
        # 如果w < 0，翻转四元数的符号，因为q和-q表示相同的旋转
        w_sign = torch.sign(predicted_quat[:, 0])
        w_sign = torch.where(w_sign < 0, -torch.ones_like(w_sign), torch.ones_like(w_sign))
        predicted_quat = predicted_quat * w_sign.unsqueeze(1)
        
        # 可以选择将pitch和roll分量置零（如原始代码所示，用于只有yaw的数据集）
        # predicted_quat = predicted_quat * torch.tensor([1.0, 0.0, 1.0, 0.0], device=predicted_quat.device)
        
        # 空间变换器层 - 应用预测的变换
        depth_maps_predicted, cloud_pred = self.spatial_transformer(
            predicted_quat, radar_input, k_matrix, decalib_gt_trans
        )
        
        return predicted_quat, depth_maps_predicted, cloud_pred


class RadNetLoss(nn.Module):
    """
    RadNet损失函数
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super(RadNetLoss, self).__init__()
        self.alpha = alpha  # photometric loss权重
        self.beta = beta    # 3D point cloud loss权重
    
    def forward(self, predicted_quat, depth_maps_predicted, cloud_pred, groundtruth_quat):
        """
        计算损失
        
        Args:
            predicted_quat: 预测的四元数，形状为 [batch_size, 4]
            depth_maps_predicted: 预测的深度图
            cloud_pred: 预测的点云
            groundtruth_quat: 地面真值四元数，形状为 [batch_size, 4]
            
        Returns:
            total_loss: 总损失
            quat_loss: 四元数损失
            cloud_loss: 点云损失
        """
        batch_size = predicted_quat.size(0)
        
        # 归一化四元数
        predicted_quat = F.normalize(predicted_quat, p=2, dim=1)
        groundtruth_quat = F.normalize(groundtruth_quat, p=2, dim=1)
        
        # 四元数损失（欧氏距离）
        diff = (groundtruth_quat - predicted_quat) ** 2
        quat_loss = torch.sqrt(torch.sum(diff, dim=1)).mean()
        
        # 点云损失（平均平方距离）
        cloud_loss = torch.tensor(0.0, device=predicted_quat.device)
        
        # 总损失
        total_loss = self.alpha * quat_loss + self.beta * cloud_loss
        
        return total_loss, quat_loss, cloud_loss 