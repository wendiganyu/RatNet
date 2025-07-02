import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from pathlib import Path

from models.radnet import RadNet
from data.dataset import RadarCameraDataset
from utils.quaternion_utils import quaternion_error, quaternion_to_euler
from nuscenes.nuscenes import NuScenes

def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    
    quat_errors = []
    euler_errors = []
    depth_errors = []
    cloud_errors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='评估中'):
            # 将数据移到设备
            rgb = batch['image'].to(device)
            radar = batch['radar_points'].to(device)
            k_matrix = batch['k_matrix'].to(device)
            decalib_quat = batch['decalib_quat'].to(device)
            decalib_trans = batch['decalib_trans'].to(device)
            
            # 前向传播
            pred_quat, pred_depth, pred_cloud = model(rgb, radar, k_matrix, decalib_trans)
            
            # 计算四元数误差
            quat_error = quaternion_error(pred_quat, decalib_quat)
            quat_errors.append(quat_error.cpu().numpy())
            
            # 计算欧拉角误差
            pred_euler = quaternion_to_euler(pred_quat)
            gt_euler = quaternion_to_euler(decalib_quat)
            euler_error = torch.abs(pred_euler - gt_euler)
            euler_errors.append(euler_error.cpu().numpy())
            
            # 计算深度图误差
            depth_error = torch.abs(pred_depth - radar.squeeze(1))
            depth_errors.append(depth_error.mean().item())
            
            # 计算点云误差
            cloud_error = torch.cdist(pred_cloud, radar[..., :3]).min(dim=1)[0]
            cloud_errors.append(cloud_error.mean().item())
    
    # 转换为numpy数组
    quat_errors = np.concatenate(quat_errors)
    euler_errors = np.concatenate(euler_errors)
    depth_errors = np.array(depth_errors)
    cloud_errors = np.array(cloud_errors)
    
    # 计算统计数据
    results = {
        'quaternion_error': {
            'mean': np.mean(quat_errors),
            'std': np.std(quat_errors),
            'median': np.median(quat_errors)
        },
        'euler_error': {
            'roll': {
                'mean': np.mean(euler_errors[:, 0]),
                'std': np.std(euler_errors[:, 0])
            },
            'pitch': {
                'mean': np.mean(euler_errors[:, 1]),
                'std': np.std(euler_errors[:, 1])
            },
            'yaw': {
                'mean': np.mean(euler_errors[:, 2]),
                'std': np.std(euler_errors[:, 2])
            }
        },
        'depth_error': {
            'mean': np.mean(depth_errors),
            'std': np.std(depth_errors)
        },
        'cloud_error': {
            'mean': np.mean(cloud_errors),
            'std': np.std(cloud_errors)
        }
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='评估RadNet模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='模型检查点路径')
    parser.add_argument('--data_root', type=str, required=True,
                      help='nuScenes数据集根目录')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                      help='nuScenes数据集版本')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    nusc = NuScenes(
        version=args.version,
        dataroot=args.data_root,
        verbose=True
    )
    
    val_dataset = RadarCameraDataset(
        nusc=nusc,
        split='val'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = RadNet(
        input_shape=[150, 240, 3],
        dropout_rate=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估模型
    results = evaluate_model(model, val_loader, device)
    
    # 打印结果
    print("\n评估结果:")
    print(f"四元数误差: {results['quaternion_error']['mean']:.4f}° ± {results['quaternion_error']['std']:.4f}°")
    print("\n欧拉角误差:")
    print(f"Roll:  {results['euler_error']['roll']['mean']:.4f}° ± {results['euler_error']['roll']['std']:.4f}°")
    print(f"Pitch: {results['euler_error']['pitch']['mean']:.4f}° ± {results['euler_error']['pitch']['std']:.4f}°")
    print(f"Yaw:   {results['euler_error']['yaw']['mean']:.4f}° ± {results['euler_error']['yaw']['std']:.4f}°")
    print(f"\n深度图误差: {results['depth_error']['mean']:.4f}m ± {results['depth_error']['std']:.4f}m")
    print(f"点云误差: {results['cloud_error']['mean']:.4f}m ± {results['cloud_error']['std']:.4f}m")

if __name__ == '__main__':
    main() 