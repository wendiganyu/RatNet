#!/usr/bin/env python3
"""
评估RadNet模型的脚本
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.radnet import RadNet
from data.dataset import RadarCameraCalibDataset, collate_fn
from utils.quaternion_utils import quaternion_angular_error, euler_from_quaternion


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='评估RadNet模型')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='评估数据集分割')
    parser.add_argument('--output_dir', type=str, default='results', help='结果输出目录')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    return parser.parse_args()


def load_config(config_path):
    """
    加载配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_dataloader(config, split):
    """
    设置数据加载器
    """
    # 数据预处理
    transform = None  # 评估时不需要数据增强
    
    # 创建数据集
    dataset = RadarCameraCalibDataset(
        dataset_path=config['dataset']['dataset_path'],
        transform=transform,
        split=split
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader


def load_model(config, checkpoint_path, device):
    """
    加载模型
    """
    # 创建模型
    model = RadNet(
        input_shape=(config['dataset']['image_height'], config['dataset']['image_width'], 3),
        dropout_rate=0.0,  # 评估时不使用dropout
        l2_reg=config['model']['l2_reg']
    ).to(device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    print(f"已加载模型检查点: {checkpoint_path}")
    print(f"模型训练周期: {checkpoint['epoch']}")
    print(f"最佳验证损失: {checkpoint['best_val_loss']:.4f}")
    
    return model


def evaluate(model, dataloader, device, output_dir, visualize=False):
    """
    评估模型
    
    Args:
        model: RadNet模型
        dataloader: 数据加载器
        device: 设备
        output_dir: 结果输出目录
        visualize: 是否可视化结果
    
    Returns:
        评估指标字典
    """
    all_quat_errors = []
    all_roll_errors = []
    all_pitch_errors = []
    all_yaw_errors = []
    
    progress_bar = tqdm(dataloader, desc='评估中')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据
            rgb_image = batch['rgb_image'].to(device)
            radar_input = batch['radar_input'].to(device)
            K = batch['K'].to(device)
            gt_trans = batch['translation'].to(device)
            gt_quat = batch['quaternion'].to(device)
            
            # 前向传播
            pred_quat, depth_maps_pred, cloud_pred = model(rgb_image, radar_input, K, gt_trans)
            
            # 计算角度误差
            quat_errors = quaternion_angular_error(pred_quat, gt_quat)
            all_quat_errors.extend(quat_errors.cpu().numpy())
            
            # 计算欧拉角误差
            pred_euler = euler_from_quaternion(pred_quat)
            gt_euler = euler_from_quaternion(gt_quat)
            
            roll_errors = torch.abs(pred_euler[:, 0] - gt_euler[:, 0])
            pitch_errors = torch.abs(pred_euler[:, 1] - gt_euler[:, 1])
            yaw_errors = torch.abs(pred_euler[:, 2] - gt_euler[:, 2])
            
            all_roll_errors.extend(roll_errors.cpu().numpy())
            all_pitch_errors.extend(pitch_errors.cpu().numpy())
            all_yaw_errors.extend(yaw_errors.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({
                'quat_err': f'{torch.mean(quat_errors).item():.2f}°',
                'roll_err': f'{torch.mean(roll_errors).item():.2f}°',
                'pitch_err': f'{torch.mean(pitch_errors).item():.2f}°',
                'yaw_err': f'{torch.mean(yaw_errors).item():.2f}°'
            })
            
            # 可视化第一批次的结果
            if batch_idx == 0 and visualize:
                visualize_results(
                    rgb_image=rgb_image.cpu(),
                    radar_input=radar_input.cpu(),
                    depth_maps_pred=depth_maps_pred.cpu(),
                    gt_quat=gt_quat.cpu(),
                    pred_quat=pred_quat.cpu(),
                    output_dir=output_dir
                )
    
    # 计算统计指标
    metrics = {
        'mean_quat_error': float(np.mean(all_quat_errors)),
        'median_quat_error': float(np.median(all_quat_errors)),
        'std_quat_error': float(np.std(all_quat_errors)),
        'mean_roll_error': float(np.mean(all_roll_errors)),
        'mean_pitch_error': float(np.mean(all_pitch_errors)),
        'mean_yaw_error': float(np.mean(all_yaw_errors))
    }
    
    # 可视化误差分布
    if visualize:
        visualize_error_distribution(
            quat_errors=all_quat_errors,
            roll_errors=all_roll_errors,
            pitch_errors=all_pitch_errors,
            yaw_errors=all_yaw_errors,
            output_dir=output_dir
        )
    
    return metrics


def visualize_results(rgb_image, radar_input, depth_maps_pred, gt_quat, pred_quat, output_dir):
    """
    可视化评估结果
    
    Args:
        rgb_image: RGB图像
        radar_input: 雷达输入
        depth_maps_pred: 预测的深度图
        gt_quat: 真实四元数
        pred_quat: 预测的四元数
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 取第一个样本
    rgb = rgb_image[0].permute(1, 2, 0).numpy()
    radar = radar_input[0, 0].numpy()
    depth_pred = depth_maps_pred[0].numpy()
    
    # 计算欧拉角
    gt_euler = euler_from_quaternion(gt_quat)
    pred_euler = euler_from_quaternion(pred_quat)
    
    # 创建图像网格
    plt.figure(figsize=(15, 10))
    
    # RGB图像
    plt.subplot(2, 2, 1)
    plt.imshow(rgb)
    plt.title('RGB图像')
    plt.axis('off')
    
    # 雷达输入
    plt.subplot(2, 2, 2)
    plt.imshow(radar, cmap='jet')
    plt.title('雷达输入')
    plt.colorbar(label='深度')
    plt.axis('off')
    
    # 预测的深度图
    plt.subplot(2, 2, 3)
    plt.imshow(depth_pred, cmap='jet')
    plt.title('预测的深度图')
    plt.colorbar(label='深度')
    plt.axis('off')
    
    # 欧拉角对比
    plt.subplot(2, 2, 4)
    angles = ['Roll', 'Pitch', 'Yaw']
    x = np.arange(len(angles))
    width = 0.35
    
    gt_values = gt_euler[0].numpy()
    pred_values = pred_euler[0].numpy()
    
    plt.bar(x - width/2, gt_values, width, label='真实值')
    plt.bar(x + width/2, pred_values, width, label='预测值')
    
    plt.ylabel('角度 (度)')
    plt.title('欧拉角对比')
    plt.xticks(x, angles)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_visualization.png'))
    plt.close()


def visualize_error_distribution(quat_errors, roll_errors, pitch_errors, yaw_errors, output_dir):
    """
    可视化误差分布
    
    Args:
        quat_errors: 四元数误差列表
        roll_errors: 横滚角误差列表
        pitch_errors: 俯仰角误差列表
        yaw_errors: 偏航角误差列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 误差直方图
    plt.figure(figsize=(15, 10))
    
    # 四元数误差
    plt.subplot(2, 2, 1)
    plt.hist(quat_errors, bins=50, alpha=0.7)
    plt.axvline(np.mean(quat_errors), color='r', linestyle='dashed', linewidth=2, label=f'平均值: {np.mean(quat_errors):.2f}°')
    plt.axvline(np.median(quat_errors), color='g', linestyle='dashed', linewidth=2, label=f'中位数: {np.median(quat_errors):.2f}°')
    plt.title('四元数角度误差分布')
    plt.xlabel('误差 (度)')
    plt.ylabel('样本数')
    plt.legend()
    
    # 横滚角误差
    plt.subplot(2, 2, 2)
    plt.hist(roll_errors, bins=50, alpha=0.7, color='orange')
    plt.axvline(np.mean(roll_errors), color='r', linestyle='dashed', linewidth=2, label=f'平均值: {np.mean(roll_errors):.2f}°')
    plt.title('横滚角误差分布')
    plt.xlabel('误差 (度)')
    plt.ylabel('样本数')
    plt.legend()
    
    # 俯仰角误差
    plt.subplot(2, 2, 3)
    plt.hist(pitch_errors, bins=50, alpha=0.7, color='green')
    plt.axvline(np.mean(pitch_errors), color='r', linestyle='dashed', linewidth=2, label=f'平均值: {np.mean(pitch_errors):.2f}°')
    plt.title('俯仰角误差分布')
    plt.xlabel('误差 (度)')
    plt.ylabel('样本数')
    plt.legend()
    
    # 偏航角误差
    plt.subplot(2, 2, 4)
    plt.hist(yaw_errors, bins=50, alpha=0.7, color='purple')
    plt.axvline(np.mean(yaw_errors), color='r', linestyle='dashed', linewidth=2, label=f'平均值: {np.mean(yaw_errors):.2f}°')
    plt.title('偏航角误差分布')
    plt.xlabel('误差 (度)')
    plt.ylabel('样本数')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()
    
    # 保存累积误差曲线
    plt.figure(figsize=(10, 6))
    
    errors = [quat_errors, roll_errors, pitch_errors, yaw_errors]
    labels = ['四元数', '横滚角', '俯仰角', '偏航角']
    colors = ['blue', 'orange', 'green', 'purple']
    
    for i, (err, label, color) in enumerate(zip(errors, labels, colors)):
        # 排序误差
        sorted_errors = np.sort(err)
        # 计算累积分布
        y = np.arange(len(sorted_errors)) / float(len(sorted_errors) - 1)
        
        plt.plot(sorted_errors, y, label=label, color=color)
    
    plt.xlabel('误差 (度)')
    plt.ylabel('累积概率')
    plt.title('角度误差的累积分布函数')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'cumulative_error.png'))
    plt.close()


def print_metrics(metrics):
    """
    打印评估指标
    
    Args:
        metrics: 评估指标字典
    """
    table = PrettyTable()
    table.field_names = ["指标", "值 (度)"]
    table.align["指标"] = "l"
    table.align["值 (度)"] = "r"
    
    table.add_row(["四元数平均误差", f"{metrics['mean_quat_error']:.4f}"])
    table.add_row(["四元数中位数误差", f"{metrics['median_quat_error']:.4f}"])
    table.add_row(["四元数标准差", f"{metrics['std_quat_error']:.4f}"])
    table.add_row(["横滚角平均误差", f"{metrics['mean_roll_error']:.4f}"])
    table.add_row(["俯仰角平均误差", f"{metrics['mean_pitch_error']:.4f}"])
    table.add_row(["偏航角平均误差", f"{metrics['mean_yaw_error']:.4f}"])
    
    print("\n评估结果:")
    print(table)


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 设置数据加载器
    dataloader = setup_dataloader(config, args.split)
    
    # 加载模型
    model = load_model(config, args.checkpoint, device)
    
    # 评估模型
    metrics = evaluate(model, dataloader, device, args.output_dir, args.visualize)
    
    # 打印结果
    print_metrics(metrics)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, f'{args.split}_metrics.npy'), metrics)
    
    print(f"评估完成! 结果保存在 {args.output_dir}")


if __name__ == "__main__":
    main()
