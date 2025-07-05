import os
import argparse
import yaml
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 导入自定义模块
from models.radnet import RadNet, RadNetLoss
from data.dataset import RadarCameraCalibDataset, collate_fn
from utils.quaternion_utils import quaternion_angular_error, euler_from_quaternion


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='训练RadNet模型')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    return parser.parse_args()


def load_config(config_path):
    """
    加载配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """
    设置设备(CPU/GPU)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f'使用设备: {device}')
    return device


def setup_dataloaders(config):
    """
    设置数据加载器
    """
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = RadarCameraCalibDataset(
        dataset_path=config['dataset']['dataset_path'],
        transform=train_transform,
        split='train'
    )
    
    val_dataset = RadarCameraCalibDataset(
        dataset_path=config['dataset']['dataset_path'],
        transform=val_transform,
        split='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def build_model(config, device):
    """
    构建模型
    """
    model = RadNet(
        input_shape=(config['dataset']['image_height'], config['dataset']['image_width'], 3),
        dropout_rate=config['model']['dropout_rate'],
        l2_reg=config['model']['l2_reg']
    ).to(device)
    
    # 损失函数
    criterion = RadNetLoss(alpha=config['loss']['alpha'], beta=config['loss']['beta']).to(device)
    
    # 优化器
    if config['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(config['optimizer']['beta1'], config['optimizer']['beta2'])
        )
    elif config['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['optimizer']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['optimizer']['type'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(config['optimizer']['beta1'], config['optimizer']['beta2'])
        )
    else:
        raise ValueError(f"不支持的优化器类型: {config['optimizer']['type']}")
    
    # 学习率调度器
    if config['training']['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    elif config['training']['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['lr_step_size'],
            gamma=config['training']['lr_gamma']
        )
    elif config['training']['lr_scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['lr_gamma'],
            patience=config['training']['lr_step_size'] // 2
        )
    else:
        raise ValueError(f"不支持的学习率调度器: {config['training']['lr_scheduler']}")
    
    return model, criterion, optimizer, scheduler


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    加载检查点
    """
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"已恢复至第 {start_epoch} 个周期，最佳验证损失: {best_val_loss:.4f}")
    return model, optimizer, scheduler, start_epoch, best_val_loss


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, is_best, output_dir):
    """
    保存检查点
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    
    # 保存最新的检查点
    latest_path = os.path.join(output_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # 如果是最佳模型，同时保存为best_model.pth
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"已保存新的最佳模型，验证损失: {best_val_loss:.4f}")
    
    # 每隔一定周期保存一次
    if (epoch + 1) % 10 == 0:
        epoch_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, epoch_path)


def train_one_epoch(model, criterion, optimizer, dataloader, device, epoch):
    """
    训练一个周期
    """
    model.train()
    running_loss = 0.0
    running_quat_loss = 0.0
    running_cloud_loss = 0.0
    epoch_quat_errors = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [Train]')
    
    for batch_idx, batch in enumerate(progress_bar):
        # 获取数据
        rgb_image = batch['rgb_image'].to(device)
        radar_input = batch['radar_input'].to(device)
        K = batch['K'].to(device)
        gt_trans = batch['translation'].to(device)
        gt_quat = batch['quaternion'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        pred_quat, depth_maps_pred, cloud_pred = model(rgb_image, radar_input, K, gt_trans)
        
        # 计算损失
        loss, quat_loss, cloud_loss = criterion(pred_quat, depth_maps_pred, cloud_pred, gt_quat)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        running_quat_loss += quat_loss.item()
        running_cloud_loss += cloud_loss.item()
        
        # 计算角度误差
        with torch.no_grad():
            angle_errors = quaternion_angular_error(pred_quat, gt_quat)
            epoch_quat_errors.extend(angle_errors.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'quat_loss': f'{quat_loss.item():.4f}',
            'angle_err': f'{torch.mean(angle_errors).item():.2f}°'
        })
    
    # 计算平均损失和误差
    avg_loss = running_loss / len(dataloader)
    avg_quat_loss = running_quat_loss / len(dataloader)
    avg_cloud_loss = running_cloud_loss / len(dataloader)
    avg_angle_error = np.mean(epoch_quat_errors)
    
    metrics = {
        'loss': avg_loss,
        'quat_loss': avg_quat_loss,
        'cloud_loss': avg_cloud_loss,
        'angle_error': avg_angle_error
    }
    
    return metrics


def validate(model, criterion, dataloader, device):
    """
    在验证集上评估模型
    """
    model.eval()
    running_loss = 0.0
    running_quat_loss = 0.0
    running_cloud_loss = 0.0
    epoch_quat_errors = []
    
    progress_bar = tqdm(dataloader, desc='Validation')
    
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
            
            # 计算损失
            loss, quat_loss, cloud_loss = criterion(pred_quat, depth_maps_pred, cloud_pred, gt_quat)
            
            # 统计信息
            running_loss += loss.item()
            running_quat_loss += quat_loss.item()
            running_cloud_loss += cloud_loss.item()
            
            # 计算角度误差
            angle_errors = quaternion_angular_error(pred_quat, gt_quat)
            epoch_quat_errors.extend(angle_errors.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'angle_err': f'{torch.mean(angle_errors).item():.2f}°'
            })
    
    # 计算平均损失和误差
    avg_loss = running_loss / len(dataloader)
    avg_quat_loss = running_quat_loss / len(dataloader)
    avg_cloud_loss = running_cloud_loss / len(dataloader)
    avg_angle_error = np.mean(epoch_quat_errors)
    
    metrics = {
        'loss': avg_loss,
        'quat_loss': avg_quat_loss,
        'cloud_loss': avg_cloud_loss,
        'angle_error': avg_angle_error
    }
    
    return metrics


def train(config, model, criterion, optimizer, scheduler, train_loader, val_loader, device, start_epoch=0, best_val_loss=float('inf')):
    """
    训练模型
    """
    # 设置输出目录
    output_dir = config['output']['checkpoint_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置TensorBoard
    log_dir = config['output']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 训练循环
    epochs = config['training']['epochs']
    early_stopping_patience = config['training']['early_stopping_patience']
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        print(f"\n开始第 {epoch + 1}/{epochs} 个周期")
        
        # 训练
        train_metrics = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch)
        
        # 验证
        val_metrics = validate(model, criterion, val_loader, device)
        
        # 更新学习率
        if config['training']['lr_scheduler'] == 'plateau':
            scheduler.step(val_metrics['loss'])
        else:
            scheduler.step()
        
        # 记录指标
        print(f"训练损失: {train_metrics['loss']:.4f}, 训练角度误差: {train_metrics['angle_error']:.2f}°")
        print(f"验证损失: {val_metrics['loss']:.4f}, 验证角度误差: {val_metrics['angle_error']:.2f}°")
        
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('QuatLoss/train', train_metrics['quat_loss'], epoch)
        writer.add_scalar('QuatLoss/val', val_metrics['quat_loss'], epoch)
        writer.add_scalar('CloudLoss/train', train_metrics['cloud_loss'], epoch)
        writer.add_scalar('CloudLoss/val', val_metrics['cloud_loss'], epoch)
        writer.add_scalar('AngleError/train', train_metrics['angle_error'], epoch)
        writer.add_scalar('AngleError/val', val_metrics['angle_error'], epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存检查点
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, is_best, output_dir)
        
        # 早停
        if patience_counter >= early_stopping_patience:
            print(f"\n验证损失连续 {early_stopping_patience} 个周期未改善，停止训练。")
            break
    
    writer.close()
    return best_val_loss


def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = setup_device(config)
    
    # 设置数据加载器
    train_loader, val_loader = setup_dataloaders(config)
    
    # 构建模型
    model, criterion, optimizer, scheduler = build_model(config, device)
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        model, optimizer, scheduler, start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, args.resume
        )
    
    # 训练模型
    best_val_loss = train(
        config, model, criterion, optimizer, scheduler,
        train_loader, val_loader, device, start_epoch, best_val_loss
    )
    
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
