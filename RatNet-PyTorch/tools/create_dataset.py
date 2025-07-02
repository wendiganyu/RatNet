import os
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import json
from tqdm import tqdm
import argparse
from pathlib import Path

def create_decalibration_dataset(nusc, output_dir, num_samples=1000, split='train'):
    """
    创建解标定数据集
    
    Args:
        nusc: NuScenes实例
        output_dir: 输出目录
        num_samples: 要生成的样本数量
        split: 数据集分割（'train' 或 'val'）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取场景列表
    scene_tokens = [s['token'] for s in nusc.scene]
    
    # 创建数据集
    dataset = []
    pbar = tqdm(total=num_samples, desc=f'生成{split}集')
    
    for scene_token in scene_tokens:
        scene = nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        
        while sample_token and len(dataset) < num_samples:
            sample = nusc.get('sample', sample_token)
            
            # 获取相机和雷达数据
            cam_token = sample['data']['CAM_FRONT']
            radar_token = sample['data']['RADAR_FRONT']
            
            cam_data = nusc.get('sample_data', cam_token)
            radar_data = nusc.get('sample_data', radar_token)
            
            # 获取标定参数
            cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            radar_calib = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
            
            # 生成随机解标定
            decalib_angles = np.random.uniform(
                low=[-np.pi/18, -np.pi/18, -np.pi/6],  # [-10°, -10°, -30°]
                high=[np.pi/18, np.pi/18, np.pi/6],    # [10°, 10°, 30°]
                size=3
            )
            
            decalib_trans = np.random.uniform(
                low=-0.1,
                high=0.1,
                size=3
            )
            
            # 保存样本信息
            sample_info = {
                'scene_token': scene_token,
                'sample_token': sample_token,
                'cam_filename': cam_data['filename'],
                'radar_filename': radar_data['filename'],
                'cam_intrinsic': cam_calib['camera_intrinsic'].tolist(),
                'cam_extrinsic': cam_calib['translation'] + cam_calib['rotation'],
                'radar_extrinsic': radar_calib['translation'] + radar_calib['rotation'],
                'decalib_angles': decalib_angles.tolist(),
                'decalib_trans': decalib_trans.tolist()
            }
            
            dataset.append(sample_info)
            pbar.update(1)
            
            # 获取下一个样本
            sample_token = sample['next']
            
            if len(dataset) >= num_samples:
                break
    
    pbar.close()
    
    # 保存数据集
    output_file = output_dir / f'decalib_dataset_{split}.json'
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f'已保存{len(dataset)}个样本到{output_file}')

def main():
    parser = argparse.ArgumentParser(description='创建解标定数据集')
    parser.add_argument('--data_root', type=str, required=True,
                      help='nuScenes数据集根目录')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                      help='nuScenes数据集版本')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出目录')
    parser.add_argument('--num_train', type=int, default=1000,
                      help='训练集样本数量')
    parser.add_argument('--num_val', type=int, default=200,
                      help='验证集样本数量')
    
    args = parser.parse_args()
    
    # 初始化NuScenes
    nusc = NuScenes(
        version=args.version,
        dataroot=args.data_root,
        verbose=True
    )
    
    # 创建训练集和验证集
    create_decalibration_dataset(
        nusc,
        args.output_dir,
        num_samples=args.num_train,
        split='train'
    )
    
    create_decalibration_dataset(
        nusc,
        args.output_dir,
        num_samples=args.num_val,
        split='val'
    )

if __name__ == '__main__':
    main() 