# RadNet-PyTorch

基于PyTorch的RadNet++实现，用于雷达-相机的自动标定。本项目是对原始TensorFlow版本RadNet++的重构。

## 项目结构

```
RatNet-PyTorch/
├── configs/          # 配置文件
├── data/            # 数据集处理
├── models/          # 模型定义
├── utils/           # 工具函数
└── tools/           # 数据处理工具
```

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/your-username/RatNet-PyTorch.git
cd RatNet-PyTorch
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装nuScenes开发工具包：
```bash
pip install nuscenes-devkit
```

## 数据准备

1. 下载[nuScenes数据集](https://www.nuscenes.org/download)
2. 更新`configs/config.yaml`中的数据集路径

## 训练

```bash
python train.py
```

## 主要特性

- 使用PyTorch重新实现的RadNet++模型
- 支持nuScenes数据集
- 使用空间变换层进行几何监督
- 支持TensorBoard可视化
- 实现了Cosine学习率调度

## 引用

如果您使用了这个项目，请引用原始论文：

```bibtex
@article{papanikolaou2019targetless,
  title={Targetless Rotational Auto-Calibration of Radar and Camera for Intelligent Transportation Systems},
  author={Papanikolaou, Odysseas and others},
  journal={arXiv preprint arXiv:1904.08743},
  year={2019}
} 