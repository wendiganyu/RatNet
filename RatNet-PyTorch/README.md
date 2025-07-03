# RadNet++ PyTorch版本

RadNet++是一个基于深度学习的雷达-相机旋转校准模型的PyTorch实现。原始论文可在[此处](https://arxiv.org/abs/1904.08743)找到。

## 简介

这个项目是基于TensorFlow实现的RadNet++模型的PyTorch重构版本。RadNet++是一个用于自动校准毫米波雷达和相机之间旋转关系的深度学习模型，通过空间变换层实现几何监督。该模型已成功应用于德国A9高速公路上安装的交通雷达和相机传感器的旋转校准。

## 安装

### 环境要求

- Python 3.7+
- PyTorch 1.8.0+
- CUDA 10.2+ (如果使用GPU)

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourname/RatNet-PyTorch.git
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

## 使用方法

### 数据集准备

1. 下载[nuScenes数据集](https://www.nuscenes.org/download)
2. 使用数据预处理脚本创建校准数据集：
```bash
python tools/create_dataset.py --config configs/config.yaml
```

### 模型训练

1. 修改`configs/config.yaml`中的配置参数
2. 运行训练脚本：
```bash
python train.py --config configs/config.yaml
```

### 评估模型

```bash
python tools/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

## 项目结构

```
RatNet-PyTorch/
├── configs/                    # 配置文件
│   └── config.yaml            # 默认配置
├── data/                       # 数据加载和预处理代码
│   ├── __init__.py
│   └── dataset.py             # 数据集类
├── models/                     # 模型定义
│   ├── __init__.py
│   └── radnet.py              # RadNet模型
├── tools/                      # 工具脚本
│   ├── create_dataset.py      # 数据集创建
│   └── evaluate.py            # 模型评估
├── utils/                      # 工具函数
│   ├── __init__.py
│   └── quaternion_utils.py    # 四元数操作
├── train.py                    # 训练脚本
├── requirements.txt            # 依赖包列表
├── setup.py                    # 安装脚本
└── README.md                   # 项目说明
```

## 引用

如果您使用本代码进行研究，请引用原始论文：

```
@article{papanikolaou2019targetless,
  title={Targetless rotational auto-calibration of radar and camera for intelligent transportation systems},
  author={Papanikolaou, Odysseas and Hampali, Shreyas and Bj\"{o}rnson, Linus and Wein, Friedrich and Ruppel, Andreas},
  journal={arXiv preprint arXiv:1904.08743},
  year={2019}
}
``` 