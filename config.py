import torch

# --------------------------------------------------------
# 配置文件
# 用于集中管理训练、验证、推理等所有超参数和路径
# 新手建议只需修改本文件即可完成大部分实验配置
# --------------------------------------------------------

config = {
    'train_dir': '/Users/yuansu/Desktop/codes/datasets/cats_and_dogs/train',  # 训练集路径
    'val_dir': '/Users/yuansu/Desktop/codes/datasets/cats_and_dogs/val',      # 验证集路径
    'num_classes': 2, 
    'class_names': ['cat', 'dog'],  # 明确指定类别名
    'image_size': 224,              # 输入图片尺寸
    'batch_size': 32,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'mps',  # 自动选择设备
    'model_name': 'RepVGG-A0',      # 可选模型
    'pretrained_path': './pretrained/RepVGG-A0-train.pth',     # 预训练权重路径
    'use_pretrained': True,         # 是否加载预训练
    'lr': 0.001,                    # 学习率
    'weight_decay': 0.0005,
    'momentum': 0.937,
    'deploy': False,                # 是否为推理权重
}