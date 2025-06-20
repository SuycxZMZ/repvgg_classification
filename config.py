import torch
# config = {
#     'train_dir': '/Users/yuansu/Desktop/codes/datasets/cats_and_dogs/train',
#     'val_dir': '/Users/yuansu/Desktop/codes/datasets/cats_and_dogs/val',
#     'num_classes': 2,  # 改为实际类别数
#     'image_size': 224,
#     'batch_size': 32,
#     'epochs': 50,
#     'lr': 0.001,
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
# }

config = {
    'train_dir': '/Users/yuansu/Desktop/codes/datasets/cats_and_dogs/train',
    'val_dir': '/Users/yuansu/Desktop/codes/datasets/cats_and_dogs/val',
    'num_classes': 2, 
    'class_names': ['cat', 'dog'],  # 自己明确指定类别
    'image_size': 224,
    'batch_size': 32,
    'epochs': 50,
    # 'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'mps',
    'model_name': 'RepVGG-A0',  # 模型选择 RepVGG-A0
    # 'model_name': 'RepVGG-A1',  # 模型选择 RepVGG-A1, 
    # 'model_name': 'RepVGG-A2',  # 模型选择 RepVGG-A2, 
    # 'model_name': 'RepVGG-A3',  # 模型选择 RepVGG-A3, 
    # 'model_name': 'RepVGG-B0',  # 模型选择 RepVGG-B0, 
    # 'model_name': 'RepVGG-B1',  # 模型选择 RepVGG-B1, 
    # 'model_name': 'RepVGG-B2',  # 模型选择 RepVGG-B2, 
    # 'model_name': 'RepVGG-B2g2',  # 模型选择 RepVGG-B2g4, 
    # 'model_name': 'RepVGG-B3',  # 模型选择 RepVGG-B3, 
    # 'model_name': 'RepVGG-B3g4',  # 模型选择 RepVGG-B3g4, 

    'pretrained_path': './pretrained/RepVGG-A0-train.pth',  # 官方ImageNet权重路径
    'use_pretrained': True,  # 启用预训练加载
    'lr': 0.001,   # or 0.001 (你的小数据集建议0.001)
    'epochs': 100,
    'batch_size': 16,
    'weight_decay': 0.0005,
    'momentum': 0.937,

    'deploy': False,  # 默认为训练权重
}