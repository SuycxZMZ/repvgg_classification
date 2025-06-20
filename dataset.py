from PIL import Image
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import config

def infer_channels(root_dir):
    """自动推测图片通道（1或3）"""
    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                with Image.open(img_path) as img:
                    return 1 if img.mode == 'L' else 3
    return 3  # 默认RGB

def get_dataloaders():
    channels = infer_channels(config['train_dir'])
    print(f"✅ 检测到输入通道数: {channels}（1=灰度，3=RGB）")

    normalize_mean = [0.5] * channels
    normalize_std = [0.5] * channels

    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    train_data = datasets.ImageFolder(root=config['train_dir'], transform=train_transform)
    val_data = datasets.ImageFolder(root=config['val_dir'], transform=val_transform)

    pin_memory = True if config['device'] == 'cuda' else False

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=pin_memory)

    return train_loader, val_loader