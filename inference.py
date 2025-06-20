import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from model.repvgg import build_model
from utils import print_model_info
from config import config
from dataset import infer_channels
from model.repvgg import get_RepVGG_func_by_name
from PIL import Image

# --------------------------------------------------------
# RepVGG 推理脚本
# 支持单张图片或文件夹批量推理，自动输出类别和置信度
# 支持与训练一致的预处理
# --------------------------------------------------------

def prepare_transform(image_size, channels):
    """推理用transform（与训练一致）"""
    normalize_mean = [0.5] * channels
    normalize_std = [0.5] * channels

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

def infer_single_image(model, image_path, transform):
    """单张图片推理"""
    image = default_loader(image_path)  # PIL.Image
    image = transform(image).unsqueeze(0).to(config['device'])  # 添加batch维度
    model.eval()
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    return pred, conf

def infer_path(input_path):
    channels = infer_channels(config['train_dir'])
    transform = prepare_transform(config['image_size'], channels)

    func = get_RepVGG_func_by_name(config['model_name'])
    model = func(deploy=config['deploy'])
    model.linear = nn.Linear(model.linear.in_features, config['num_classes'])

    checkpoint = torch.load(config['pretrained_path'], map_location=config['device'])
    if config['deploy']:
        model.load_state_dict(checkpoint)
        print(f"✅ [Deploy] Loaded deploy model from {config['pretrained_path']}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ [Train] Loaded training checkpoint from {config['pretrained_path']}")

    model.to(config['device'])
    model.eval()

    classes = config['class_names']  # 显式类别（如 ['cat', 'dog']）

    def infer_single_image(img_path):
        img = Image.open(img_path).convert('RGB' if channels == 3 else 'L')
        img = transform(img).unsqueeze(0).to(config['device'])
        with torch.no_grad():
            output = model(img)
            prob = torch.softmax(output, dim=1)  # 计算置信度（softmax概率）
            conf, pred = torch.max(prob, 1)      # 取置信度及类别索引
            return classes[pred.item()], conf.item()  # 返回类别名与置信度

    if os.path.isfile(input_path):
        cls, conf = infer_single_image(input_path)
        print(f"🖼️ {input_path} => 预测类别: {cls} | 置信度: {conf:.4f}")
    elif os.path.isdir(input_path):
        img_paths = [os.path.join(input_path, img) for img in os.listdir(input_path)
                     if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"📂 文件夹推理，共 {len(img_paths)} 张图片：")

        results = []  # 收集推理结果
        for img_path in tqdm(img_paths, desc="推理中", unit="img"):
            cls, conf = infer_single_image(img_path)
            results.append((os.path.basename(img_path), cls, conf))

        print("\n📋 推理结果（图片名 | 类别 | 置信度）：")
        print("=" * 60)
        for img_name, cls, conf in results:
            print(f"{img_name:30s}  ==>  {cls:15s} | 置信度: {conf:.4f}")
        print("=" * 60)
    else:
        raise ValueError(f"❌ 输入路径 {input_path} 不是有效的文件或文件夹！")