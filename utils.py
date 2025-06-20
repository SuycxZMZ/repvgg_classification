import torch
from ptflops import get_model_complexity_info
import os

# --------------------------------------------------------
# 工具函数集合
# 包含准确率计算、模型信息打印、checkpoint保存等
# --------------------------------------------------------

def accuracy(preds, labels):
    _, pred_classes = torch.max(preds, dim=1)
    return (pred_classes == labels).float().mean().item()

def get_checkpoint_dir(base_dir, model_name):
    """根据模型名自动生成子文件夹，如 checkpoints/RepVGG-A0_1"""
    idx = 1
    while True:
        folder_name = f"{model_name}_{idx}"
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path
        idx += 1

def save_checkpoint(model, optimizer, epoch, acc, base_dir, model_name):
    """保存checkpoint到指定路径"""
    ckpt_dir = get_checkpoint_dir(base_dir, model_name)
    save_path = os.path.join(ckpt_dir, f'best.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': acc
    }, save_path)
    print(f"✅ Checkpoint 已保存至: {save_path}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model, image_size=224):
    macs, params = get_model_complexity_info(model, (3, image_size, image_size),
                                             as_strings=True, print_per_layer_stat=False)
    print(f"\n📌 Model Summary:")
    print(f"  Total params   : {count_parameters(model):,}")
    print(f"  Computational : {macs}")
    print(f"  Total MACs    : {params}\n")