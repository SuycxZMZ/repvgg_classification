import torch
from torch import nn, optim
from tqdm import tqdm
from dataset import get_dataloaders
from model.repvgg import build_model
from utils import accuracy, save_checkpoint, print_model_info, get_checkpoint_dir
from config import config
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------
# RepVGG 训练主流程
# 支持预训练权重加载、自动保存最佳模型
# ---------------------------

def train():
    train_loader, val_loader = get_dataloaders()
    model = build_model(config['model_name'],
                        num_classes=config['num_classes'],
                        pretrained_path=config['pretrained_path'],
                        use_pretrained=config['use_pretrained']).to(config['device'])

    print_model_info(model, image_size=config['image_size'])

    # 只加载训练中断时的checkpoint
    # 如果use_pretrained为False，说明是继续训练
    if config['pretrained_path'] and not config['use_pretrained']:
        checkpoint = torch.load(config['pretrained_path'], map_location=config['device'])
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded training checkpoint from {config['pretrained_path']}, epoch {checkpoint.get('epoch', '?')}, acc {checkpoint.get('accuracy', '?')}")
        else:
            raise ValueError("❌ 加载失败：权重文件不包含 model_state_dict，请检查路径是否为训练checkpoint而非ImageNet预训练权重")

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=config['lr'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['lr'] * 0.01)

    best_acc = 0
    ckpt_dir = get_checkpoint_dir('checkpoints', config['model_name'])

    for epoch in range(config['epochs']):
        model.train()
        total_loss, total_acc = 0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')

        # Warmup逻辑（前三轮）：逐步提升学习率和动量，防止初始震荡
        warmup_epochs = 3
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['lr'] * lr_scale
                param_group['momentum'] = 0.8 + (0.937 - 0.8) * lr_scale  # warmup_momentum=0.8 → momentum=0.937

        for imgs, labels in pbar:
            imgs, labels = imgs.to(config['device']), labels.to(config['device'])
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy(output, labels)
            total_loss += loss.item()
            total_acc += acc
            pbar.set_postfix(loss=loss.item(), acc=acc)

        # 每轮结束后在验证集评估
        val_acc = validate(model, val_loader)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch+1, val_acc, ckpt_dir, config['model_name'])

        scheduler.step()  # cosine调整

# 验证集评估函数
def validate(model, val_loader):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(config['device']), labels.to(config['device'])
            output = model(imgs)
            total_acc += accuracy(output, labels)
    avg_acc = total_acc / len(val_loader)
    print(f"[Val] Accuracy: {avg_acc:.4f}")
    return avg_acc