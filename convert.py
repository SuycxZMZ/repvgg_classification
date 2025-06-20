import torch
import argparse
import os
from model.repvgg import get_RepVGG_func_by_name, repvgg_model_convert

def convert(model_name, input_path, output_path=None, num_classes=2):
    func = get_RepVGG_func_by_name(model_name)
    model = func(deploy=False)  # 构建训练版
    model.linear = torch.nn.Linear(model.linear.in_features, num_classes)  # ✅ 替换为你训练时的头

    checkpoint = torch.load(input_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)  # 权重正确加载
    print(f"✅ Loaded checkpoint from {input_path}")

    deploy_model = repvgg_model_convert(model)  # 转为 deploy

    # 自动保存路径（未指定）
    if output_path is None:
        dir_path = os.path.dirname(input_path)
        output_path = os.path.join(dir_path, 'deploy.pth')
        print(f"ℹ️ 未指定输出路径，自动保存至: {output_path}")

    torch.save(deploy_model.state_dict(), output_path)
    print(f"✅ Saved deploy model to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RepVGG 推理模型转换')
    parser.add_argument('--model', type=str, required=True, help='模型名称，如 RepVGG-A0')
    parser.add_argument('--input', type=str, required=True, help='训练权重路径 (best.pth)')
    parser.add_argument('--output', type=str, default=None, help='推理权重保存路径（默认与输入同目录）')
    parser.add_argument('--num_classes', type=int, default=2, help='分类类别数')
    args = parser.parse_args()

    convert(args.model, args.input, args.output, num_classes=args.num_classes)