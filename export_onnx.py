import torch
import argparse
import os
from model.repvgg import get_RepVGG_func_by_name

def export_to_onnx(args):
    """
    加载PyTorch部署模型并将其导出为ONNX格式。
    """
    # --- 1. 参数检查与路径设置 ---
    output_path = args.output
    if output_path is None:
        output_path = os.path.splitext(args.weights)[0] + ".onnx"
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")

    device = torch.device("cpu")
    # ==== FIX: 将device转换为字符串 ====
    print(f"ℹ️ 使用设备: {str(device).upper()}")  # 修复行
    print(f"📊 模型: {args.model}, 类别数: {args.num_classes}, 图像尺寸: {args.image_size}x{args.image_size}")

    # --- 2. 加载部署模型 ---
    print("="*40)
    print("🔎 正在加载部署模型 (deploy=True)...")
    
    if not os.path.exists(args.weights):
        print(f"❌ 错误: 找不到部署权重文件 {args.weights}")
        return

    deploy_model_func = get_RepVGG_func_by_name(args.model)
    model = deploy_model_func(deploy=True)
    model.linear = torch.nn.Linear(model.linear.in_features, args.num_classes)
    
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ 成功加载部署权重: {args.weights}")
    print("="*40)

    # --- 3. 导出为ONNX ---
    print("🚀 开始导出到 ONNX...")
    dummy_input = torch.randn(1, 3, args.image_size, args.image_size, device=device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input' : {0 : 'batch_size'}, 
                          'output' : {0 : 'batch_size'}}
        )
        print(f"\n🎉 成功！模型已导出到: {output_path}")
        print("ℹ️ 您现在可以在C++项目中使用此 .onnx 文件了。")

    except Exception as e:
        print(f"❌ 导出失败: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将RepVGG部署模型转换为ONNX格式')
    parser.add_argument('--model', type=str, required=True, help='模型名称，如 RepVGG-A0')  # 改为必填
    parser.add_argument('--weights', type=str, required=True, help='部署权重路径 (deploy.pth)')  # 改为必填
    parser.add_argument('--output', type=str, default=None, help='ONNX模型保存路径 (默认与输入同目录)')
    parser.add_argument('--num_classes', type=int, default=2, help='分类类别数')
    parser.add_argument('--image_size', type=int, default=224, help='输入图像尺寸')
    args = parser.parse_args()

    export_to_onnx(args)
