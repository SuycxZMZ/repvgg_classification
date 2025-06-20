import torch
import argparse
import time
import numpy as np
import os
from model.repvgg import get_RepVGG_func_by_name

def run_benchmark(model_name, model, device, image_size, warmup_runs=20, benchmark_runs=100):
    """
    运行指定模型的基准测试。

    :param model_name: 模型的可读名称（例如 "训练模型"）
    :param model: 要测试的 PyTorch 模型
    :param device: 运行设备 ('cpu' or 'cuda')
    :param image_size: 输入图像的尺寸
    :param warmup_runs: 预热运行次数
    :param benchmark_runs: 基准测试运行次数
    :return: (平均延迟 ms, 平均吞吐量 FPS)
    """
    model.to(device)
    model.eval()

    # 创建虚拟输入数据
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    # 预热
    print(f"🔥 ({model_name}) 正在预热...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # 同步等待GPU操作完成
    if device.startswith('cuda'):
        torch.cuda.synchronize()

    # 开始基准测试
    print(f"🚀 ({model_name}) 正在运行基准测试...")
    timings = []
    with torch.no_grad():
        for i in range(benchmark_runs):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append(end_time - start_time)

    # 计算并返回结果
    avg_latency_ms = np.mean(timings) * 1000
    fps = 1 / np.mean(timings)
    print(f"✅ ({model_name}) 测试完成!")
    return avg_latency_ms, fps

def benchmark(args):
    """
    主函数，执行模型加载和基准测试流程。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"ℹ️ 使用设备: {device.upper()}")
    print(f"📊 模型: {args.model}, 类别数: {args.num_classes}, 图像尺寸: {args.image_size}x{args.image_size}\n")

    # --- 1. 测试训练时模型 (重参数化前) ---
    print("="*40)
    print("🔎 开始测试训练时模型 (多分支结构)")
    print("="*40)
    train_model_func = get_RepVGG_func_by_name(args.model)
    train_model = train_model_func(deploy=False)
    train_model.linear = torch.nn.Linear(train_model.linear.in_features, args.num_classes)

    if not os.path.exists(args.train_weights):
        print(f"❌ 错误: 找不到训练权重文件 {args.train_weights}")
        return

    checkpoint = torch.load(args.train_weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    train_model.load_state_dict(state_dict)
    print(f"✅ 成功加载训练权重: {args.train_weights}")

    train_latency, train_fps = run_benchmark("训练模型", train_model, device, args.image_size, args.warmup, args.runs)
    print(f"⏱️  训练模型平均延迟: {train_latency:.2f} ms")
    print(f"🚀  训练模型平均吞吐量: {train_fps:.2f} FPS\n")


    # --- 2. 测试部署时模型 (重参数化后) ---
    print("="*40)
    print("🔎 开始测试部署时模型 (单卷积结构)")
    print("="*40)
    deploy_model_func = get_RepVGG_func_by_name(args.model)
    deploy_model = deploy_model_func(deploy=True)
    deploy_model.linear = torch.nn.Linear(deploy_model.linear.in_features, args.num_classes)

    if not os.path.exists(args.deploy_weights):
        print(f"❌ 错误: 找不到部署权重文件 {args.deploy_weights}")
        print(f"ℹ️ 您可以使用 `convert.py` 脚本生成部署权重。")
        return

    deploy_model.load_state_dict(torch.load(args.deploy_weights, map_location=device))
    print(f"✅ 成功加载部署权重: {args.deploy_weights}")

    deploy_latency, deploy_fps = run_benchmark("部署模型", deploy_model, device, args.image_size, args.warmup, args.runs)
    print(f"⏱️  部署模型平均延迟: {deploy_latency:.2f} ms")
    print(f"🚀  部署模型平均吞吐量: {deploy_fps:.2f} FPS\n")


    # --- 3. 性能对比 ---
    print("="*50)
    print("🎉 性能对比总结 🎉")
    print("="*50)
    print(f"| {'模型类型':<12} | {'平均延迟 (ms)':<18} | {'吞吐量 (FPS)':<15} |")
    print(f"|{'-'*14}|{'-'*20}|{'-'*17}|")
    print(f"| {'训练模型':<12} | {train_latency:<18.2f} | {train_fps:<15.2f} |")
    print(f"| {'部署模型':<12} | {deploy_latency:<18.2f} | {deploy_fps:<15.2f} |")
    print("-" * 52)
    
    if train_fps > 0:
        speedup_factor = deploy_fps / train_fps
        print(f"\n✨ 重参数化后，模型速度提升了 {speedup_factor:.2f} 倍！")
    
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RepVGG 训练与部署模型推理速度基准测试')
    parser.add_argument('--model', type=str, default='RepVGG-A0', help='模型名称，如 RepVGG-A0')
    parser.add_argument('--train_weights', type=str, default='./checkpoints/RepVGG-A0_4/best.pth', help='训练权重路径 (best.pth)')
    parser.add_argument('--deploy_weights', type=str, default='./checkpoints/RepVGG-A0_4/deploy.pth', help='部署权重路径 (deploy.pth)')
    parser.add_argument('--num_classes', type=int, default=2, help='分类类别数')
    parser.add_argument('--image_size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--warmup', type=int, default=20, help='预热运行次数')
    parser.add_argument('--runs', type=int, default=100, help='基准测试运行次数')
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    benchmark(args) 