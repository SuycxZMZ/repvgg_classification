import argparse
from train import train, validate
from inference import infer_path
from config import config

# ---------------------------
# RepVGG 主控脚本
# 用于统一入口管理训练、验证、推理等流程
# 支持命令行参数灵活切换模式和权重路径
# ---------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'validate', 'test', 'inference'], required=True,
                        help="运行模式：train / validate / test / inference")
    parser.add_argument('--input', type=str, default=None,
                        help="推理模式下的图片路径或目录路径（仅inference模式有效）")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="指定加载的训练权重路径（validate/test/inference模式有效）")
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help="指定ImageNet预训练权重路径（train模式有效）")
    parser.add_argument('--deploy', type=bool, default=False,
                    help='是否为deploy模式，True加载转换后的deploy.pth，False加载训练权重')
    args = parser.parse_args()

    # 设置是否为deploy模式（推理结构）
    config['deploy'] = args.deploy

    # CLI参数优先级覆盖config
    # checkpoint_path优先用于加载训练中断权重
    if args.checkpoint_path:
        config['pretrained_path'] = args.checkpoint_path
        config['use_pretrained'] = False  # 说明此路径用于加载训练checkpoint而非ImageNet预训练
    elif args.pretrained_path:
        config['pretrained_path'] = args.pretrained_path
        config['use_pretrained'] = True  # CLI指定使用ImageNet预训练权重

    # 根据mode选择执行流程
    if args.mode == 'train':
        train()
    elif args.mode in ['validate', 'test']:
        from train import run_validate
        run_validate()
    elif args.mode == 'inference':
        # 推理模式必须指定input路径
        assert args.input is not None, "❌ 请在inference模式下指定 --input 路径！"
        infer_path(args.input)