#!/usr/bin/env python
"""
CSI Pre-Evaluation CLI
======================
用法:
    # 评估所有模型
    python run_eval.py --models all

    # 评估指定模型
    python run_eval.py --models wifo csinet crnet

    # 指定输出目录
    python run_eval.py --models all --output ./results/my_eval

    # 指定评估样本数
    python run_eval.py --models all --max-samples 5000

    # 跳过某些评估模块
    python run_eval.py --models all --skip-robustness --skip-cross-scenario
"""
import os
import sys
import argparse
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description='CSI Pre-Evaluation: 统一评估框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_eval.py --models wifo csinet crnet
  python run_eval.py --models all --output ./results/my_eval
  python run_eval.py --models all --max-samples 5000 --skip-robustness
  python run_eval.py --models wifo --checkpoint ./results/checkpoints/wifo_best.pth
        """
    )

    # 模型
    parser.add_argument('--models', nargs='+', default=['wifo'],
                        help='模型列表，如 wifo csinet crnet，或 all（所有内置模型）')
    parser.add_argument('--all-models', action='store_true',
                        help='等价于 --models all')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='指定模型 checkpoint 路径（单个模型时使用）')

    # 数据 & 评估规模
    parser.add_argument('--max-samples', type=int, default=15000,
                        help='评估最大样本数（默认 15000）')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='推理 batch size（默认 128）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda 或 cpu（默认 cuda）')

    # 评估配置
    parser.add_argument('--task', type=str, default='joint',
                        choices=['frequency', 'spatial', 'joint'],
                        help='预测任务（默认 joint）')
    parser.add_argument('--mask-mode', type=str, default='comb',
                        choices=['comb', 'grid', 'random_2d', 'block',
                                 'random', 'random_subset'],
                        help='掩码模式（默认 comb）')
    parser.add_argument('--mask-ratio', type=float, default=0.5,
                        help='掩码比例（默认 0.5）')

    # 评估模块开关
    parser.add_argument('--skip-task-metrics', action='store_true',
                        help='跳过任务性能指标')
    parser.add_argument('--skip-storage-metrics', action='store_true',
                        help='跳过存储与部署指标')
    parser.add_argument('--skip-compute-metrics', action='store_true',
                        help='跳过计算效率指标')
    parser.add_argument('--skip-robustness', action='store_true',
                        help='跳过泛化鲁棒性（cross-mask / cross-ratio）')
    parser.add_argument('--skip-noise-robustness', action='store_true',
                        help='跳过噪声鲁棒性')
    parser.add_argument('--skip-cross-scenario', action='store_true',
                        help='跳过跨场景评估')

    # 输出
    parser.add_argument('--output', '--output-dir', type=str, default=None,
                        help='结果输出目录（默认 results/pre_eval/）')
    parser.add_argument('--timestamp', action='store_true',
                        help='在输出目录名中加上时间戳')

    return parser.parse_args()


def resolve_models(model_names: list, all_available: list) -> list:
    """解析模型列表，all -> 所有可用模型"""
    if not model_names:
        return []
    names = [m.lower() for m in model_names]
    if 'all' in names:
        return all_available
    return names


def build_eval_config(args) -> '.config.EvalConfig':
    """根据 CLI 参数构建 EvalConfig"""
    from . import config as cfg

    out_dir = args.output or cfg.RESULTS_DIR
    if args.timestamp:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(out_dir, f'eval_{ts}')

    return cfg.EvalConfig(
        task=args.task,
        mask_mode=args.mask_mode,
        mask_ratio=args.mask_ratio,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=out_dir,
        run_task_metrics=not args.skip_task_metrics,
        run_storage_metrics=not args.skip_storage_metrics,
        run_compute_metrics=not args.skip_compute_metrics,
        run_robustness=not args.skip_robustness,
        run_noise_robustness=not args.skip_noise_robustness,
        run_cross_scenario=not args.skip_cross_scenario,
    )


def main():
    args = parse_args()

    # 解析模型列表（从 config 而非 model_registry.registry 获取）
    from . import config as cfg
    all_models = cfg.ALL_MODELS
    model_list = resolve_models(args.models, all_models)
    if args.all_models:
        model_list = all_models

    if not model_list:
        print('错误: 未指定任何模型')
        sys.exit(1)

    print(f'待评估模型: {model_list}')
    print(f'可用模型: {all_models}')
    unknown = [m for m in model_list if m not in all_models]
    if unknown:
        print(f'警告: 以下模型未在注册表中: {unknown}')

    # 构建配置
    eval_cfg = build_eval_config(args)

    # 初始化评估器
    from .evaluator import CSIPreEvaluator
    evaluator = CSIPreEvaluator(
        output_dir=eval_cfg.output_dir,
        device=args.device,
        max_samples=args.max_samples,
        eval_cfg=eval_cfg,
    )

    # 注册模型
    for model_name in model_list:
        ckpt = args.checkpoint if len(model_list) == 1 else None
        evaluator.add_model(model_name, checkpoint=ckpt)

    # 运行评估
    results = evaluator.run_all()

    # 打印汇总
    evaluator.summary()

    # 保存报告
    paths = evaluator.save()
    print(f'\n报告已生成:')
    for fmt, path in paths.items():
        print(f'  [{fmt}] {path}')

    return results


if __name__ == '__main__':
    main()
