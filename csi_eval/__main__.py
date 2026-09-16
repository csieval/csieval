"""csi_eval CLI 入口。

用法::

    # 安装后（推荐）
    csi-eval --task prediction --model-bundle ./wifo_bundle --data ./data_pre --out ./results

    # 不安装直接运行
    python -m csi_eval --task prediction --model-bundle ./wifo_bundle --data ./data_pre

    # 或直接运行完整评估
    python -m csi_eval --task prediction --model-bundle ./wifo_bundle --data ./data_pre \\
        --samples 15000 --device cuda --out ./results/my_model
"""

import argparse
import sys
import os

__version__ = "0.2.0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSI Evaluation Framework — feedback compression & prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # CSI 预测任务（默认 joint 模式）
  python -m csi_eval --task prediction --model-bundle ./my_model_bundle \\
      --data ./data_pre --out ./results/my_model

  # CSI 反馈压缩任务
  python -m csi_eval --task feedback --model-bundle ./my_feedback_bundle \\
      --data ./data_feedback/2_6GHz --out ./results/my_feedback

  # 指定评估样本数
  python -m csi_eval --task prediction --model-bundle ./bundle \\
      --data ./data_pre --samples 5000 --device cpu
        """
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--task', type=str, required=True,
                       choices=['prediction', 'feedback'],
                       help='评估任务：prediction (空频预测) 或 feedback (CSI反馈压缩)')
    parser.add_argument('--model-bundle', type=str, required=True,
                       help='模型 bundle 目录（需含 best.pth + model_meta.json）')
    parser.add_argument('--data', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--out', '--output', dest='out_dir', type=str, default='./results',
                       help='输出目录（默认 ./results）')
    parser.add_argument('--samples', '--max-samples', type=int, default=15000,
                       help='最大评估样本数（默认 15000）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='运行设备：cuda 或 cpu（默认 cuda）')
    return parser.parse_args()


def main():
    args = parse_args()

    # 构建相对路径为绝对路径
    bundle = os.path.abspath(args.model_bundle)
    data   = os.path.abspath(args.data)
    out    = os.path.abspath(args.out_dir)

    print(f"[csi_eval] task={args.task}")
    print(f"[csi_eval] model_bundle={bundle}")
    print(f"[csi_eval] data={data}")
    print(f"[csi_eval] out_dir={out}")

    from . import CSIEvaluator

    report = CSIEvaluator(
        task=args.task,
        model_bundle=bundle,
        data=data,
        out_dir=out,
        device=args.device,
        max_samples=args.samples,
    ).run()

    paths = report.save()   # 同时保存 json / html / markdown
    print("\n报告已生成:")
    for fmt, p in paths.items():
        print(f"  [{fmt}] {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
