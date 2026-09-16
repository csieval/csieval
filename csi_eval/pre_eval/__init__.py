"""
CSI Pre-Evaluation Package
==========================
统一评估框架：支持多模型、多任务、多指标的系统化 CSI 预测评估。

与 csi_eval.feedback_eval 共享同一套 EvalReport 接口，report.save() 完全一致。

调用示例 (test.py):
    from csi_pre_eval import CSIPreEvaluator, EvalConfig

    evaluator = CSIPreEvaluator(
        output_dir='./results/my_eval',
        device='cuda',
        max_samples=1000,
    )
    evaluator.add_model('wifo', checkpoint='path/to/best.pth')
    evaluator.add_model('csinet', checkpoint='path/to/best.pth')
    evaluator.run_all()

    # 与 csi_eval.feedback_eval 完全统一的报告接口
    evaluator.report.save("json")
    evaluator.report.save("html")
    evaluator.report.save("markdown")
"""

from .evaluator import CSIPreEvaluator
from .config import EvalConfig

__all__ = ['CSIPreEvaluator', 'EvalConfig']