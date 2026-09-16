"""csi_eval — Unified CSI Evaluation Framework.

用户统一入口：``CSIEvaluator`` + ``CSIEvalReport``。

支持的评估任务
----------------
task="prediction"  — CSI 时域/频域插值与外推
    对应子包: csi_eval.pre_eval (CSIPreEvaluator)
    数据格式: data_pre/generated_scenario_{1,2}_*/

task="feedback"    — CSI 特征向量反馈压缩
    对应子包: csi_eval.feedback_eval (CSIFeedbackEvaluator)
    数据格式: data_feedback/DATA_H*.npy

使用示例
--------
    from csi_eval import CSIEvaluator

    # CSI 预测任务
    report = CSIEvaluator(
        task="prediction",
        model_bundle="./my_wifo_bundle",   # 含 best.pth + model_meta.json
        data="./data_pre",                 # 含 generated_scenario_1_* / generated_scenario_2_*
        out_dir="./results/my_model",
    ).run()

    report.save()                          # 同时保存 json / html / markdown

    # CSI 反馈压缩任务
    report = CSIEvaluator(
        task="feedback",
        model_bundle="./my_csinet_bundle",
        data="./data_feedback/2_6GHz",
        out_dir="./results/my_feedback_model",
    ).run()

    report.save()

CLI 用法（需安装可视化依赖）
---------------------------
    pip install "csi-eval[vis]"
    python -m csi_eval --task prediction --model-bundle ./wifo_bundle --data ./data_pre --out ./results

"""

from .evaluator import CSIEvaluator, CSIEvalReport

__version__ = "0.2.0"
__all__ = ["CSIEvaluator", "CSIEvalReport"]
