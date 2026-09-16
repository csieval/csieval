"""CSIEvaluator — 统一 CSI 评估门面。

调用示例
--------
::

    from csi_eval import CSIEvaluator

    # 反馈压缩任务
    report = CSIEvaluator(
        task="feedback",
        out_dir="results/feedback",
        device="cuda",
        data="data_feedback/2_6GHz",
        model_bundle="ev_csinet_bundle/",
    ).run()
    report.save()

    # 空频预测任务
    report = CSIEvaluator(
        task="prediction",
        out_dir="results/prediction",
        device="cuda",
        data="data_pre",
        model_bundle="wifo_bundle",
    ).run()
    report.save()

设计说明
--------
- 本类只承担**路由 + 报告归一化**的职责，所有评估逻辑均委托给子评估包。
- 子评估包内部的所有参数（mask_mode / mask_ratio / 任务指标开关等）均
  取各自默认，**不再向外暴露**。
- 子任务名称（feedback 包的 ``eigenvector_feedback``，pre 包的
  ``joint``）在 ``_SUBTASK`` 中固定，不暴露。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────
# 内部常量：暴露给用户的大任务（task） ↔ 子评估包子任务（subtask）映射
# ──────────────────────────────────────────────────────────────────────

# 暴露给用户的顶层 task 名称。
TASK_FEEDBACK = "feedback"
TASK_PREDICTION = "prediction"

# 每个顶层 task 对应的子评估包"子任务名"，是子评估包内部要求的细分项。
# 这些名字在子评估包内是固定的，不通过本接口暴露。
_SUBTASK: Dict[str, str] = {
    TASK_FEEDBACK: "eigenvector_feedback",   # csi_feedback_eval 唯一子任务
    TASK_PREDICTION: "joint",                # csi_pre_eval 唯一子任务
}


def _subtask_for(task: str) -> str:
    """返回顶层 task 对应的子评估包子任务名。"""
    if task not in _SUBTASK:
        raise ValueError(
            f"Unknown task={task!r}. Expected one of {list(_SUBTASK.keys())}."
        )
    return _SUBTASK[task]


# ──────────────────────────────────────────────────────────────────────
# 归一化报告：把两种评估包各自异构的输出，包装成一种最小的统一接口
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CSIEvalReport:
    """统一评估报告。

    两种评估包的"原生 report"在结构上差异很大（feedback 包返回 ``EvalReport``
    对象，可直接 ``save("json"/"html"/"markdown")``；pre 包返回 dict，需配合
    ``EvalReporter``）。本类把"原生结果"原封不动地挂在 ``raw`` 上，再叠加
    一层统一的 ``save()`` 接口，便于上层调用方不必关心细节。
    """

    task: str
    subtask: str
    raw: Any                                # 子评估包返回的原始结果
    output_dir: str

    # ---------- 反馈压缩任务：原生 EvalReport 的便捷方法 ----------
    def _is_feedback(self) -> bool:
        return self.task == TASK_FEEDBACK

    # ---------- 空频预测任务：原生 dict + EvalReporter 封装 ----------
    def _is_prediction(self) -> bool:
        return self.task == TASK_PREDICTION

    def save(self, fmt: Optional[str] = None, output_dir: Optional[str] = None) -> Any:
        """统一的报告保存入口。

        Parameters
        ----------
        fmt:
            - feedback 任务：``None`` 或 ``"json"|"html"|"markdown"``；
              ``None`` 等价于一次性保存全部三种。
            - prediction 任务：忽略，统一保存全部格式（JSON + HTML + Markdown）。
        output_dir:
            覆盖报告输出目录，默认使用构造时的 ``output_dir``。
        """
        out_dir = output_dir or self.output_dir
        # EvalReporter.save_all 内部会 os.makedirs(self.output_dir)，此处不重复创建

        if self._is_feedback():
            # csi_feedback_eval.EvalReport.save(fmt, output_dir=...)
            if fmt is None:
                self.raw.save("json", output_dir=out_dir)
                self.raw.save("html", output_dir=out_dir)
                self.raw.save("markdown", output_dir=out_dir)
            else:
                self.raw.save(fmt, output_dir=out_dir)
            return None

        if self._is_prediction():
            # self.raw 是 csi_pre_eval 包的 EvalReport（来自 CSIPreEvaluator.report）
            # 直接透传给 EvalReport.save()，与 feedback 完全一致
            if fmt is None:
                self.raw.save("json", output_dir=out_dir)
                self.raw.save("html", output_dir=out_dir)
                self.raw.save("markdown", output_dir=out_dir)
            else:
                self.raw.save(fmt, output_dir=out_dir)
            return None

        return None


# ──────────────────────────────────────────────────────────────────────
# CSIEvaluator：统一门面
# ──────────────────────────────────────────────────────────────────────

class CSIEvaluator:
    """CSI 统一评估器。

    Parameters
    ----------
    task:
        ``"feedback"`` 或 ``"prediction"``。
    out_dir:
        报告输出目录。
    device:
        ``"cuda"`` 或 ``"cpu"``。``"cuda"`` 在不可用时自动回落 ``"cpu"``。
    data:
        评估数据路径。对 ``feedback`` 是数据目录（含 ``DATA_H*.npy`` 的
        场景目录），对 ``prediction`` 是 ``data_pre`` 根目录。
    model_bundle:
        模型 bundle 目录路径，由子评估包按各自约定解析。

    Notes
    -----
    所有其他参数（mask_mode / mask_ratio / 评估样本数 / 各模块开关等）
    在子评估包中固定为其默认值，**本接口不暴露**。如需修改，请直接使用
    对应的子评估包 API。
    """

    def __init__(
        self,
        task: str,
        out_dir: str = "./results",
        device: str = "cuda",
        data: str = "",
        model_bundle: str = "",
        max_samples: int = 2000,
    ):
        """统一 CSI 评估门面。

        Parameters
        ----------
        task:
            ``"prediction"`` — CSI 时域/频域插值外推（对应 WiFo 等模型）
            ``"feedback"``   — CSI 特征向量反馈压缩（对应 CsiNet 等模型）
        out_dir:
            报告输出目录，默认 ``"./results"``。
            运行后会在该目录下生成 ``metrics.json`` / ``report.html`` / ``report.md``。
        device:
            推理设备，``"cuda"`` 或 ``"cpu"``，默认 ``"cuda"``。
        data:
            数据目录。prediction 任务需包含 ``generated_scenario_1_*`` 和
            ``generated_scenario_2_*`` 子目录；feedback 任务需包含 ``DATA_H*.npy`` 文件。
        model_bundle:
            模型 bundle 目录，需含 ``best.pth``（或同义权重文件）+ ``model_meta.json``。
        max_samples:
            最大评估样本数，默认 15000。

        Notes
        -----
        所有其他参数（mask_mode / mask_ratio / 各评估模块开关等）
        在子评估包中固定为其默认值，**本接口不暴露**。如需修改，
        请直接使用对应的子评估包 API（``CSIPreEvaluator`` / ``CSIFeedbackEvaluator``）。
        """
        if task not in _SUBTASK:
            raise ValueError(
                f"Invalid task={task!r}. Expected one of {list(_SUBTASK.keys())}."
            )
        self.task = task
        self.subtask = _subtask_for(task)
        self.out_dir = out_dir
        self.device = device
        self.data = data
        self.model_bundle = model_bundle
        self.max_samples = max_samples
        # 目录创建统一由子评估包负责，此处不提前 mkdir（避免与子包默认目录冲突）

    # ------------------------------------------------------------------
    # 路由分发
    # ------------------------------------------------------------------
    def run(self) -> CSIEvalReport:
        """分发到对应子评估包并执行完整评估。"""
        if self.task == TASK_FEEDBACK:
            raw = self._run_feedback()
            return CSIEvalReport(
                task=self.task,
                subtask=self.subtask,
                raw=raw,
                output_dir=self.out_dir,
            )

        if self.task == TASK_PREDICTION:
            raw = self._run_prediction()
            return CSIEvalReport(
                task=self.task,
                subtask=self.subtask,
                raw=raw,
                output_dir=self.out_dir,
            )

        raise ValueError(f"Unhandled task: {self.task!r}")  # 不可能到达

    # ------------------------------------------------------------------
    # feedback 分支：完全沿用 csi_feedback_eval 的全部默认行为
    # ------------------------------------------------------------------
    def _run_feedback(self):
        from csi_eval.feedback_eval import Evaluator

        # 在 test.py 中暴露的写法：
        #   Evaluator(task="eigenvector_feedback",
        #             model_bundle=...,
        #             data=...,
        #             output_dir=...,
        #            ).run()
        ev = Evaluator(
            task=self.subtask,                  # 固定 "eigenvector_feedback"
            model_bundle=self.model_bundle,
            data=self.data,
            output_dir=self.out_dir,
            device=self.device,
        )
        return ev.run()

    # ------------------------------------------------------------------
    # prediction 分支：完全沿用 csi_pre_eval 的全部默认行为
    # ------------------------------------------------------------------
    def _run_prediction(self):
        from csi_eval.pre_eval import CSIPreEvaluator, EvalConfig as PreEvalConfig

        # 完整沿用 test_pre/test.py 中的固定默认设置（不暴露）。
        eval_cfg = PreEvalConfig()              # 全部取 csi_pre_eval 默认值
        evaluator = CSIPreEvaluator(
            output_dir=self.out_dir,
            device=self.device,
            max_samples=self.max_samples,
            eval_cfg=eval_cfg,
            s1_data_dir=self.data,
            s2_data_dir=self.data,
        )
        # 子任务类型（"wifo" / "csinet" / ...）无法从 model_bundle 自动推断，
        # 这里以 bundle 的目录名（去掉可能的路径后缀）作为 model_type。
        model_type = os.path.basename(os.path.normpath(self.model_bundle))
        evaluator.add_model(
            model_type=model_type,
            checkpoint=self.model_bundle,
            alias=model_type,
        )
        evaluator.run_all()
        evaluator.summary()
        # 覆盖 task 字段，使报告标题显示为 "CSI Prediction Evaluation Report"
        evaluator._report.meta["task"] = "prediction"
        return evaluator.report
