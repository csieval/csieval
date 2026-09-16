"""CSI_Feedback_Eval — CSI feedback compression model evaluation framework.

A self-contained, pluggable evaluation framework for neural-network-based
CSI feedback compression models used in 5G/6G wireless systems.

Quick start::

    from csi_eval.feedback_eval import Evaluator

    # With model bundle (recommended)
    report = Evaluator(
        task="eigenvector_feedback",
        model_bundle="bundles/my_model/",
        data="data/Dataset/wair_d_output/2_6GHz",
    ).run()

    report.print_summary()
    print(report["sgcs"])

API reference::

    from csi_eval.feedback_eval import Evaluator, EvalConfig, EvalReport, EvalContext

    cfg = EvalConfig(
        task="eigenvector_feedback",
        model_bundle="bundles/my_model/",
        data="data/Dataset/wair_d_output/2_6GHz",
    )
    report = Evaluator(cfg).run()

    report["sgcs"]                 # single metric
    report["ood/part1_new::gap_nmse"]  # OOD sub-result
    report.save("html")            # interactive HTML report
    report.save("json")            # structured JSON
    report.save("markdown")        # markdown table
"""

from .bundle import BundleError, ModelBundle
from .core.config import EvalConfig
from .core.evaluator import Evaluator
from .core.context import EvalContext
from .core.report import EvalReport, MetricRecord
from .core.registries import TaskRegistry, MetricRegistry

__all__ = [
    "Evaluator",
    "EvalConfig",
    "EvalContext",
    "EvalReport",
    "MetricRecord",
    "TaskRegistry",
    "MetricRegistry",
    "ModelBundle",
    "BundleError",
]
