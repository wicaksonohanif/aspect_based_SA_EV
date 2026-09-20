"""
Labeling & Weak Supervision Module
Spec Compliance: specs/02_preprocessing_labeling.spec.md
"""

from .gemini_labeler import GeminiAspectLabeler
from .weak_supervision import WeakSupervisionEngine
from .eval_kappa import CohenKappaEvaluator
from .dataset_splitter import DatasetSplitter

__all__ = [
    "GeminiAspectLabeler",
    "WeakSupervisionEngine",
    "CohenKappaEvaluator",
    "DatasetSplitter",
]
