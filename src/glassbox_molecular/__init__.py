"""
GlassBox Molecular AI: Mechanistic Explainable AI (XAI) & Substructure Attribution Engine
"""

from .explainer import GNNExplainerSubstructure, SubstructureAttributionResult
from .fidelity import evaluate_explanation_fidelity, FidelityMetrics

__version__ = "1.0.0"
__all__ = [
    "GNNExplainerSubstructure",
    "SubstructureAttributionResult",
    "evaluate_explanation_fidelity",
    "FidelityMetrics",
    "__version__"
]
