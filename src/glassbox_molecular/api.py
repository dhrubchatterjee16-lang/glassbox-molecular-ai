"""
FastAPI Microservice for GlassBox Molecular AI
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .explainer import GNNExplainerSubstructure
from .fidelity import evaluate_explanation_fidelity

if HAS_FASTAPI:
    app = FastAPI(
        title="GlassBox Molecular AI API",
        description="GNN Substructure Attribution & Mechanistic Explainability Service",
        version="1.0.0"
    )

    class ExplainRequest(BaseModel):
        smiles: str = Field(..., description="SMILES chemical representation")
        target_property: str = Field("Aqueous Solubility (logS)")

    @app.get("/")
    def root():
        return {
            "service": "GlassBox Molecular AI",
            "version": "1.0.0",
            "status": "operational"
        }

    @app.post("/xai/explain")
    def api_explain(req: ExplainRequest):
        explainer = GNNExplainerSubstructure(target_property=req.target_property)
        res = explainer.explain_molecule(req.smiles)
        fid = evaluate_explanation_fidelity(res.atom_level_attribution_scores)
        return {
            "smiles": res.smiles,
            "target_property": res.target_property,
            "total_atoms": res.total_atoms,
            "dominant_motif": res.dominant_motif,
            "motifs": res.identified_functional_motifs,
            "atom_attributions": res.atom_level_attribution_scores,
            "fidelity": {
                "fidelity_plus": fid.fidelity_plus,
                "fidelity_minus": fid.fidelity_minus,
                "sparsity": fid.sparsity,
                "is_valid": fid.is_valid_explanation
            }
        }
else:
    app = None
