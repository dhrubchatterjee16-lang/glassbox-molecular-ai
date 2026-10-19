# GlassBox Molecular AI: Mechanistic Explainability & Substructure Attribution

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-glassbox--molecular--ai-yellow)](https://huggingface.co/spaces/dhrubchatterjee/glassbox-molecular-ai)
[![Tests: Passing](https://img.shields.io/badge/tests-3%2F3%20passed-brightgreen.svg)](tests/)

**GlassBox Molecular AI** is a mechanistic Explainable AI (XAI) framework for molecular Graph Neural Networks (GNNs). It identifies the causal chemical functional groups and atom-level attributions governing property predictions, accompanied by quantitative **Fidelity+** (sufficiency) and **Fidelity-** (necessity) benchmark evaluations.

---

## 🔬 Scientific Foundations & Governing Equations

### 1. Information-Theoretic Subgraph Masking (GNNExplainer)
Extracts a compact explanatory subgraph $G_S$ that maximizes mutual information with the target molecular property prediction $\hat{Y}$:

$$\max_{G_S} \mathrm{MI}(Y, G_S) = H(Y) - H(Y \mid G = G_S)$$

Optimization balances explanatory coverage with subgraph sparsity penalties:

$$\mathcal{L} = -\sum_{c=1}^C \mathbb{I}(y = c) \log P_{\Phi}(Y = c \mid G = G_S) + \lambda_1 \|M_E\|_1 + \lambda_2 H(M_E)$$

### 2. Quantitative Faithfulness Metrics
- **Fidelity+ (Sufficiency):** Measures prediction degradation when non-explanatory atoms are pruned:
  $$\mathrm{Fid}_+ = \frac{1}{N} \sum_{i=1}^N \left| \hat{y}_i - \hat{y}_i^{G \setminus G_S} \right|$$
- **Fidelity- (Necessity):** Measures prediction drop when key explanatory functional motifs are masked:
  $$\mathrm{Fid}_- = \frac{1}{N} \sum_{i=1}^N \left| \hat{y}_i - \hat{y}_i^{G_S} \right|$$
- **Sparsity:** Quantifies the proportion of non-essential atoms removed:
  $$\mathrm{Sparsity} = 1 - \frac{|V_S|}{|V|}$$

---

## ⚡ Quickstart

### Installation

```bash
pip install -e .
```

### Python API

```python
from glassbox_molecular import GNNExplainerSubstructure, evaluate_explanation_fidelity

# 1. Identify Explanatory Substructures Driving Molecular Solubility
explainer = GNNExplainerSubstructure(target_property="Aqueous Solubility (logS)")
res = explainer.explain_molecule("Cc1ccc(cc1)C(=O)O") # p-Toluic acid

print(f"Total Atoms: {res.total_atoms}")
print(f"Dominant Driver: {res.dominant_motif}")
for m in res.identified_functional_motifs:
    print(f"  * Motif: {m['motif']} -> {m['directional_impact']} (Weight: {m['attribution_magnitude']})")

# 2. Evaluate Faithfulness & Fidelity Metrics
fid = evaluate_explanation_fidelity(res.atom_level_attribution_scores)
print(f"Fidelity+ (Sufficiency): {fid.fidelity_plus}")
print(f"Fidelity- (Necessity): {fid.fidelity_minus}")
print(f"Subgraph Sparsity: {fid.sparsity} (Valid Explanation: {fid.is_valid_explanation})")
```

---

## 🖥️ Command-Line Interface (CLI)

```bash
# Explain prediction for aspirin or drug-like molecule
glassbox explain --smiles "CC(=O)Oc1ccccc1C(=O)O" --property "logS"
```

---

## 🧪 Automated Verification Suite

```bash
python run_tests.py
```
```text
Ran 3 tests in 0.090s
OK - ALL GLASSBOX-MOLECULAR-AI TESTS PASSED (3/3)!
```

---

## 📚 References
- Ying, Z., et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks*. Advances in Neural Information Processing Systems (NeurIPS 2019).
- Yuan, H., et al. (2022). *Explainability in graph neural networks: A taxonomic survey*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5), 5782-5799.
- Sanchez-Lengeling, B., et al. (2020). *Evaluating attribution for graph neural networks*. Advances in Neural Information Processing Systems.

---

## 📄 License
MIT License. Copyright (c) 2026 Dhrub Chatterjee.
