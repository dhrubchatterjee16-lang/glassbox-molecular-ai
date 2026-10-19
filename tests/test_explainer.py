"""
Tests for GNNExplainer Substructure & Motif Attribution
"""

import unittest
from glassbox_molecular.explainer import GNNExplainerSubstructure

class TestExplainer(unittest.TestCase):
    def test_carboxylic_acid_attribution(self):
        # Benzoic acid (c1ccccc1C(=O)O) has carboxylic acid and aromatic ring
        explainer = GNNExplainerSubstructure(target_property="Aqueous Solubility (logS)")
        res = explainer.explain_molecule("c1ccccc1C(=O)O")
        self.assertEqual(res.smiles, "c1ccccc1C(=O)O")
        self.assertGreater(res.total_atoms, 0)
        self.assertEqual(len(res.atom_level_attribution_scores), res.total_atoms)
        # Should identify functional motifs
        motif_names = [m["motif"] for m in res.identified_functional_motifs]
        self.assertTrue(any("Carboxylic" in m or "Aromatic" in m for m in motif_names))

if __name__ == "__main__":
    unittest.main()
