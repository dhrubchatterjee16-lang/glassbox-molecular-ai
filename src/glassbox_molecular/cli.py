"""
Command-Line Interface for GlassBox Molecular AI
"""

import argparse
from .explainer import GNNExplainerSubstructure
from .fidelity import evaluate_explanation_fidelity

def main():
    parser = argparse.ArgumentParser(
        prog="glassbox",
        description="GlassBox Molecular AI: Mechanistic Explainability & Substructure Attribution"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    explain_parser = subparsers.add_parser("explain", help="Explain molecular prediction with substructure motifs")
    explain_parser.add_argument("--smiles", type=str, default="Cc1ccc(cc1)C(=O)O", help="SMILES string (e.g., p-Toluic acid)")
    explain_parser.add_argument("--property", type=str, default="Aqueous Solubility (logS)")

    args = parser.parse_args()

    if args.command == "explain":
        explainer = GNNExplainerSubstructure(target_property=args.property)
        res = explainer.explain_molecule(args.smiles)
        fid = evaluate_explanation_fidelity(res.atom_level_attribution_scores)

        print("\n--- GlassBox Molecular XAI Explanation ---")
        print(f"SMILES:             {res.smiles}")
        print(f"Target Property:    {res.target_property}")
        print(f"Total Atoms:        {res.total_atoms}")
        print(f"Dominant Motif:     {res.dominant_motif}")
        print(f"Fidelity+ (Suff):   {fid.fidelity_plus}")
        print(f"Fidelity- (Nec):    {fid.fidelity_minus}")
        print(f"Subgraph Sparsity:  {fid.sparsity}")
        print(f"Explanation Valid:  {'PASS' if fid.is_valid_explanation else 'FAIL'}")
        print("\nIdentified Functional Motifs:")
        for m in res.identified_functional_motifs:
            print(f"  * {m['motif']} (Atoms {m['atom_indices']}): {m['directional_impact']} (Weight: {m['attribution_magnitude']})")
        print()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
