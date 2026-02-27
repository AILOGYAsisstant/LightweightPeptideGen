#!/usr/bin/env python
"""Generate diagnostic plots from a FASTA of generated peptides.

This script recomputes evaluation metrics (stability, diversity, AA distribution)
using the package `src.evaluation` and saves the same diagnostic plots used
by `evaluate.py --plots`.
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from peptidegen.evaluation import (
    PeptideStabilityAnalyzer,
    calculate_diversity_metrics,
    calculate_amino_acid_distribution,
)


def read_fasta(fp: Path):
    seqs = []
    with open(fp, 'r') as f:
        cur = None
        buf = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur is not None:
                    seqs.append((''.join(buf)).upper())
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur is not None:
            seqs.append((''.join(buf)).upper())
    return seqs


def save_plots(results: dict, out_dir: Path, threshold: float = 40.0):
    out_dir.mkdir(parents=True, exist_ok=True)

    stability = results.get('stability', {})
    metrics = stability.get('metrics', [])

    if metrics:
        instability_vals = [m.get('instability_index', 0.0) for m in metrics]
        gravy_vals = [m.get('gravy', 0.0) for m in metrics]
        lengths = [m.get('length', 0) for m in metrics]

        plt.figure(figsize=(6,4))
        plt.hist(instability_vals, bins=30, color='C0', alpha=0.8)
        plt.axvline(x=threshold, color='r', linestyle='--')
        plt.xlabel('Instability Index')
        plt.ylabel('Count')
        plt.title('Instability Index Distribution')
        plt.tight_layout()
        plt.savefig(out_dir / 'instability_hist.png')
        plt.close()

        plt.figure(figsize=(6,4))
        plt.hist(gravy_vals, bins=30, color='C1', alpha=0.8)
        plt.xlabel('GRAVY')
        plt.ylabel('Count')
        plt.title('GRAVY Distribution')
        plt.tight_layout()
        plt.savefig(out_dir / 'gravy_hist.png')
        plt.close()

        plt.figure(figsize=(6,4))
        plt.hist(lengths, bins=30, color='C2', alpha=0.8)
        plt.xlabel('Sequence Length')
        plt.ylabel('Count')
        plt.title('Sequence Length Distribution')
        plt.tight_layout()
        plt.savefig(out_dir / 'length_hist.png')
        plt.close()

        plt.figure(figsize=(6,4))
        plt.scatter(instability_vals, gravy_vals, s=10, alpha=0.6)
        plt.xlabel('Instability Index')
        plt.ylabel('GRAVY')
        plt.title('Instability vs GRAVY')
        plt.tight_layout()
        plt.savefig(out_dir / 'instability_vs_gravy.png')
        plt.close()

    # AA distribution
    aa_dist = results.get('aa_distribution', {})
    if aa_dist:
        aas = sorted([k for k in aa_dist.keys() if k != 'non_standard'])
        freqs = [aa_dist[a]['frequency'] for a in aas]
        plt.figure(figsize=(10,4))
        plt.bar(aas, freqs, color='C3')
        plt.xlabel('Amino Acid')
        plt.ylabel('Frequency')
        plt.title('Amino Acid Distribution')
        plt.tight_layout()
        plt.savefig(out_dir / 'aa_distribution.png')
        plt.close()

    # Diversity summary
    diversity = results.get('diversity', {})
    if diversity:
        keys = ['uniqueness_ratio', 'bigram_diversity', 'trigram_diversity', 'diversity_score']
        vals = [diversity.get(k, 0.0) for k in keys]
        plt.figure(figsize=(6,4))
        plt.bar(keys, vals, color='C4')
        plt.ylabel('Value')
        plt.title('Diversity Metrics')
        plt.tight_layout()
        plt.savefig(out_dir / 'diversity_metrics.png')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot evaluation results from FASTA')
    parser.add_argument('--fasta', type=str, required=True, help='Input generated FASTA')
    parser.add_argument('--reference', type=str, default=None, help='Optional reference FASTA for comparison')
    parser.add_argument('--outdir', type=str, default='results/plots', help='Output directory for plots')
    parser.add_argument('--threshold', type=float, default=40.0, help='Instability threshold')

    args = parser.parse_args()
    fasta = Path(args.fasta)
    outdir = Path(args.outdir)

    if not fasta.exists():
        print(f"FASTA not found: {fasta}")
        return

    sequences = read_fasta(fasta)

    analyzer = PeptideStabilityAnalyzer(stability_threshold=args.threshold)
    stability = analyzer.analyze_batch(sequences)
    diversity = calculate_diversity_metrics(sequences)
    aa_distribution = calculate_amino_acid_distribution(sequences)

    results = {
        'stability': stability,
        'diversity': diversity,
        'aa_distribution': aa_distribution,
    }

    # If reference provided, compute comparison (best-effort)
    if args.reference:
        ref_fp = Path(args.reference)
        if ref_fp.exists():
            from peptidegen.evaluation.metrics import compare_distributions
            # read_fasta returns list of sequences
            ref_seqs = read_fasta(ref_fp)
            results['comparison'] = compare_distributions(sequences, ref_seqs)

    save_plots(results, outdir, threshold=args.threshold)
    print(f"Plots written to {outdir}")


if __name__ == '__main__':
    main()
