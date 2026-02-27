#!/usr/bin/env python
"""
Evaluate generated peptide sequences.

Usage:
    python evaluate.py --input generated.fasta
    python evaluate.py --input generated.fasta --reference dataset/train.fasta
    python evaluate.py --input generated.fasta --output report.json --plot
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Configure logging BEFORE peptidegen imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def read_fasta(filepath: str) -> List[str]:
    """Read sequences from FASTA file."""
    sequences = []
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                current_seq = []
            else:
                current_seq.append(line.upper())
        if current_seq:
            sequences.append(''.join(current_seq))
    
    return sequences


def evaluate_sequences(sequences: List[str]) -> dict:
    """Comprehensive evaluation of peptide sequences."""
    from peptidegen.evaluation import (
        PeptideStabilityAnalyzer,
        calculate_diversity_metrics,
        calculate_length_statistics,
        calculate_amino_acid_distribution,
        detect_mode_collapse,
        analyze_amp_properties,
    )
    
    results = {
        'total_sequences': len(sequences),
        'stability': {},
        'diversity': {},
        'length_stats': {},
        'aa_distribution': {},
        'mode_collapse': {},
        'amp_properties': {},
    }
    
    # Filter valid sequences
    valid_seqs = [s for s in sequences if s and len(s) >= 5]
    
    # Stability analysis
    analyzer = PeptideStabilityAnalyzer()
    stability_results = []
    stable_count = 0
    
    for seq in valid_seqs:
        try:
            metrics = analyzer.analyze(seq)
            stability_results.append(metrics)
            if metrics.get('instability_index', 100) < 40:
                stable_count += 1
        except Exception:
            pass

    results['stability'] = {
        'valid_sequences': len(stability_results),
        'stable_sequences': stable_count,
        'stability_rate': (stable_count / len(stability_results) * 100) if stability_results else 0,
    }

    # Diversity metrics
    results['diversity'] = calculate_diversity_metrics(valid_seqs)

    # Length statistics
    results['length_stats'] = calculate_length_statistics(valid_seqs)

    # AA distribution
    results['aa_distribution'] = calculate_amino_acid_distribution(valid_seqs)

    # Mode collapse detection
    results['mode_collapse'] = detect_mode_collapse(valid_seqs)

    # AMP properties
    try:
        results['amp_properties'] = analyze_amp_properties(valid_seqs)
    except Exception:
        results['amp_properties'] = {}
    
    return results


def print_report(results: dict):
    """Print evaluation report."""
    print("\n" + "="*60)
    print("PEPTIDE EVALUATION REPORT")
    print("="*60)
    
    print(f"\n[OVERVIEW]")
    print(f"  Total sequences: {results['total_sequences']}")
    print(f"  Valid sequences: {results['stability']['valid_sequences']}")
    print(f"  Stable (II<40): {results['stability']['stable_sequences']}")
    print(f"  Stability rate: {results['stability']['stability_rate']:.1f}%")
    
    div = results.get('diversity', {})
    print(f"\n[DIVERSITY]")
    print(f"  Unique: {div.get('unique_sequences', 'N/A')}/{div.get('num_sequences', 'N/A')}")
    print(f"  Uniqueness: {div.get('uniqueness_ratio', 0)*100:.1f}%")
    print(f"  Bigram diversity: {div.get('bigram_diversity', 0):.3f}")
    print(f"  Trigram diversity: {div.get('trigram_diversity', 0):.3f}")
    
    length = results.get('length_stats', {})
    print(f"\n[LENGTH]")
    print(f"  Mean: {length.get('mean_length', 0):.1f}")
    print(f"  Std: {length.get('std_length', 0):.1f}")
    print(f"  Range: {length.get('min_length', 0)} - {length.get('max_length', 0)}")
    
    collapse = results.get('mode_collapse', {})
    print(f"\n[MODE COLLAPSE]")
    print(f"  Detected: {'YES ⚠️' if collapse.get('detected', False) else 'NO ✓'}")
    print(f"  Unique AA types: {collapse.get('unique_aa_count', 0)}/20")
    print(f"  Entropy: {collapse.get('entropy', 0):.3f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate peptide sequences')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input FASTA file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output JSON report')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()
    
    # Load sequences
    logger.info(f"Loading sequences from {args.input}")
    sequences = read_fasta(args.input)
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # Evaluate
    results = evaluate_sequences(sequences)
    
    # Print report
    print_report(results)
    
    # Save JSON
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved report to {args.output}")
    
    # Check mode collapse
    mode_collapse_data = results.get('mode_collapse', {})
    if isinstance(mode_collapse_data, dict) and mode_collapse_data.get('mode_collapse', False):
        logger.warning("Mode collapse detected!")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
