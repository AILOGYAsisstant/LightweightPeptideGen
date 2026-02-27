#!/usr/bin/env python
"""
Consolidated Data Analysis Script

Combines quick analysis, deep analysis, and dataset checks.

Usage:
    python scripts/analyze_data.py                          # Analyze default train.csv
    python scripts/analyze_data.py dataset/custom.csv       # Analyze custom file
    python scripts/analyze_data.py --deep                   # Full deep analysis
"""

import argparse
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path


def analyze_data(csv_path: str, deep: bool = False):
    """Main analysis function."""
    
    df = pd.read_csv(csv_path)
    
    print('=' * 60)
    print(f'DATA ANALYSIS: {csv_path}')
    print('=' * 60)
    
    # === BASIC STATS ===
    print(f'\nTotal samples: {len(df):,}')
    if 'label' in df.columns:
        print(f'Label 0: {(df["label"]==0).sum():,}, Label 1: {(df["label"]==1).sum():,}')
    
    # Length stats
    lens = df['sequence'].str.len()
    print(f'\nLength: min={lens.min()}, max={lens.max()}, mean={lens.mean():.1f}, std={lens.std():.1f}')
    
    # === AA DISTRIBUTION ===
    all_aa = ''.join(df['sequence'].values)
    aa_counts = Counter(all_aa)
    total = len(all_aa)
    
    freqs = [c/total for c in aa_counts.values()]
    print(f'\nAA freq range: {min(freqs)*100:.2f}% - {max(freqs)*100:.2f}%')
    print(f'Imbalance ratio: {max(freqs)/min(freqs):.1f}x')
    
    # Non-standard chars
    std = set('ACDEFGHIKLMNPQRSTVWY')
    unusual = set(aa_counts.keys()) - std
    if unusual:
        print(f'Non-standard chars: {unusual}')
    
    # === DIVERSITY ===
    unique = len(set(df['sequence']))
    print(f'\nUnique sequences: {unique:,} / {len(df):,} ({100*unique/len(df):.1f}%)')
    
    # N-gram coverage
    bigrams = set()
    trigrams = set()
    for seq in df['sequence']:
        for i in range(len(seq) - 1):
            bigrams.add(seq[i:i+2])
        for i in range(len(seq) - 2):
            trigrams.add(seq[i:i+3])
    
    print(f'Bigram coverage: {len(bigrams)}/400 ({100*len(bigrams)/400:.1f}%)')
    print(f'Trigram coverage: {len(trigrams)}/8000 ({100*len(trigrams)/8000:.1f}%)')
    
    if not deep:
        return _quick_summary(df, unique)
    
    # === DEEP ANALYSIS ===
    print('\n' + '=' * 60)
    print('DEEP ANALYSIS')
    print('=' * 60)
    
    # Duplicates
    seq_counts = Counter(df['sequence'])
    dup_seqs = [(s, c) for s, c in seq_counts.items() if c > 1]
    print(f'\nDuplicate analysis:')
    print(f'  Sequences with duplicates: {len(dup_seqs):,}')
    print(f'  Total duplicate entries: {sum(c-1 for s,c in dup_seqs):,}')
    
    if dup_seqs:
        print('  Most repeated:')
        for seq, count in seq_counts.most_common(3):
            print(f'    {seq[:25]}... : {count}x')
    
    # Position-wise entropy
    print('\nPosition entropy (first 10):')
    for pos in range(min(10, lens.min())):
        aa_at_pos = [s[pos] for s in df['sequence']]
        counts = Counter(aa_at_pos)
        probs = [c/len(aa_at_pos) for c in counts.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_ent = np.log(20)
        print(f'  Pos {pos}: {entropy:.2f}/{max_ent:.2f} ({100*entropy/max_ent:.0f}%)')
    
    # Feature stats
    features = ['instability_index', 'therapeutic_score', 'hemolytic_score', 
                'hydrophobic_moment', 'gravy', 'charge_at_pH7', 'aromaticity']
    print('\nFeature statistics:')
    for feat in features:
        if feat in df.columns:
            vals = df[feat].dropna()
            print(f'  {feat:22s}: {vals.mean():7.2f} ± {vals.std():.2f}')
    
    # AMP vs non-AMP
    if 'label' in df.columns:
        amp = df[df['label'] == 1]
        non_amp = df[df['label'] == 0]
        print(f'\nAMP vs non-AMP length: {amp["sequence"].str.len().mean():.1f} vs {non_amp["sequence"].str.len().mean():.1f}')
    
    return _quick_summary(df, unique)


def _quick_summary(df, unique):
    """Print summary and return status."""
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    
    issues = []
    unique_ratio = unique / len(df)
    
    if unique_ratio < 0.6:
        issues.append(f'Low uniqueness ({unique_ratio*100:.1f}%)')
    
    if len(df) < 10000:
        issues.append(f'Small dataset ({len(df):,} samples)')
    
    if issues:
        print(f'⚠️  Issues: {", ".join(issues)}')
    else:
        print('✓ Data looks good for training')
    
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description='Analyze peptide dataset')
    parser.add_argument('csv', nargs='?', default='dataset/train.csv',
                        help='Path to CSV file')
    parser.add_argument('--deep', '-d', action='store_true',
                        help='Run deep analysis')
    args = parser.parse_args()
    
    if not Path(args.csv).exists():
        print(f'Error: {args.csv} not found')
        return 1
    
    analyze_data(args.csv, deep=args.deep)
    return 0


if __name__ == '__main__':
    exit(main())
