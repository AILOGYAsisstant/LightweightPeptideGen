#!/usr/bin/env python3
"""
🔍 DATASET QUALITY VALIDATOR
============================
Validate processed dataset quality before training
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import json

def validate_dataset(dataset_dir: str = 'dataset'):
    """Validate dataset quality"""
    
    dataset_dir = Path(dataset_dir)
    
    print("=" * 60)
    print("🔍 DATASET QUALITY VALIDATION")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    for split in ['train', 'val', 'test']:
        csv_path = dataset_dir / f'{split}.csv'
        if not csv_path.exists():
            issues.append(f"Missing {split}.csv")
            continue
        
        df = pd.read_csv(csv_path)
        print(f"\n{'='*40}")
        print(f"📊 {split.upper()} SET ({len(df):,} samples)")
        print(f"{'='*40}")
        
        # 1. Check duplicates
        n_dup = len(df) - len(df['sequence'].unique())
        if n_dup > 0:
            issues.append(f"{split}: {n_dup} duplicate sequences")
        print(f"  Duplicates: {n_dup}")
        
        # 2. Check label distribution
        if 'label' in df.columns:
            label_dist = df['label'].value_counts().to_dict()
            print(f"  Label distribution: {label_dist}")
            
            n_amp = label_dist.get(1, 0)
            n_non = label_dist.get(0, 0)
            
            if n_amp > 0 and n_non > 0:
                ratio = n_amp / n_non
                if ratio < 0.5 or ratio > 2.0:
                    warnings.append(f"{split}: Imbalanced labels (ratio={ratio:.2f})")
        
        # 3. Check sequence quality
        lengths = df['sequence'].str.len()
        print(f"  Length: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")
        
        if lengths.min() < 5:
            issues.append(f"{split}: Sequences too short ({lengths.min()})")
        if lengths.max() > 50:
            warnings.append(f"{split}: Sequences too long ({lengths.max()})")
        
        # 4. AA distribution
        all_aa = ''.join(df['sequence'])
        aa_counts = Counter(all_aa)
        total = len(all_aa)
        
        freqs = [c/total for c in aa_counts.values()]
        min_freq = min(freqs) * 100
        max_freq = max(freqs) * 100
        
        print(f"  AA freq range: {min_freq:.2f}% - {max_freq:.2f}%")
        
        if max_freq / max(min_freq, 0.01) > 10:
            warnings.append(f"{split}: High AA imbalance ({max_freq/min_freq:.1f}x)")
        
        # 5. Check for non-standard AAs
        std_aa = set('ACDEFGHIKLMNPQRSTVWY')
        unusual = set(aa_counts.keys()) - std_aa
        if unusual:
            issues.append(f"{split}: Non-standard AAs found: {unusual}")
            print(f"  ⚠️ Non-standard AAs: {unusual}")
        
        # 6. Check required features
        required_features = ['instability_index', 'therapeutic_score', 'hemolytic_score',
                           'hydrophobic_moment', 'gravy', 'charge_at_pH7', 'aromaticity']
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            warnings.append(f"{split}: Missing features: {missing_features}")
        
        # 7. Check for NaN values
        nan_counts = df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            warnings.append(f"{split}: NaN values in {list(nan_cols.index)}")
        
        # 8. Sequence diversity
        unique_ratio = len(df['sequence'].unique()) / len(df)
        print(f"  Unique ratio: {unique_ratio*100:.1f}%")
        
        if unique_ratio < 0.9:
            warnings.append(f"{split}: Low uniqueness ({unique_ratio*100:.1f}%)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    if issues:
        print("\n❌ ISSUES (must fix):")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("\n⚠️ WARNINGS (consider fixing):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("\n✅ All checks passed! Dataset is ready for training.")
        return True
    elif not issues:
        print("\n✓ No critical issues. Dataset can be used with caution.")
        return True
    else:
        print("\n✗ Critical issues found. Please fix before training.")
        return False


def compare_before_after(original_path: str, processed_dir: str):
    """Compare original vs processed dataset"""
    
    print("\n" + "=" * 60)
    print("📊 BEFORE vs AFTER COMPARISON")
    print("=" * 60)
    
    # Load original
    original = pd.read_csv(original_path)
    
    # Load processed
    processed_dir = Path(processed_dir)
    train = pd.read_csv(processed_dir / 'train.csv')
    val = pd.read_csv(processed_dir / 'val.csv')
    test = pd.read_csv(processed_dir / 'test.csv')
    processed = pd.concat([train, val, test])
    
    print(f"\n  Original samples:  {len(original):,}")
    print(f"  Processed samples: {len(processed):,}")
    print(f"  Reduction:         {(1 - len(processed)/len(original))*100:.1f}%")
    
    # Duplicate comparison
    orig_dup = len(original) - len(original['sequence'].unique())
    proc_dup = len(processed) - len(processed['sequence'].unique())
    
    print(f"\n  Original duplicates:  {orig_dup:,}")
    print(f"  Processed duplicates: {proc_dup:,}")
    
    # Label balance
    if 'label' in original.columns:
        orig_ratio = original['label'].mean()
        proc_ratio = processed['label'].mean()
        
        print(f"\n  Original AMP ratio:  {orig_ratio*100:.1f}%")
        print(f"  Processed AMP ratio: {proc_ratio*100:.1f}%")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', default='dataset', help='Dataset directory')
    parser.add_argument('--compare', '-c', help='Original CSV for comparison')
    
    args = parser.parse_args()
    
    validate_dataset(args.dir)
    
    if args.compare:
        compare_before_after(args.compare, args.dir)
