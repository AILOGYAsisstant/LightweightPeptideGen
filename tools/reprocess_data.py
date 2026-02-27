#!/usr/bin/env python3
"""
🚀 QUICK DATA REPROCESS
=======================
Reprocess existing dataset to fix issues:
- Remove duplicates
- Balance AMP/non-AMP
- Add augmentation for diversity
- Validate quality
"""

import sys
sys.path.insert(0, '.')

from scripts.process_data import PeptideDataProcessor, ProcessingConfig
from scripts.validate_data import validate_dataset, compare_before_after

def main():
    print("=" * 70)
    print("🚀 QUICK DATA REPROCESSING FOR LightweightPeptideGen")
    print("=" * 70)
    
    # Configuration - optimized for fixing mode collapse issues
    config = ProcessingConfig()
    
    # Strict duplicate removal
    config.REMOVE_EXACT_DUPLICATES = True
    config.REMOVE_SIMILAR_SEQUENCES = True
    config.SIMILARITY_THRESHOLD = 0.92  # Remove if >92% similar
    
    # Balance dataset
    config.BALANCE_LABELS = True
    config.TARGET_RATIO = 1.0
    
    # Enable augmentation for diversity
    config.ENABLE_AUGMENTATION = True
    config.AUGMENTATION_FACTOR = 1.3
    
    # Quality filters
    config.MIN_AA_DIVERSITY = 5
    config.MAX_REPEAT_RATIO = 0.4
    
    # Split
    config.TRAIN_RATIO = 0.7
    config.VAL_RATIO = 0.15
    config.TEST_RATIO = 0.15
    
    # Process
    input_csv = 'dataset/train.csv'  # Use existing train.csv as source
    output_dir = 'dataset_clean'
    
    print(f"\nInput:  {input_csv}")
    print(f"Output: {output_dir}")
    print(f"\nConfiguration:")
    print(f"  - Remove duplicates: {config.REMOVE_EXACT_DUPLICATES}")
    print(f"  - Similarity threshold: {config.SIMILARITY_THRESHOLD}")
    print(f"  - Balance labels: {config.BALANCE_LABELS}")
    print(f"  - Augmentation: {config.ENABLE_AUGMENTATION}")
    
    # Run processing
    processor = PeptideDataProcessor(input_csv, output_dir, config)
    train_size, val_size, test_size = processor.run()
    
    # Validate
    print("\n")
    validate_dataset(output_dir)
    
    # Compare
    compare_before_after(input_csv, output_dir)
    
    print("\n" + "=" * 70)
    print("📝 NEXT STEPS:")
    print("=" * 70)
    print(f"\n1. Review the processed data in: {output_dir}/")
    print(f"2. If satisfied, replace the old dataset:")
    print(f"   - Backup: mv dataset dataset_backup")
    print(f"   - Replace: mv {output_dir} dataset")
    print(f"3. Retrain the model with clean data")


if __name__ == '__main__':
    main()
