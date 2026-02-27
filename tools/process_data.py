#!/usr/bin/env python3
"""
🔬 ADVANCED PEPTIDE DATA PROCESSOR
===================================
Enhanced version for LightweightPeptideGen with:
- Duplicate removal with similarity filtering
- AA distribution balancing
- Data augmentation for diversity
- Quality validation
- Feature computation with advanced metrics

Author: Enhanced from script_process
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import warnings
import gc
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
import hashlib
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class ProcessingConfig:
    """Configuration for data processing"""
    
    # Sequence constraints
    MIN_LENGTH = 5
    MAX_LENGTH = 50
    VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Duplicate removal
    REMOVE_EXACT_DUPLICATES = True
    REMOVE_SIMILAR_SEQUENCES = True
    SIMILARITY_THRESHOLD = 0.95  # Remove if >95% similar
    
    # Balancing
    BALANCE_LABELS = True
    TARGET_RATIO = 1.0  # AMP:non-AMP ratio
    
    # Data augmentation
    ENABLE_AUGMENTATION = True
    AUGMENTATION_FACTOR = 1.5  # Multiply rare class
    
    # Quality filtering
    MIN_AA_DIVERSITY = 5  # Minimum unique AAs in sequence
    MAX_REPEAT_RATIO = 0.5  # Max ratio of repeated AAs
    
    # Split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42


# ============================================================================
# STAGE 1: ADVANCED SEQUENCE CLEANER
# ============================================================================

class AdvancedSequenceCleaner:
    """Enhanced sequence cleaning with quality metrics"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.stats = defaultdict(int)
    
    def clean_sequence(self, seq: str) -> str:
        """Clean and normalize sequence"""
        # Basic cleaning
        seq = str(seq).upper().strip()
        seq = ''.join(c for c in seq if c.isalpha())
        
        # Remove common non-standard AAs
        replacements = {'B': 'N', 'Z': 'Q', 'X': '', 'U': 'C', 'O': 'K', 'J': 'L'}
        for old, new in replacements.items():
            seq = seq.replace(old, new)
        
        return seq
    
    def compute_quality_score(self, seq: str) -> Dict:
        """Compute quality metrics for a sequence"""
        if not seq:
            return {'valid': False, 'reason': 'empty'}
        
        length = len(seq)
        aa_counts = Counter(seq)
        unique_aas = len(aa_counts)
        most_common_ratio = max(aa_counts.values()) / length if length > 0 else 1.0
        
        # Check consecutive repeats
        max_repeat = 1
        current_repeat = 1
        for i in range(1, length):
            if seq[i] == seq[i-1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1
        
        repeat_ratio = max_repeat / length if length > 0 else 0
        
        # Calculate entropy (diversity score)
        probs = [c / length for c in aa_counts.values()]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
        max_entropy = np.log2(min(20, length))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            'length': length,
            'unique_aas': unique_aas,
            'most_common_ratio': most_common_ratio,
            'repeat_ratio': repeat_ratio,
            'max_repeat': max_repeat,
            'entropy': normalized_entropy
        }
    
    def is_valid(self, seq: str) -> Tuple[bool, str]:
        """Check if sequence is valid with detailed reason"""
        if not seq:
            return False, "empty"
        
        # Length check
        if len(seq) < self.config.MIN_LENGTH:
            return False, f"too_short ({len(seq)} < {self.config.MIN_LENGTH})"
        if len(seq) > self.config.MAX_LENGTH:
            return False, f"too_long ({len(seq)} > {self.config.MAX_LENGTH})"
        
        # Invalid characters
        invalid_chars = set(seq) - self.config.VALID_AA
        if invalid_chars:
            return False, f"invalid_chars: {invalid_chars}"
        
        # Quality checks
        quality = self.compute_quality_score(seq)
        
        if quality['unique_aas'] < self.config.MIN_AA_DIVERSITY:
            return False, f"low_diversity ({quality['unique_aas']} unique AAs)"
        
        if quality['repeat_ratio'] > self.config.MAX_REPEAT_RATIO:
            return False, f"high_repeat ({quality['repeat_ratio']:.2f})"
        
        if quality['max_repeat'] > 5:
            return False, f"long_repeat ({quality['max_repeat']} consecutive)"
        
        return True, "valid"
    
    def process_batch(self, sequences: List[Dict]) -> List[Dict]:
        """Process a batch of sequences"""
        cleaned = []
        
        for item in sequences:
            self.stats['total'] += 1
            
            seq = self.clean_sequence(item.get('sequence', ''))
            is_valid, reason = self.is_valid(seq)
            
            if not is_valid:
                self.stats[f'rejected_{reason.split()[0]}'] += 1
                continue
            
            item['sequence'] = seq
            item['quality'] = self.compute_quality_score(seq)
            cleaned.append(item)
            self.stats['valid'] += 1
        
        return cleaned


# ============================================================================
# STAGE 2: ADVANCED DUPLICATE REMOVER
# ============================================================================

class AdvancedDuplicateRemover:
    """Remove exact and near-duplicates"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.seen_hashes: Set[str] = set()
        self.seen_sequences: Set[str] = set()
    
    def _hash_sequence(self, seq: str) -> str:
        """Create hash for sequence"""
        return hashlib.md5(seq.encode()).hexdigest()
    
    def _compute_similarity(self, seq1: str, seq2: str) -> float:
        """Compute sequence similarity (Levenshtein-based)"""
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        
        # Quick check with length difference
        len_diff = abs(len(seq1) - len(seq2)) / max(len(seq1), len(seq2))
        if len_diff > (1 - self.config.SIMILARITY_THRESHOLD):
            return 0.0
        
        # Simple identity-based similarity for efficiency
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / max(len(seq1), len(seq2))
    
    def remove_duplicates(self, sequences: List[Dict], batch_size: int = 10000) -> List[Dict]:
        """Remove duplicates with progress tracking"""
        print("\n" + "=" * 60)
        print("STAGE 2: DUPLICATE REMOVAL")
        print("=" * 60)
        
        unique = []
        exact_dups = 0
        similar_dups = 0
        
        # First pass: exact duplicates
        print("\n[1/2] Removing exact duplicates...")
        for item in tqdm(sequences):
            seq = item['sequence']
            seq_hash = self._hash_sequence(seq)
            
            if seq_hash in self.seen_hashes:
                exact_dups += 1
                continue
            
            if seq in self.seen_sequences:
                exact_dups += 1
                continue
            
            self.seen_hashes.add(seq_hash)
            self.seen_sequences.add(seq)
            unique.append(item)
        
        print(f"  Removed {exact_dups:,} exact duplicates")
        print(f"  Remaining: {len(unique):,}")
        
        # Second pass: similar sequences (if enabled)
        if self.config.REMOVE_SIMILAR_SEQUENCES:
            print("\n[2/2] Removing similar sequences...")
            unique = self._remove_similar(unique, batch_size)
            similar_dups = len(sequences) - exact_dups - len(unique)
            print(f"  Removed {similar_dups:,} similar sequences")
        
        print(f"\n  ✓ Final unique: {len(unique):,}")
        
        return unique
    
    def _remove_similar(self, sequences: List[Dict], batch_size: int) -> List[Dict]:
        """Remove sequences too similar to others"""
        # Sort by length for efficient comparison
        sequences = sorted(sequences, key=lambda x: len(x['sequence']))
        
        keep_indices = set(range(len(sequences)))
        
        # Process in batches for memory efficiency
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_end = min(i + batch_size, len(sequences))
            
            for j in range(i, batch_end):
                if j not in keep_indices:
                    continue
                
                seq1 = sequences[j]['sequence']
                
                # Compare with nearby sequences (same length ± 2)
                for k in range(j + 1, len(sequences)):
                    if k not in keep_indices:
                        continue
                    
                    seq2 = sequences[k]['sequence']
                    
                    # Skip if length difference too large
                    if abs(len(seq1) - len(seq2)) > 2:
                        break
                    
                    similarity = self._compute_similarity(seq1, seq2)
                    
                    if similarity > self.config.SIMILARITY_THRESHOLD:
                        keep_indices.discard(k)
        
        return [sequences[i] for i in sorted(keep_indices)]


# ============================================================================
# STAGE 3: DATA BALANCER & AUGMENTER
# ============================================================================

class DataBalancerAugmenter:
    """Balance dataset and augment underrepresented classes"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
    
    def _shuffle_preserve_structure(self, seq: str) -> str:
        """Shuffle middle of sequence while preserving termini"""
        if len(seq) <= 4:
            return seq
        
        start = seq[0]
        end = seq[-1]
        middle = list(seq[1:-1])
        np.random.shuffle(middle)
        
        return start + ''.join(middle) + end
    
    def _point_mutation(self, seq: str, mutation_rate: float = 0.05) -> str:
        """Apply conservative point mutations"""
        # Conservative substitution groups
        groups = {
            'A': 'GAVLI', 'G': 'AGVLIP', 'V': 'AVLIMG', 'L': 'LIVMFWA', 'I': 'ILVMFA',
            'M': 'MLIV', 'F': 'FYWLM', 'W': 'WFY', 'Y': 'YWFH',
            'P': 'PA', 'S': 'STNA', 'T': 'TS', 'C': 'CA',
            'N': 'NQSD', 'Q': 'QNEK', 'D': 'DNEQ', 'E': 'EDQK',
            'K': 'KRQ', 'R': 'RKQ', 'H': 'HKR'
        }
        
        seq = list(seq)
        for i in range(len(seq)):
            if np.random.random() < mutation_rate:
                aa = seq[i]
                if aa in groups:
                    possible = groups[aa]
                    seq[i] = np.random.choice(list(possible))
        
        return ''.join(seq)
    
    def _reverse_sequence(self, seq: str) -> str:
        """Reverse sequence (sometimes valid for AMPs)"""
        return seq[::-1]
    
    def augment_sequence(self, seq: str, method: str = 'random') -> Optional[str]:
        """Augment a single sequence"""
        methods = ['mutation', 'shuffle', 'reverse']
        
        if method == 'random':
            method = np.random.choice(methods, p=[0.6, 0.3, 0.1])
        
        if method == 'mutation':
            return self._point_mutation(seq)
        elif method == 'shuffle':
            return self._shuffle_preserve_structure(seq)
        elif method == 'reverse':
            return self._reverse_sequence(seq)
        
        return None
    
    def balance_and_augment(self, sequences: List[Dict]) -> List[Dict]:
        """Balance dataset with augmentation"""
        print("\n" + "=" * 60)
        print("STAGE 3: BALANCING & AUGMENTATION")
        print("=" * 60)
        
        # Separate by label
        amp = [s for s in sequences if s.get('label') == 1]
        non_amp = [s for s in sequences if s.get('label') == 0]
        unknown = [s for s in sequences if s.get('label') not in [0, 1]]
        
        print(f"\n  Original distribution:")
        print(f"    AMP:     {len(amp):,}")
        print(f"    non-AMP: {len(non_amp):,}")
        print(f"    Unknown: {len(unknown):,}")
        
        if not self.config.BALANCE_LABELS:
            return sequences
        
        # Determine target size
        target_size = int(max(len(amp), len(non_amp)) * self.config.TARGET_RATIO)
        
        # Augment smaller class
        if len(amp) < len(non_amp):
            minority, majority = amp, non_amp
            minority_label = 1
        else:
            minority, majority = non_amp, amp
            minority_label = 0
        
        print(f"\n  Augmenting minority class (label={minority_label})...")
        
        augmented = list(minority)
        seen_aug = set(s['sequence'] for s in minority)
        
        # Augmentation
        if self.config.ENABLE_AUGMENTATION:
            needed = target_size - len(minority)
            attempts = 0
            max_attempts = needed * 3
            
            pbar = tqdm(total=needed, desc="  Augmenting")
            while len(augmented) < target_size and attempts < max_attempts:
                attempts += 1
                
                # Pick random sequence to augment
                source = np.random.choice(minority)
                new_seq = self.augment_sequence(source['sequence'])
                
                if new_seq and new_seq not in seen_aug and len(new_seq) >= self.config.MIN_LENGTH:
                    new_item = source.copy()
                    new_item['sequence'] = new_seq
                    new_item['augmented'] = True
                    augmented.append(new_item)
                    seen_aug.add(new_seq)
                    pbar.update(1)
            
            pbar.close()
        
        # Downsample majority if needed
        if len(majority) > target_size:
            print(f"  Downsampling majority class to {target_size:,}...")
            indices = np.random.choice(len(majority), target_size, replace=False)
            majority = [majority[i] for i in indices]
        
        # Combine
        balanced = augmented + majority + unknown
        np.random.shuffle(balanced)
        
        # Recount
        final_amp = sum(1 for s in balanced if s.get('label') == 1)
        final_non = sum(1 for s in balanced if s.get('label') == 0)
        
        print(f"\n  Final distribution:")
        print(f"    AMP:     {final_amp:,}")
        print(f"    non-AMP: {final_non:,}")
        print(f"    Ratio:   {final_amp/max(final_non,1):.2f}")
        
        return balanced


# ============================================================================
# STAGE 4: ADVANCED FEATURE EXTRACTOR
# ============================================================================

class AdvancedFeatureExtractor:
    """Extract comprehensive features for peptides"""
    
    # Hydrophobicity scales
    KYTE_DOOLITTLE = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    # Hydrophobic moment calculation
    EISENBERG = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
    }
    
    def __init__(self):
        pass
    
    def extract_all_features(self, seq: str) -> Optional[Dict]:
        """Extract all features for a sequence"""
        try:
            features = {}
            
            # Basic features from BioPython
            analysis = ProteinAnalysis(seq)
            aa_comp = analysis.get_amino_acids_percent()
            
            features['length'] = len(seq)
            features['molecular_weight'] = analysis.molecular_weight()
            features['aromaticity'] = analysis.aromaticity()
            features['instability_index'] = analysis.instability_index()
            features['isoelectric_point'] = analysis.isoelectric_point()
            features['gravy'] = analysis.gravy()
            features['charge_at_pH7'] = analysis.charge_at_pH(7.0)
            
            # AA composition ratios
            features['hydrophobic_ratio'] = sum(aa_comp.get(aa, 0) for aa in 'AILMFWV')
            features['positive_ratio'] = sum(aa_comp.get(aa, 0) for aa in 'KRH')
            features['negative_ratio'] = sum(aa_comp.get(aa, 0) for aa in 'DE')
            features['aromatic_ratio'] = sum(aa_comp.get(aa, 0) for aa in 'FYW')
            features['aliphatic_index'] = sum(aa_comp.get(aa, 0) for aa in 'AVIL')
            
            # Boman index
            boman_values = {
                'A': 0.61, 'C': 1.07, 'D': 0.46, 'E': 0.47, 'F': 2.02,
                'G': 0.0, 'H': 0.61, 'I': 2.22, 'K': 1.15, 'L': 1.53,
                'M': 1.18, 'N': 0.06, 'P': 1.95, 'Q': 0.0, 'R': 0.60,
                'S': 0.42, 'T': 0.71, 'V': 1.32, 'W': 2.65, 'Y': 1.88
            }
            features['boman_index'] = np.mean([boman_values.get(aa, 0) for aa in seq])
            
            # Hydrophobic moment (for alpha-helix, angle=100°)
            features['hydrophobic_moment'] = self._compute_hydrophobic_moment(seq, angle=100)
            
            # Amphipathicity
            features['amphipathicity'] = self._compute_amphipathicity(seq)
            
            # Hemolytic score estimation (simplified)
            features['hemolytic_score'] = self._estimate_hemolytic_score(seq, features)
            
            # Therapeutic score (AMP potential)
            features['therapeutic_score'] = self._estimate_therapeutic_score(seq, features)
            
            return features
            
        except Exception as e:
            return None
    
    def _compute_hydrophobic_moment(self, seq: str, angle: float = 100) -> float:
        """Compute hydrophobic moment for helical structure"""
        angle_rad = np.deg2rad(angle)
        
        h_cos = 0.0
        h_sin = 0.0
        
        for i, aa in enumerate(seq):
            h = self.EISENBERG.get(aa, 0)
            h_cos += h * np.cos(i * angle_rad)
            h_sin += h * np.sin(i * angle_rad)
        
        return np.sqrt(h_cos**2 + h_sin**2) / len(seq) if len(seq) > 0 else 0
    
    def _compute_amphipathicity(self, seq: str) -> float:
        """Compute amphipathicity score"""
        if len(seq) < 2:
            return 0.0
        
        # Sliding window approach
        window = 7
        scores = []
        
        for i in range(len(seq) - window + 1):
            window_seq = seq[i:i+window]
            hydro = [self.KYTE_DOOLITTLE.get(aa, 0) for aa in window_seq]
            
            # variance indicates amphipathicity
            scores.append(np.var(hydro))
        
        return np.mean(scores) if scores else 0
    
    def _estimate_hemolytic_score(self, seq: str, features: Dict) -> float:
        """Estimate hemolytic potential (lower = safer)"""
        # Based on known hemolytic motifs and properties
        score = 0.0
        
        # High hydrophobicity increases hemolysis
        score += max(0, features.get('gravy', 0) * 2)
        
        # Positive charge can increase hemolysis
        if features.get('charge_at_pH7', 0) > 5:
            score += 1.0
        
        # Amphipathicity correlates with hemolysis
        score += features.get('amphipathicity', 0) * 3
        
        # Large hydrophobic moment
        score += features.get('hydrophobic_moment', 0) * 2
        
        # Normalize to 0-10 scale
        return min(10, max(0, score))
    
    def _estimate_therapeutic_score(self, seq: str, features: Dict) -> float:
        """Estimate therapeutic potential (higher = better)"""
        score = 0.0
        
        # Moderate positive charge is good
        charge = features.get('charge_at_pH7', 0)
        if 2 <= charge <= 7:
            score += 2.0
        elif charge > 7:
            score += 1.0
        
        # Moderate hydrophobicity
        gravy = features.get('gravy', 0)
        if -1 <= gravy <= 1:
            score += 1.5
        
        # Stability (low instability index)
        if features.get('instability_index', 100) < 40:
            score += 2.0
        
        # Good length range
        length = features.get('length', 0)
        if 15 <= length <= 35:
            score += 1.0
        
        # Amphipathic structure
        if 0.3 <= features.get('hydrophobic_moment', 0) <= 0.8:
            score += 1.5
        
        return score
    
    def process_sequences(self, sequences: List[Dict], batch_size: int = 5000) -> List[Dict]:
        """Process sequences in batches"""
        print("\n" + "=" * 60)
        print("STAGE 4: FEATURE EXTRACTION")
        print("=" * 60)
        
        processed = []
        failed = 0
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="  Extracting"):
            batch = sequences[i:i+batch_size]
            
            for item in batch:
                features = self.extract_all_features(item['sequence'])
                
                if features:
                    item.update(features)
                    processed.append(item)
                else:
                    failed += 1
            
            gc.collect()
        
        print(f"\n  ✓ Processed: {len(processed):,}")
        print(f"  ✗ Failed:    {failed:,}")
        
        return processed


# ============================================================================
# STAGE 5: SPLIT & EXPORT
# ============================================================================

class DatasetExporter:
    """Export processed dataset"""
    
    def __init__(self, output_dir: str = 'dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def stratified_split(self, sequences: List[Dict], config: ProcessingConfig) -> Tuple:
        """Perform stratified split"""
        print("\n" + "=" * 60)
        print("STAGE 5: SPLITTING DATASET")
        print("=" * 60)
        
        # Extract labels
        labels = [s.get('label', 0) for s in sequences]
        
        # First split: train vs temp
        train, temp, train_labels, temp_labels = train_test_split(
            sequences, labels,
            test_size=(1 - config.TRAIN_RATIO),
            stratify=labels,
            random_state=config.RANDOM_SEED
        )
        
        # Second split: val vs test
        val_ratio = config.VAL_RATIO / (1 - config.TRAIN_RATIO)
        val, test, _, _ = train_test_split(
            temp, temp_labels,
            test_size=(1 - val_ratio),
            stratify=temp_labels,
            random_state=config.RANDOM_SEED
        )
        
        print(f"\n  Split sizes:")
        print(f"    Train: {len(train):,}")
        print(f"    Val:   {len(val):,}")
        print(f"    Test:  {len(test):,}")
        
        return train, val, test
    
    def export(self, train: List, val: List, test: List):
        """Export all splits"""
        print("\n" + "=" * 60)
        print("STAGE 6: EXPORTING")
        print("=" * 60)
        
        for name, data in [('train', train), ('val', val), ('test', test)]:
            self._export_split(data, name)
        
        self._export_statistics(train, val, test)
        self._export_feature_config()
        
        print(f"\n  ✓ All files saved to: {self.output_dir.absolute()}")
    
    def _export_split(self, data: List[Dict], name: str):
        """Export a single split"""
        df = pd.DataFrame(data)
        
        # Remove internal fields
        cols_to_drop = ['quality', 'augmented', 'source']
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # CSV
        df.to_csv(self.output_dir / f'{name}.csv', index=False)
        
        # FASTA
        with open(self.output_dir / f'{name}.fasta', 'w') as f:
            for idx, row in df.iterrows():
                label = 'AMP' if row.get('label', 0) == 1 else 'nonAMP'
                f.write(f">{label}_{idx}\n{row['sequence']}\n")
        
        print(f"  ✓ {name}.csv & {name}.fasta")
    
    def _export_statistics(self, train: List, val: List, test: List):
        """Export dataset statistics"""
        stats = {}
        
        for name, data in [('train', train), ('val', val), ('test', test)]:
            n_amp = sum(1 for s in data if s.get('label') == 1)
            n_non = sum(1 for s in data if s.get('label') == 0)
            
            stats[name] = {
                'n_samples': len(data),
                'n_amp': n_amp,
                'n_non_amp': n_non
            }
        
        with open(self.output_dir / 'dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _export_feature_config(self):
        """Export feature configuration"""
        config = {
            'features': {
                'sequence': 'sequence',
                'label': 'label',
                'basic_features': [
                    'length', 'molecular_weight', 'aromaticity',
                    'instability_index', 'isoelectric_point', 'gravy',
                    'charge_at_pH7', 'hydrophobic_ratio', 'positive_ratio',
                    'negative_ratio', 'aromatic_ratio', 'aliphatic_index',
                    'boman_index'
                ],
                'advanced_toxicity_features': [
                    'hydrophobic_moment', 'amphipathicity',
                    'hemolytic_score', 'therapeutic_score'
                ]
            }
        }
        
        with open(self.output_dir / 'feature_config.json', 'w') as f:
            json.dump(config, f, indent=2)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class PeptideDataProcessor:
    """Main pipeline orchestrator"""
    
    def __init__(
        self,
        input_path: str,
        output_dir: str = 'dataset_processed',
        config: ProcessingConfig = None
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.config = config or ProcessingConfig()
        
        # Initialize components
        self.cleaner = AdvancedSequenceCleaner(self.config)
        self.deduplicator = AdvancedDuplicateRemover(self.config)
        self.balancer = DataBalancerAugmenter(self.config)
        self.feature_extractor = AdvancedFeatureExtractor()
        self.exporter = DatasetExporter(output_dir)
    
    def run(self):
        """Run full pipeline"""
        print("=" * 70)
        print("🔬 ADVANCED PEPTIDE DATA PROCESSOR")
        print("=" * 70)
        print(f"\nInput:  {self.input_path}")
        print(f"Output: {self.output_dir}")
        
        # Load data
        print("\n" + "=" * 60)
        print("STAGE 1: LOADING & CLEANING")
        print("=" * 60)
        
        df = pd.read_csv(self.input_path)
        print(f"\n  Loaded {len(df):,} sequences")
        
        # Convert to list of dicts
        sequences = df.to_dict('records')
        
        # Clean
        sequences = self.cleaner.process_batch(sequences)
        print(f"\n  Cleaning stats:")
        for key, val in self.cleaner.stats.items():
            print(f"    {key}: {val:,}")
        
        # Remove duplicates
        sequences = self.deduplicator.remove_duplicates(sequences)
        
        # Balance & augment
        sequences = self.balancer.balance_and_augment(sequences)
        
        # Extract features
        sequences = self.feature_extractor.process_sequences(sequences)
        
        # Split
        train, val, test = self.exporter.stratified_split(sequences, self.config)
        
        # Export
        self.exporter.export(train, val, test)
        
        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE")
        print("=" * 70)
        
        return len(train), len(val), len(test)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process peptide dataset')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', default='dataset_processed', help='Output directory')
    parser.add_argument('--no-augment', action='store_true', help='Disable augmentation')
    parser.add_argument('--no-balance', action='store_true', help='Disable balancing')
    parser.add_argument('--similarity-threshold', type=float, default=0.95)
    parser.add_argument('--min-length', type=int, default=5)
    parser.add_argument('--max-length', type=int, default=50)
    
    args = parser.parse_args()
    
    # Configure
    config = ProcessingConfig()
    config.ENABLE_AUGMENTATION = not args.no_augment
    config.BALANCE_LABELS = not args.no_balance
    config.SIMILARITY_THRESHOLD = args.similarity_threshold
    config.MIN_LENGTH = args.min_length
    config.MAX_LENGTH = args.max_length
    
    # Run
    processor = PeptideDataProcessor(args.input, args.output, config)
    processor.run()


if __name__ == '__main__':
    main()
