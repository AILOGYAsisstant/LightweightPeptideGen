# LightweightPeptideGen: A GAN-Based Framework for Conditional Generation of Structurally Stable Antimicrobial Peptides

> **Báo cáo kỹ thuật / Technical Report** — v1.0, February 2026

---

## Tóm tắt (Abstract)

Chúng tôi trình bày **LightweightPeptideGen**, một framework học sâu dựa trên Generative Adversarial Network (GAN) để sinh tự động các **antimicrobial peptide (AMP)** có độ ổn định cấu trúc cao. Framework tích hợp nhiều cơ chế tiên tiến: (1) bộ sinh chuỗi tự hồi quy GRU/LSTM/Transformer với điều kiện hóa đa tính năng, (2) bộ phân biệt CNN đa kernel kết hợp Spectral Normalization và Minibatch Discrimination, (3) tích hợp mô hình ngôn ngữ protein ESM2 (650M tham số) làm đánh giá viên cấu trúc kết hợp với Graph Attention Network (GAT), (4) hệ thống loss đa thành phần chống mode collapse gồm Diversity Loss, N-gram Diversity Loss, Length Penalty Loss và Feature Matching Loss. Mô hình được huấn luyện trên **129,121 chuỗi peptide** với điều kiện hóa 8 đặc trưng sinh hóa, đạt G validation loss ≈ 0.407, entropy > 2.89, D–G gap < 0.25 sau hơn 200 epoch.

---

## Mục lục

1. [Giới thiệu và Bối cảnh](#1-giới-thiệu)
2. [Kiến trúc Hệ thống](#2-kiến-trúc-hệ-thống)
3. [Các Kỹ thuật Chính](#3-các-kỹ-thuật-chính)
4. [Hàm Mất mát](#4-hàm-mất-mát)
5. [Dữ liệu và Dataset](#5-dữ-liệu-và-dataset)
6. [Đặc trưng Điều kiện Hóa](#6-đặc-trưng-điều-kiện-hóa)
7. [Đánh giá và Chỉ số Chất lượng](#7-đánh-giá-và-chỉ-số-chất-lượng)
8. [Yêu cầu Phần cứng & GPU](#8-yêu-cầu-phần-cứng--gpu)
9. [Thư viện và Dependencies](#9-thư-viện-và-dependencies)
10. [Cấu hình Siêu tham số](#10-cấu-hình-siêu-tham-số)
11. [Kết quả Thực nghiệm](#11-kết-quả-thực-nghiệm)
12. [Cấu trúc Dự án](#12-cấu-trúc-dự-án)
13. [Hướng dẫn Sử dụng](#13-hướng-dẫn-sử-dụng)
14. [Tài liệu Tham khảo](#14-tài-liệu-tham-khảo)

---

## 1. Giới thiệu

### 1.1 Vấn đề nghiên cứu

Antimicrobial peptides (AMPs) là các phân tử peptide ngắn (thường 5–50 amino acid) có khả năng tiêu diệt vi khuẩn kháng thuốc. Việc thiết kế AMP theo phương pháp truyền thống (mutagenesis, sàng lọc thực nghiệm) tốn kém và không có khả năng mở rộng. Các phương pháp học sâu sinh tạo (generative deep learning) mở ra khả năng khám phá không gian chuỗi rộng lớn một cách hiệu quả.

### 1.2 Thách thức

- **Mode collapse**: GAN có xu hướng sinh ra các chuỗi lặp đi lặp lại
- **Length collapse**: Mô hình kết thúc chuỗi quá sớm hoặc quá muộn
- **Độ ổn định sinh hóa**: Cần kiểm soát Instability Index, GRAVY, Aliphatic Index
- **Tính đa dạng vs. chất lượng**: Trade-off giữa novelty và tính hợp lệ sinh học

### 1.3 Đóng góp chính

1. Framework GAN với hệ thống loss đa thành phần chuyên biệt cho peptide
2. Tích hợp ESM2 + GAT làm structural evaluator trong vòng lặp huấn luyện
3. Cơ chế điều kiện hóa 8 đặc trưng sinh hóa cho phép kiểm soát tính chất
4. Adaptive Discriminator Training chống D dominance
5. N-gram và Length Penalty Loss giải quyết motif collapse và length collapse

---

## 2. Kiến trúc Hệ thống

### 2.1 Tổng quan kiến trúc GAN

```
Latent Vector z ~ N(0, I)          Condition Vector c (8-dim)
         │                                     │
         └──────────────┬──────────────────────┘
                        ↓
               ┌─────────────────┐
               │   GENERATOR G   │  GRU/LSTM/Transformer
               │  (autoregressive│  + Self-Attention
               │   decoding)     │  + Condition Fusion
               └────────┬────────┘
                        │  Soft one-hot logits (B, L, V)
              ┌─────────┴──────────┐
              ↓                    ↓
   ┌──────────────────┐   ┌─────────────────────┐
   │  DISCRIMINATOR D  │   │  STRUCTURE EVAL      │
   │  CNN multi-kernel │   │  ESM2-650M + GAT     │
   │  + Spectral Norm  │   │  (stability scoring) │
   │  + Minibatch Std  │   └─────────────────────┘
   └──────────────────┘
```

### 2.2 Generator — `GRUGenerator`

**File:** `peptidegen/models/generator.py`

Bộ sinh chính là `GRUGenerator`, kế thừa từ base class `PeptideGenerator`:

```
Input:
  z  ∈ ℝ^{B × latent_dim}       # Latent noise vector
  c  ∈ ℝ^{B × condition_dim}    # Condition features (8 physicochemical)

Architecture:
  1. Latent projection: MLP(z ⊕ c) → h₀ ∈ ℝ^{B × hidden_dim}
  2. Embedding: AA token → ℝ^{embedding_dim}
  3. GRU (num_layers=2, hidden_dim=256, bidirectional=True*)
     *bidirectional for resumed checkpoints; unidirectional for new runs
  4. SelfAttention (attention head over GRU outputs)
  5. Output projection: Linear → ℝ^{vocab_size=24}

Output: logits ∈ ℝ^{B × L × 24}  (soft one-hot distribution)

Decoding modes:
  - Teacher forcing: during training (target provided)
  - Autoregressive: during inference (token-by-token)
  - Sampling: temperature, top-k, top-p (nucleus sampling)
```

**Các biến thể Generator:**

| Model | Tham số | Đặc điểm |
|---|---|---|
| `GRUGenerator` | ~2.4M | Autoregressive GRU + SelfAttention, gradient checkpointing |
| `LSTMGenerator` | ~2.5M | LSTM với cell state + condition fusion |
| `TransformerGenerator` | ~3M | Multi-head attention, positional encoding |
| `ESM2ConditionedGenerator` | ~10M+ | Điều kiện hóa bằng ESM2 embedding |

**Kỹ thuật quan trọng trong Generator:**
- **Gradient Checkpointing**: `use_gradient_checkpointing=True` để giảm VRAM khi hidden_dim lớn
- **Condition Fusion**: concat(z, c) → Linear → tanh cho hidden state ban đầu
- **Autoregressive sampling**: chọn token tại mỗi bước theo phân phối softmax(logits/T)

### 2.3 Discriminator — `CNNDiscriminator`

**File:** `peptidegen/models/discriminator.py`

```
Input: x ∈ ℝ^{B × L} (token indices) hoặc ℝ^{B × L × V} (soft logits)

Architecture:
  1. Embedding: token → ℝ^{embedding_dim}
  2. Parallele 1D-Conv branches (TextCNN style):
     - Conv1D(in=embedding_dim, out=64,  kernel=3) + ReLU + MaxPool
     - Conv1D(in=embedding_dim, out=128, kernel=5) + ReLU + MaxPool
     - Conv1D(in=embedding_dim, out=256, kernel=7) + ReLU + MaxPool
  3. Feature concatenation: [64 + 128 + 256] = 448-dim
  4. Minibatch Standard Deviation (anti mode-collapse)
  5. MLP classifier: 448+1 → 256 → 1 (logit score)

Regularization:
  - Spectral Normalization trên tất cả Linear/Conv layers
  - Label Smoothing (0.15 cho real labels)
  - Instance Noise (σ=0.08) thêm vào input D

Output: score ∈ ℝ^{B × 1}
```

**Các biến thể Discriminator:**

| Model | Đặc điểm |
|---|---|
| `CNNDiscriminator` | Multi-kernel TextCNN, Spectral Norm, Minibatch Std |
| `RNNDiscriminator` | Bidirectional GRU + Attention, Spectral Norm |
| `HybridDiscriminator` | CNN + RNN kết hợp, tổng hợp hai luồng feature |

---

## 3. Các Kỹ thuật Chính

### 3.1 ESM2 Integration — Protein Language Model

**File:** `peptidegen/models/esm2_embedder.py`

ESM2 (Evolutionary Scale Modeling 2) là mô hình ngôn ngữ protein được huấn luyện trên 250M protein sequences từ UniRef50 database.

```
ESM2Embedder:
  - Model: esm2_t33_650M_UR50D (33 layers, 1280-dim, 650M params)
  - Weights: FROZEN (freeze_esm=True) — chỉ dùng làm feature extractor
  - Pooling: mean pooling qua token dimension → ℝ^{1280}
  - Projection: LightweightESMProjector
      Linear(1280 → 640) + GELU + Dropout + Linear(640 → 128) + LayerNorm

Sử dụng:
  sequences: List[str] → ESM2 tokenize → forward → mean pool → project
  → embedding ∈ ℝ^{B × 128}
```

**Bảng ESM2 Model Variants:**

| Model | Layers | Params | Embed Dim | GPU RAM |
|---|---|---|---|---|
| `esm2_t6_8M_UR50D` | 6 | 8M | 320 | ~2 GB |
| `esm2_t12_35M_UR50D` | 12 | 35M | 480 | ~4 GB |
| `esm2_t30_150M_UR50D` | 30 | 150M | 640 | ~8 GB |
| `esm2_t33_650M_UR50D` | 33 | **650M** | **1280** | **~16 GB** |
| `esm2_t36_3B_UR50D` | 36 | 3B | 2560 | ~40 GB |

### 3.2 Graph Attention Network (GAT) — Structure Evaluator

**File:** `peptidegen/models/structure_evaluator.py`

GAT mô hình hóa peptide như một **đồ thị tuyến tính** trong đó mỗi amino acid là một node, các cạnh kết nối các residue lân cận trong cửa sổ sliding window.

```
GraphAttentionLayer:
  Input: node features h ∈ ℝ^{B × N × d_in}
         adjacency matrix A ∈ {0,1}^{B × N × N}  (window_size=3)
  
  Attention coefficient:
    e_ij = LeakyReLU(a^T [Wh_i ‖ Wh_j])  (LeakyReLU α=0.2)
    α_ij = softmax_j(e_ij)               (masked by A)
  
  Update:
    h'_i = σ(Σ_j α_ij · Wh_j)
  
  Multi-head: concat K heads → projection

LightweightGAT:
  num_layers: 2
  num_heads: 4
  hidden_dim: 64
  output_dim: 32

ESM2StructureEvaluator:
  ESM2Embedder → token embeddings (B, L, 1280)
  → Project (B, L, 128)
  → LightweightGAT (B, L, 32)
  → Global mean pool (B, 32)
  → MLP → stability_score ∈ ℝ^{B × 1}
```

### 3.3 Minibatch Discrimination

**File:** `peptidegen/models/discriminator.py` — `CNNDiscriminator`

Kỹ thuật chống mode collapse: tính **Minibatch Standard Deviation** để cung cấp thông tin về diversity của batch cho Discriminator.

```python
# Tính std theo batch dimension, append vào feature map
minibatch_std = x.std(dim=0, keepdim=True).mean().expand(B, 1)
x = torch.cat([x, minibatch_std], dim=-1)
# → feature dim: 448 → 449
```

### 3.4 Spectral Normalization

Áp dụng cho **tất cả Linear và Conv layers** trong Discriminator để ràng buộc Lipschitz constant về 1, ổn định training GAN:

```python
nn.utils.spectral_norm(nn.Linear(in, out))
nn.utils.spectral_norm(nn.Conv1d(in, out, k))
```

### 3.5 Adaptive Discriminator Training

**File:** `peptidegen/training/trainer.py` — `GANTrainer`

```
Mỗi training step:
  gap = D_real_loss - D_fake_loss

  Nếu gap > d_threshold (2.0):
    → Skip D update (D quá mạnh)
    → Tăng G steps lên g_steps × g_steps_boost

  Nếu gap ≤ d_threshold:
    → Cập nhật D bình thường (d_steps=1)
    → Cập nhật G (g_steps=8)
```

### 3.6 Mixed Precision Training (AMP)

Sử dụng `torch.cuda.amp` với `GradScaler` để:
- Giảm 50% memory usage (float16 thay float32)
- Tăng tốc ~2x trên Tensor Core GPUs
- `use_amp=True` trong config

### 3.7 Conditional Generation

**File:** `peptidegen/training/trainer.py` — `ConditionalGANTrainer`

```
Condition vector c ∈ ℝ^{B × 8}  (8 physicochemical features)

Fusion trong Generator:
  z_fused = concat(z, c)          # ℝ^{B × (latent_dim + condition_dim)}
  h₀ = tanh(Linear(z_fused))     # Initial hidden state

ConditionalGANTrainer:
  Inherits GANTrainer
  Adds: feature_loss_weight=0.1
  Passes conditions tensor qua _generate()
```

---

## 4. Hàm Mất mát

**File:** `peptidegen/training/losses.py`

Tổng loss của Generator:

```
L_G = w_adv · L_adversarial
    + w_rec · L_reconstruction
    + w_div · L_diversity
    + w_ngram · L_ngram
    + w_len · L_length_penalty
    + w_fm · L_feature_matching
    + w_stab · L_stability_bias
```

### 4.1 Adversarial Loss

**Binary Cross Entropy** với label smoothing:

```
L_adv_D = BCE(D(x_real), smooth_real) + BCE(D(x_fake), 0)
L_adv_G = BCE(D(x_fake), 1)

smooth_real = 1 - label_smoothing = 0.85
```

Noise injection vào D input: `x_noisy = x + N(0, noise_std²)` với `noise_std=0.08`

### 4.2 Diversity Loss — `DiversityLoss`

Ba thành phần chống mode collapse:

```
1. Token Entropy (per position):
   H_pos = -Σ_v p_v · log(p_v + ε)    over vocab dimension
   L_entropy = -mean(H_pos)             → minimize → maximize entropy

2. Batch Similarity:
   probs_mean = mean(softmax(logits), dim=0)   # Mean distribution
   L_batch_sim = cosine_sim(probs_i, probs_mean)  → penalize similarity

3. Pairwise Distance:
   Sample pairs (i,j) from batch
   L_pairwise = ReLU(margin - ||logits_i - logits_j||²)
   margin = 1.0

L_diversity = w_ent·H + w_batch·L_batch_sim + w_pair·L_pairwise
weights: (0.3, 0.3, 0.4)
Total weight in training: diversity_weight = 1.4
```

### 4.3 N-gram Diversity Loss — `NgramDiversityLoss`

Phạt các motif lặp lại ở cấp độ bigram và trigram:

```
Bigram joint distribution:
  p(a,b) = Σ_t softmax(logits[:,t,:]) ⊗ softmax(logits[:,t+1,:])
  H_bigram = -Σ p(a,b) · log(p(a,b) + ε)
  
Trigram: tương tự với t, t+1, t+2

L_ngram = bigram_weight · (1 - H_bigram/log(V²))
        + trigram_weight · (1 - H_trigram/log(V³))
weights: (0.5, 0.5)
Total weight: ngram_weight = 0.45
```

### 4.4 Length Penalty Loss — `LengthPenaltyLoss`

Giải quyết length collapse bằng EOS cumulative probability supervision:

```
eos_prob_t = softmax(logits[:,t,:])[eos_idx]       # P(EOS at position t)
cum_eos_t  = 1 - Π_{s≤t} (1 - eos_prob_s)         # P(EOS by position t)

Target:
  t < target_min (=10): cum_eos_t should be ≈ 0
  t > target_max (=30): cum_eos_t should be ≈ 1

L_early = Σ_{t<10}  early_weight · cum_eos_t²
L_late  = Σ_{t>30}  late_weight  · (1 - cum_eos_t)²
L_length = (L_early + L_late) / seq_len
Total weight: length_penalty_weight = 0.5
```

### 4.5 Feature Matching Loss — `FeatureMatchingLoss`

Ổn định training bằng cách khớp intermediate features giữa real và fake:

```
L_fm = ||E[D_feat(x_real)] - E[D_feat(x_fake)]||²₂

Lấy features từ CNNDiscriminator.get_feature() trước output layer
Total weight: feature_matching_weight = 0.2
```

### 4.6 Reconstruction Loss — `ReconstructionLoss`

```
L_rec = CrossEntropy(logits, real_targets, ignore_index=PAD)
Hoặc MSE với soft one-hot targets (reconstruction_loss="mse")
Total weight: reconstruction_weight = 0.3
```

### 4.7 Stability Bias Loss — `StabilityBiasLoss`

Soft nudge hướng đến Instability Index < 30:

```
II_proxy = heuristic_instability_score(generated_tokens)
L_stab = ReLU(II_proxy - target_stability_ii) / target_stability_ii
Total weight: stability_weight = 0.1
```

---

## 5. Dữ liệu và Dataset

### 5.1 Thống kê Dataset

| Split | Sequences | File |
|---|---|---|
| Train | **129,121** | `dataset/train.fasta` + `train.csv` |
| Validation | **27,669** | `dataset/val.fasta` + `val.csv` |
| Test | — | `dataset/test.fasta` + `test.csv` |
| **Tổng** | **~156,790** | |

### 5.2 Format Dữ liệu

- **FASTA files**: chuỗi amino acid raw sequence
- **CSV files**: kèm 8 cột đặc trưng sinh hóa (xem Section 6)
- **Vocabulary**: 24 tokens = 20 standard amino acids + `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- **Độ dài**: min=5, max=50 residues

### 5.3 Data Pipeline

**Files:** `peptidegen/data/`

```
dataset.py:
  PeptideDataset          — FASTA-only loader
  ConditionalPeptideDataset — CSV + FASTA loader với 8-dim condition

dataloader.py:
  create_dataloader()     — batch_size=8192, num_workers=8, pin_memory=True

features.py:
  FeatureExtractor        — Tính 8 đặc trưng từ chuỗi amino acid
  Normalization           — z-score normalization của features

vocabulary.py:
  VOCAB                   — Global singleton vocab object
  encode(seq) / decode(ids)
```

---

## 6. Đặc trưng Điều kiện Hóa

8 đặc trưng sinh hóa dùng để điều kiện hóa generation:

| Feature | Mô tả | Ngưỡng/Ý nghĩa | File tính |
|---|---|---|---|
| `instability_index` | Chỉ số bất ổn định (Guruprasad 1990) | < 40 = stable | `stability.py` |
| `therapeutic_score` | Điểm tiềm năng điều trị (0–10) | > 7 = high | `metrics.py` |
| `hemolytic_score` | Độ độc tế bào hồng cầu (0–10) | < 3 = safe | `metrics.py` |
| `aliphatic_index` | Chỉ số aliphatic (Ikai 1980) | Cao = bền nhiệt | `stability.py` |
| `hydrophobic_moment` | Amphipathicity (Eisenberg scale) | Cao → AMP tốt | `metrics.py` |
| `gravy` | Grand Average Hydropathicity | < 0 = hydrophilic | `stability.py` |
| `charge_at_pH7` | Điện tích thực tại pH 7.0 | > 0 = cationic AMP | `stability.py` |
| `aromaticity` | Tỉ lệ Phe/Trp/Tyr | — | `stability.py` |

---

## 7. Đánh giá và Chỉ số Chất lượng

**Files:** `peptidegen/evaluation/`

### 7.1 Training Metrics (theo dõi mỗi epoch)

| Metric | Ký hiệu | Mục tiêu | Công thức |
|---|---|---|---|
| G Validation Loss | `val_g_loss` | **< 0.35** | BCE(D(G(z)), 1) |
| D–G Gap | `gap` | **< 0.25** | D_real_loss − D_fake_loss |
| Token Entropy | `ent` | **> 2.80** | −Σ p log p (per token) |
| N-gram Diversity | `ngram` | **> 0.08** | Bigram/Trigram joint entropy |
| Length Penalty | `len_pen` | **< 0.005** | EOS cumulative prob loss |
| Stability Loss | `stab` | minimize | ReLU(II − 30) |

### 7.2 Sequence Quality Metrics

**File:** `peptidegen/evaluation/stability.py`

```python
calculate_instability_index(seq)   # Guruprasad DIWV dipeptide weights
calculate_gravy(seq)                # Kyte-Doolittle hydropathy
calculate_aliphatic_index(seq)     # Ikai: 100×(xA + 2.9×xV + 3.9×(xI+xL))
calculate_isoelectric_point(seq)   # Bisection method, pKa table
calculate_charge_at_pH(seq, pH=7)  # Henderson-Hasselbalch
calculate_molecular_weight(seq)    # Residue MW table
calculate_aromaticity(seq)         # freq(F) + freq(W) + freq(Y)
calculate_secondary_structure_propensity(seq)  # Helix/Sheet Chou-Fasman
```

### 7.3 AMP-specific Metrics

**File:** `peptidegen/evaluation/metrics.py`

```python
calculate_hydrophobicity(seq)      # Eisenberg scale
calculate_hydrophobic_moment(seq)  # α-helix angle 100°, window=11
calculate_net_charge(seq)          # pH-aware charge
calculate_hemolytic_score(seq)     # Hemolytic propensity scale
calculate_therapeutic_score(seq)   # Composite: charge + hydrophobicity + amphipathicity
estimate_amp_probability(seq)      # Heuristic ML-based AMP probability
analyze_amp_properties(seqs)       # Batch analysis với summary statistics
```

### 7.4 Diversity Metrics

```python
calculate_diversity_metrics(seqs):
  - uniqueness_ratio     = |unique_seqs| / |total_seqs|
  - ngram_diversity_2    = |unique_bigrams| / |total_bigrams|
  - ngram_diversity_3    = |unique_trigrams| / |total_trigrams|
  - avg_levenshtein      = mean pairwise edit distance (sampled)
  - entropy_aa           = Shannon entropy of AA distribution

detect_mode_collapse(seqs):
  - entropy_threshold: 0.3  (entropy < threshold → collapse)
  - aa_usage_threshold: 0.5 (fraction of AAs used < threshold → collapse)
  - repetition_ratio: fraction of duplicated subsequences
```

### 7.5 Quality Filter

**File:** `peptidegen/evaluation/quality_filter.py`

Filter pipeline sau khi sinh:
```
1. Độ dài: 10 ≤ len ≤ 30
2. Instability Index < 40 (stable)
3. min_stability_score ≥ 0.5
4. Chỉ chứa 20 standard amino acids
5. Không có run quá dài (> 5 ký tự giống nhau liên tiếp)
```

---

## 8. Yêu cầu Phần cứng & GPU

### 8.1 GPU thực tế sử dụng

```
GPU:     NVIDIA Quadro RTX 6000
VRAM:    23.5 GB GDDR6
CUDA:    12.x
Batch:   8,192 sequences/batch
Speed:   ~44 giây/epoch (15 batches × 8192 = 122,880 seqs)
```

### 8.2 Yêu cầu tối thiểu

| Cấu hình | GPU | VRAM | Batch | ESM2 Model |
|---|---|---|---|---|
| **Minimum** | GTX 1080 Ti | 11 GB | 512 | esm2_t6_8M (8M) |
| **Recommended** | RTX 3090 | 24 GB | 4096 | esm2_t12_35M |
| **Optimal** | RTX 3090 / A100 | 24–40 GB | 8192 | esm2_t33_650M |
| **Production** | A100 80GB | 80 GB | 16384 | esm2_t36_3B |

### 8.3 Tối ưu Memory

| Kỹ thuật | Cấu hình | Tiết kiệm |
|---|---|---|
| Mixed Precision (AMP) | `use_amp: true` | ~50% VRAM |
| Gradient Checkpointing | `gradient_checkpointing: true` | ~30% VRAM |
| ESM2 Frozen | `freeze_esm: true` | Không train 650M params |
| Pin Memory | `pin_memory: true` | Tăng PCIe bandwidth |
| Num Workers | `num_workers: 8` | Tăng CPU→GPU throughput |

---

## 9. Thư viện và Dependencies

### 9.1 Core Dependencies

| Thư viện | Phiên bản | Vai trò |
|---|---|---|
| **PyTorch** | ≥ 2.0.0 | Deep learning framework chính |
| **fair-esm** | ≥ 2.0.0 | ESM2 protein language model (Facebook AI) |
| **torch-geometric** | ≥ 2.3.0 | Graph Attention Network (GAT) layers |
| **NumPy** | ≥ 1.24.0 | Numerical computing |
| **SciPy** | ≥ 1.10.0 | Scientific computing, optimization |
| **Pandas** | ≥ 2.0.0 | CSV data loading và xử lý |
| **PyYAML** | ≥ 6.0 | Configuration file parsing |
| **tqdm** | ≥ 4.65.0 | Progress bars |

### 9.2 Optional Dependencies

| Thư viện | Vai trò |
|---|---|
| matplotlib ≥ 3.7 | Visualization của training curves |
| seaborn ≥ 0.12 | Statistical plots |
| biopython ≥ 1.81 | Advanced bioinformatics analysis |
| pytest ≥ 7.3 | Unit testing |

### 9.3 Cài đặt

```bash
# Conda environment (khuyến nghị)
conda env create -f environment.yml
conda activate peptidegen

# pip
pip install -r requirements.txt

# ESM2 (nếu cần tải model separately)
python -c "import esm; esm.pretrained.esm2_t33_650M_UR50D()"
```

---

## 10. Cấu hình Siêu tham số

### 10.1 Kiến trúc Model

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `latent_dim` | 128 | Chiều noise vector z |
| `embedding_dim` | 128 | Chiều AA embedding (config mới: 128, checkpoint cũ: 64) |
| `hidden_dim` | 512 | Chiều ẩn GRU (config mới: 512, checkpoint cũ: 256) |
| `num_layers` | 2 | Số GRU layers |
| `dropout` | 0.2 | Dropout rate |
| `condition_dim` | 8 | Chiều condition vector (8 physicochemical features) |
| `vocab_size` | 24 | 20 AA + 4 special tokens |
| `max_seq_length` | 50 | Max peptide length |

### 10.2 Training Hyperparameters

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `batch_size` | 8192 | Sequences per batch |
| `learning_rate` | 3e-5 | Generator LR (Adam) |
| `lr_discriminator` | 1e-5 | Discriminator LR (Adam) |
| `beta1` | 0.5 | Adam β₁ |
| `beta2` | 0.999 | Adam β₂ |
| `weight_decay` | 1e-4 | L2 regularization |
| `grad_clip` | 1.0 | Gradient clipping |
| `g_steps` | 8 | G updates per D update |
| `d_steps` | 1 | D updates per iteration |
| `label_smoothing` | 0.15 | Real label smoothing |
| `noise_std` | 0.08 | Instance noise trong D |
| `d_threshold` | 2.0 | D skip threshold (adaptive) |
| `patience` | 50 | Early stopping patience |

### 10.3 Loss Weights

| Loss | Weight | Chú thích |
|---|---|---|
| `adversarial_weight` | 0.5 | GAN adversarial BCE |
| `diversity_weight` | **1.4** | Entropy + batch sim + pairwise |
| `entropy_weight` | 0.8 | Thành phần entropy trong diversity |
| `ngram_weight` | 0.45 | Bigram + Trigram diversity |
| `length_penalty_weight` | 0.5 | EOS supervision |
| `reconstruction_weight` | 0.3 | MSE/CrossEntropy với real sequences |
| `feature_matching_weight` | 0.2 | D feature matching |
| `stability_weight` | 0.1 | Stability bias loss |

---

## 11. Kết quả Thực nghiệm

### 11.1 Kết quả Training (Training Run #8 — stable8)

**Hardware:** Quadro RTX 6000 (23.5 GB), CUDA  
**Dataset:** 129,121 train / 27,669 val  
**Resumed from:** epoch 187 (checkpoint_epoch_187.pt)

| Epoch | G Loss (val) | D Loss (val) | D–G Gap | Entropy | N-gram | Len Pen |
|---|---|---|---|---|---|---|
| 189 | **0.4072** ⭐ | 2.0525 | 0.232 | 2.877 | 0.0100 | 0.0038 |
| 190 | 0.4128 | 2.0050 | 0.217 | 2.881 | 0.0097 | 0.0005 |
| 191 | 0.4132 | 2.0004 | 0.225 | 2.884 | 0.0097 | 0.0005 |
| 195 | 0.4137 | 1.9972 | 0.246 | 2.888 | 0.0094 | 0.0004 |
| 200 | 0.4137 | 1.9971 | 0.248 | 2.890 | 0.0091 | 0.0003 |
| 206 | 0.4136 | 1.9968 | 0.248 | 2.892 | 0.0078 | 0.0003 |

**Best checkpoint:** epoch 189, `val_g_loss = 0.4072`

### 11.2 Đánh giá theo Mục tiêu Đề ra

| Metric | Mục tiêu | Đạt được | Trạng thái |
|---|---|---|---|
| G val loss | < 0.35 | **0.407** | 🔄 Tiếp tục cải thiện |
| D–G gap | < 0.25 | **0.232** ✓ | ✅ Đạt (epoch 189) |
| Entropy | > 2.80 | **2.892** | ✅ Đạt |
| N-gram diversity | > 0.08 | **0.0078** | 🔄 Còn thấp |
| Length penalty | < 0.005 | **0.0003** | ✅ Đạt |

### 11.3 Thông số Mô hình

| Component | Parameters |
|---|---|
| Generator (GRUGenerator) | **2,392,344** |
| Discriminator (CNNDiscriminator) | **285,122** |
| **Tổng** | **2,677,466** |
| ESM2-650M (frozen evaluator) | 650,000,000 |

### 11.4 Tốc độ Training

```
Epoch time:    ~44 giây/epoch
Batches:       15 batches (129,121 / 8,192 ≈ 15)
Throughput:    ~8,192 × 15 / 44 ≈ 2,793 sequences/second
Checkpoint:    saved every 10 epochs (~7.3 phút/checkpoint)
```

---

## 12. Cấu trúc Dự án

```
LightweightPeptideGen/
├── train.py                    # Entry point: training GAN
├── generate.py                 # Entry point: sinh peptide từ checkpoint
├── evaluate.py                 # Entry point: đánh giá chất lượng
│
├── peptidegen/                 # Core library package
│   ├── __init__.py             # Public API exports
│   ├── constants.py            # DIPEPTIDE_WEIGHTS, pKa tables, ...
│   ├── utils.py                # load_config(), set_seed(), ...
│   ├── logger_config.py        # Logging setup
│   │
│   ├── data/                   # Data pipeline
│   │   ├── vocabulary.py       # VOCAB (24 tokens), encode/decode
│   │   ├── dataset.py          # PeptideDataset, ConditionalPeptideDataset
│   │   ├── dataloader.py       # create_dataloader()
│   │   └── features.py         # FeatureExtractor, normalization
│   │
│   ├── models/                 # Neural network models
│   │   ├── components.py       # PositionalEncoding, MultiHeadAttention, SelfAttention, ResidualBlock
│   │   ├── generator.py        # GRUGenerator, LSTMGenerator, TransformerGenerator
│   │   ├── discriminator.py    # CNNDiscriminator, RNNDiscriminator, HybridDiscriminator
│   │   ├── esm2_embedder.py    # ESM2Embedder, LightweightESMProjector, ESM2StructureEvaluator
│   │   ├── esm2_generator.py   # ESM2ConditionedGenerator
│   │   ├── structure_evaluator.py  # GraphAttentionLayer, LightweightGAT, StructureEvaluator
│   │   └── feature_loss.py     # FeatureBasedLoss (stability + therapeutic + toxicity)
│   │
│   ├── training/               # Training logic
│   │   ├── trainer.py          # GANTrainer, ConditionalGANTrainer
│   │   └── losses.py           # DiversityLoss, NgramDiversityLoss, LengthPenaltyLoss,
│   │                           #   FeatureMatchingLoss, ReconstructionLoss, StabilityBiasLoss
│   │
│   ├── inference/              # Generation/sampling
│   │   └── sampler.py          # PeptideSampler (from_checkpoint, sample, save_fasta)
│   │
│   └── evaluation/             # Evaluation metrics
│       ├── stability.py        # calculate_instability_index, GRAVY, aliphatic_index, pI, ...
│       ├── metrics.py          # Diversity metrics + AMP metrics (merged from amp_metrics.py)
│       └── quality_filter.py   # QualityFilter pipeline
│
├── config/
│   └── config.yaml             # Toàn bộ hyperparameter config
│
├── dataset/                    # Training data (FASTA + CSV)
│   ├── train.fasta / train.csv
│   ├── val.fasta / val.csv
│   └── test.fasta / test.csv
│
├── checkpoints/                # Saved model checkpoints (.pt files)
├── logs/                       # Training logs (train_stable8.log, ...)
├── tests/                      # Unit tests (pytest)
│   ├── test_quick.py
│   └── test_conditional.py
├── tools/                      # Utility scripts
│   ├── process_data.py         # Data preprocessing
│   ├── analyze_data.py         # Dataset analysis
│   ├── validate_data.py        # Data validation
│   └── export.py               # Model export
├── requirements.txt
└── environment.yml             # Conda environment spec
```

---

## 13. Hướng dẫn Sử dụng

### 13.1 Cài đặt môi trường

```bash
conda env create -f environment.yml
conda activate peptidegen
```

### 13.2 Huấn luyện

```bash
# Standard training
python train.py --config config/config.yaml

# Conditional training (sử dụng 8 physicochemical features)
python train.py --config config/config.yaml --conditional

# Resume từ checkpoint
python train.py --config config/config.yaml --conditional \
    --resume checkpoints/checkpoint_epoch_189.pt \
    --epochs 500 \
    --fresh-optimizer           # reset Adam state nếu có NaN

# Với custom hyperparameters
python train.py --config config/config.yaml --conditional \
    --epochs 200 --batch-size 4096 --lr 0.00003
```

### 13.3 Sinh Peptide

```bash
# Basic generation
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --num 1000 \
    --temperature 1.0 \
    --top-p 0.9

# Với quality filter
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --num 1000 \
    --filter \
    --min-stability 0.5 \
    --max-instability 40 \
    --output filtered_peptides.fasta

# Diverse generation
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --num 500 --diverse --temperature 0.8
```

### 13.4 Đánh giá

```bash
# Đánh giá cơ bản
python evaluate.py --input generated.fasta

# So sánh với training data
python evaluate.py \
    --input generated.fasta \
    --reference dataset/train.fasta \
    --output report.json
```

### 13.5 API Python

```python
from peptidegen import GANTrainer, ConditionalGANTrainer
from peptidegen import GRUGenerator, CNNDiscriminator
from peptidegen import PeptideSampler, VOCAB, load_config

# Load config
config = load_config('config/config.yaml')

# Build models
generator = GRUGenerator(
    vocab_size=24,
    embedding_dim=64,
    hidden_dim=256,
    latent_dim=128,
    num_layers=2,
    condition_dim=8,
    use_attention=True,
)
discriminator = CNNDiscriminator(
    vocab_size=24,
    embedding_dim=64,
    hidden_dim=256,
    num_filters=[64, 128, 256],
    kernel_sizes=[3, 5, 7],
    use_spectral_norm=True,
    use_minibatch_std=True,
)

# Train
trainer = ConditionalGANTrainer(generator, discriminator, config['training'])
trainer.fit(train_loader, val_loader, epochs=200, checkpoint_dir='checkpoints')

# Generate
sampler = PeptideSampler.from_checkpoint('checkpoints/best_model.pt')
sequences = sampler.sample(n=1000, temperature=0.8, top_p=0.9)
sampler.save_fasta(sequences, 'generated.fasta')
```

---

## 14. Tài liệu Tham khảo

### Kiến trúc & Kỹ thuật GAN

1. **Goodfellow et al.** (2014). Generative Adversarial Nets. *NeurIPS*.
2. **Salimans et al.** (2016). Improved Techniques for Training GANs. *NeurIPS*. *(Feature Matching, Minibatch Discrimination)*
3. **Miyato et al.** (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR*.
4. **Gulrajani et al.** (2017). Improved Training of Wasserstein GANs. *NeurIPS*.

### Mô hình Ngôn ngữ Protein

5. **Lin et al.** (2022). Language models of protein sequences at the scale of evolution enable accurate structure prediction. *bioRxiv*. *(ESM2)*
6. **Rives et al.** (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS*. *(ESM)*

### Graph Neural Networks

7. **Veličković et al.** (2018). Graph Attention Networks. *ICLR*. *(GAT)*

### Sinh hóa Peptide

8. **Guruprasad et al.** (1990). Correlation between stability of a protein and its dipeptide composition. *Protein Engineering*. *(Instability Index)*
9. **Ikai** (1980). Thermostability and Aliphatic Index of Globular Proteins. *Journal of Biochemistry*. *(Aliphatic Index)*
10. **Kyte & Doolittle** (1982). A simple method for displaying the hydropathic character of a protein. *Journal of Molecular Biology*. *(GRAVY/Hydropathy)*
11. **Eisenberg et al.** (1982). Hydrophobic moments and protein structure. *Faraday Symposia*.

### Antimicrobial Peptides

12. **Boman** (2003). Antibacterial peptides: basic facts and emerging concepts. *Journal of Internal Medicine*.
13. **Hancock & Sahl** (2006). Antimicrobial and host-defense peptides as new anti-infective therapeutic strategies. *Nature Biotechnology*.

### Sinh tạo Peptide với Deep Learning

14. **Tucs et al.** (2023). Generating amphibian-inspired antimicrobial peptides with a recurrent neural network. *PLOS ONE*.
15. **Dean et al.** (2021). Deep learning for antimicrobial peptide discovery using generative models. *Briefings in Bioinformatics*.

---

## Appendix A: Checkpoint Format

```python
checkpoint = {
    'epoch': int,
    'generator_state_dict': dict,
    'discriminator_state_dict': dict,
    'g_optimizer_state_dict': dict,
    'd_optimizer_state_dict': dict,
    'scaler_state_dict': dict,      # AMP GradScaler
    'best_val_metric': float,
    'history': {
        'g_loss': [...],
        'd_loss': [...],
        'val_g_loss': [...],
        'val_d_loss': [...],
    },
    'model_config': {               # Lưu kiến trúc để reload đúng
        'vocab_size': 24,
        'embedding_dim': 64,
        'hidden_dim': 256,
        'latent_dim': 128,
        'max_length': 50,
        'num_layers': 2,
        'dropout': 0.2,
        'condition_dim': 8,
        'bidirectional': True,
        'use_attention': True,
        'pad_idx': 0,
        'sos_idx': 1,
        'eos_idx': 2,
    }
}
```

## Appendix B: Vocabulary

```
Index 0: <PAD>   — padding token
Index 1: <SOS>   — start of sequence
Index 2: <EOS>   — end of sequence
Index 3: <UNK>   — unknown amino acid
Index 4–23: A C D E F G H I K L M N P Q R S T V W Y
             (20 standard amino acids, alphabetical)
```

---

*Tài liệu này được tổng hợp từ phân tích toàn bộ mã nguồn LightweightPeptideGen. Cập nhật lần cuối: Tháng 2/2026.*
