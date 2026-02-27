# LightweightPeptideGen

A lightweight GAN-based framework for generating structurally stable antimicrobial peptides.

## Features

- **GAN Architecture**: GRU/LSTM/Transformer generators with CNN discriminators
- **Anti-Mode-Collapse**: Diversity loss, adaptive D training, entropy regularization
- **ESM2 Integration**: Optional protein language model for enhanced evaluation
- **Conditional Generation**: Generate peptides with desired properties
- **Quality Filtering**: Filter by stability, therapeutic potential, and safety

## Project Structure

```
LightweightPeptideGen/
├── train.py              # Train the model
├── generate.py           # Generate sequences
├── evaluate.py           # Evaluate quality
│
├── peptidegen/           # Core library
│   ├── data/             # Data loading
│   │   ├── dataset.py    # PeptideDataset, ConditionalPeptideDataset
│   │   ├── dataloader.py # DataLoader utilities
│   │   ├── vocabulary.py # VOCAB: AA encoding
│   │   └── features.py   # Feature extraction
│   │
│   ├── models/           # Neural network models
│   │   ├── generator.py  # GRUGenerator, LSTMGenerator, etc.
│   │   ├── discriminator.py  # CNNDiscriminator, etc.
│   │   ├── esm2_embedder.py  # ESM2 integration (optional)
│   │   └── components.py # Shared components
│   │
│   ├── training/         # Training logic
│   │   ├── trainer.py    # GANTrainer, ConditionalGANTrainer
│   │   └── losses.py     # DiversityLoss, FeatureMatchingLoss
│   │
│   ├── inference/        # Generation
│   │   └── sampler.py    # PeptideSampler
│   │
│   └── evaluation/       # Evaluation metrics
│       ├── stability.py  # Instability index, GRAVY, etc.
│       ├── metrics.py    # Diversity metrics
│       └── amp_metrics.py # AMP-specific metrics
│
├── tools/                # Utility scripts
│   ├── process_data.py   # Data preprocessing
│   ├── analyze_data.py   # Data analysis
│   ├── validate_data.py  # Data validation
│   └── export.py         # Export models
│
├── config/
│   └── config.yaml       # Configuration
│
├── dataset/              # Training data
├── checkpoints/          # Saved models
└── tests/                # Unit tests
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python train.py --config config/config.yaml --conditional

# Generate
python generate.py --checkpoint checkpoints/best_model.pt --num 1000

# Evaluate
python evaluate.py --input generated.fasta
```

## Training

```bash
# Standard training
python scripts/train.py --config config/config.yaml

# Conditional training (with peptide properties)
python scripts/train.py --config config/config.yaml --conditional

# Resume from checkpoint
python scripts/train.py --resume checkpoints/best_model.pt --epochs 100
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0003 | Generator learning rate |
| `lr_discriminator` | 0.00002 | Discriminator learning rate (10-15x lower) |
| `g_steps` | 5 | Generator steps per iteration |
| `d_steps` | 1 | Discriminator steps per iteration |
| `diversity_weight` | 0.8 | Diversity loss weight |
| `adversarial_weight` | 0.4 | Adversarial loss weight |
| `label_smoothing` | 0.3 | Label smoothing factor |

## API Usage

```python
from src import GANTrainer, GRUGenerator, CNNDiscriminator
from src import PeptideSampler, load_config, VOCAB

# Training
config = load_config('config/config.yaml')
generator = GRUGenerator(vocab_size=VOCAB.vocab_size, ...)
discriminator = CNNDiscriminator(vocab_size=VOCAB.vocab_size, ...)

trainer = GANTrainer(generator, discriminator, config['training'])
trainer.fit(train_loader, epochs=100)

# Generation
sampler = PeptideSampler.from_checkpoint('checkpoints/best_model.pt')
sequences = sampler.sample(n=100, temperature=0.8, top_p=0.9)
sampler.save_fasta(sequences, 'generated.fasta')
```
    --checkpoint-dir checkpoints
```

### Generation

```bash
# Basic generation
python generate.py --checkpoint checkpoints/best_model.pt --num-sequences 1000

# With quality filtering
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --num-sequences 1000 \
    --filter \
    --min-stability 0.5 \
    --max-instability 40 \
    --output filtered_peptides.fasta

# Diverse generation
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --num-sequences 500 \
    --diverse \
    --temperature 0.8
```

### Evaluation

```bash
# Basic evaluation
python evaluate.py --input generated.fasta

# Compare with training data
python evaluate.py --input generated.fasta --reference dataset/train.fasta

# Save detailed report
python evaluate.py --input generated.fasta --output report.json
```

### Quality Filtering

```bash
# Filter generated peptides
python -m src.quality_filter \
    --input generated.fasta \
    --output filtered.fasta \
    --report quality_report.json
```

## Configuration

Key settings in `config/config.yaml`:

```yaml
model:
  latent_dim: 128
  embedding_dim: 64
  hidden_dim: 256
  num_layers: 2
  dropout: 0.1

generator:
  type: "GRU"              # GRU, LSTM, or Transformer
  condition_dim: 8         # Number of condition features
  esm_conditioned: false

discriminator:
  type: "CNN"              # CNN, RNN, or Hybrid

esm2:
  use_esm: true
  esm_model: "esm2_t6_8M_UR50D"  # Lightweight model for 4GB GPU
  freeze_esm: true
  projection_dim: 128

structure_evaluator:
  use_gat: true
  gat_heads: 4
  gat_hidden: 64

training:
  batch_size: 64
  num_epochs: 200
  learning_rate: 0.0002
  stability_weight: 0.3

feature_loss:
  enabled: true
  stability_weight: 0.3
  therapeutic_weight: 0.2
  toxicity_weight: 0.3

data:
  train_path: "dataset/train.fasta"
  train_csv: "dataset/train.csv"
  max_seq_length: 50
  min_seq_length: 5
```

## Conditional Features

When using `--conditional` mode, the model uses these features from CSV:

| Feature | Description |
|---------|-------------|
| `instability_index` | Protein stability (< 40 = stable) |
| `therapeutic_score` | Therapeutic potential score |
| `hemolytic_score` | Hemolytic toxicity score |
| `aliphatic_index` | Thermostability indicator |
| `hydrophobic_moment` | Amphipathicity measure |
| `gravy` | Hydropathicity score |
| `charge_at_pH7` | Net charge at physiological pH |
| `aromaticity` | Fraction of aromatic residues |

## Evaluation Metrics

### Stability Metrics
- **Instability Index (II)**: < 40 indicates stable protein
- **GRAVY**: Negative = hydrophilic, Positive = hydrophobic
- **Aliphatic Index**: Higher = better thermostability
- **Isoelectric Point (pI)**: pH at zero net charge

### Quality Metrics
- **Diversity Score**: Sequence diversity (1 - avg pairwise similarity)
- **Uniqueness Ratio**: Fraction of unique sequences
- **N-gram Diversity**: Bigram and trigram diversity
- **Mode Collapse Detection**: Entropy-based detection

## Model Variants

### Generators
| Model | Parameters | Best For |
|-------|------------|----------|
| GRUGenerator | ~2.4M | Short peptides, fast training |
| LSTMGenerator | ~2.5M | Similar to GRU with cell state |
| TransformerGenerator | ~3M | Better quality, slower |
| ESM2ConditionedGenerator | ~10M+ | Highest quality |

### ESM2 Models
| Model | Layers | Parameters | GPU Memory |
|-------|--------|------------|------------|
| esm2_t6_8M_UR50D | 6 | 8M | ~2GB |
| esm2_t12_35M_UR50D | 12 | 35M | ~4GB |
| esm2_t30_150M_UR50D | 30 | 150M | ~8GB |
| esm2_t33_650M_UR50D | 33 | 650M | ~16GB |

## References

- **ESM2**: Lin et al., "Language models of protein sequences at the scale of evolution enable accurate structure prediction", bioRxiv 2022
- **Instability Index**: Guruprasad et al., 1990
- **Aliphatic Index**: Ikai, 1980

## License

MIT License
