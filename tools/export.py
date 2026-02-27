#!/usr/bin/env python
"""
Export trained models to separate files.

Usage:
    python tools/export.py --checkpoint checkpoints/best_model.pt --output exported_models/
"""

import argparse
import json
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from peptidegen.utils import load_config
from peptidegen.data import VOCAB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Export models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--output', type=str, default='exported_models', help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    logger.info(f"Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    
    # Export generator
    torch.save({
        'state_dict': ckpt['generator'],
        'config': ckpt.get('config', {}),
    }, output_dir / 'generator.pt')
    logger.info(f"Exported: generator.pt")
    
    # Export discriminator
    torch.save({
        'state_dict': ckpt['discriminator'],
        'config': ckpt.get('config', {}),
    }, output_dir / 'discriminator.pt')
    logger.info(f"Exported: discriminator.pt")
    
    # Export config
    config = {
        'vocab_size': VOCAB.vocab_size,
        'epoch': ckpt.get('epoch', 0),
        'training_config': ckpt.get('config', {}),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Exported: config.json")
    
    # Create usage example
    usage_code = '''"""
Usage example for exported models.
"""
import torch
from peptidegen.models import GRUGenerator
from peptidegen.data import VOCAB

# Load generator
ckpt = torch.load('generator.pt', map_location='cpu')
config = ckpt['config']

generator = GRUGenerator(
    vocab_size=VOCAB.vocab_size,
    embedding_dim=config.get('embedding_dim', 64),
    hidden_dim=config.get('hidden_dim', 256),
    latent_dim=config.get('latent_dim', 128),
)
generator.load_state_dict(ckpt['state_dict'])
generator.eval()

# Generate
z = torch.randn(10, generator.latent_dim)
with torch.no_grad():
    logits = generator(z)
    tokens = logits.argmax(dim=-1)
    
# Decode
for i in range(tokens.size(0)):
    seq = VOCAB.decode(tokens[i].tolist())
    print(seq)
'''
    with open(output_dir / 'usage_example.py', 'w') as f:
        f.write(usage_code)
    logger.info(f"Exported: usage_example.py")
    
    logger.info(f"All models exported to {output_dir}/")


if __name__ == '__main__':
    main()
