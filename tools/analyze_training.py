#!/usr/bin/env python
"""Analyze training metrics to diagnose mode collapse"""

import json
from pathlib import Path

def analyze_metrics():
    log_file = Path('checkpoints/logs/metrics_20260202_183307.jsonl')
    
    with open(log_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print("=" * 70)
    print("TRAINING ANALYSIS - Diagnosing Mode Collapse")
    print("=" * 70)
    
    # Group by epoch
    epochs_data = {}
    for d in data:
        e = d['epoch']
        if e not in epochs_data:
            epochs_data[e] = []
        epochs_data[e].append(d)
    
    total_epochs = len(epochs_data)
    last_epoch = max(epochs_data.keys())
    
    print(f"\nTotal epochs trained: {total_epochs}")
    print(f"Last epoch: {last_epoch}")
    print(f"Total steps: {data[-1]['global_step']}")
    
    # Analyze key metrics over time
    print("\n" + "=" * 70)
    print("METRICS PROGRESSION")
    print("=" * 70)
    print(f"{'Epoch':>6} | {'recon':>8} | {'adv':>8} | {'d_loss':>8} | {'d_real':>8} | {'d_fake':>8} | {'gap':>8}")
    print("-" * 70)
    
    key_epochs = [0, 5, 10, 20, 30, 40, 50, 60, 70, last_epoch]
    for e in key_epochs:
        if e in epochs_data:
            last = epochs_data[e][-1]
            gap = last['d_real'] - last['d_fake']
            print(f"{e:>6} | {last['recon_loss']:>8.3f} | {last['adv_loss']:>8.3f} | "
                  f"{last['d_loss']:>8.4f} | {last['d_real']:>8.3f} | {last['d_fake']:>8.3f} | {gap:>+8.3f}")
    
    # Diagnose issues
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    last = epochs_data[last_epoch][-1]
    first = epochs_data[0][-1]
    
    issues = []
    recommendations = []
    
    # 1. Check discriminator dominance
    gap = last['d_real'] - last['d_fake']
    if gap > 5:
        issues.append(f"❌ Discriminator quá mạnh: gap={gap:.2f} (d_real - d_fake)")
        recommendations.append("→ Giảm learning rate của discriminator (lr_discriminator)")
        recommendations.append("→ Tăng số bước generator (g_steps: 2 → 3)")
        recommendations.append("→ Thêm noise vào discriminator input")
    
    # 2. Check d_loss
    if last['d_loss'] < 0.2:
        issues.append(f"❌ D_loss quá thấp: {last['d_loss']:.4f} - Discriminator classify quá tốt")
        recommendations.append("→ Tăng label_smoothing (0.1 → 0.2)")
        recommendations.append("→ Thêm instance noise")
    
    # 3. Check adversarial loss trend
    adv_increase = last['adv_loss'] - first['adv_loss']
    if adv_increase > 3:
        issues.append(f"❌ Adv_loss tăng mạnh: {first['adv_loss']:.2f} → {last['adv_loss']:.2f}")
        recommendations.append("→ Generator không theo kịp discriminator")
        recommendations.append("→ Giảm adversarial_weight hoặc tăng recon_weight")
    
    # 4. Check reconstruction loss
    if last['recon_loss'] > 1.5:
        issues.append(f"⚠️ Recon_loss vẫn cao: {last['recon_loss']:.3f}")
        recommendations.append("→ Generator chưa học được cấu trúc sequence tốt")
    
    # 5. Check diversity_weight
    print("\n[Config Analysis]")
    print(f"  diversity_weight: 0.2 (trong config.yaml)")
    print(f"  learning_rate: 1e-5")
    print(f"  lr_discriminator: 5e-5")
    print(f"  g_steps: 2, d_steps: 1")
    
    if issues:
        print("\n[Issues Detected]")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n[Recommendations]")
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("\n[No major issues detected in metrics]")
    
    # Final summary
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS - MODE COLLAPSE")
    print("=" * 70)
    print("""
Mode collapse xảy ra do:

1. DISCRIMINATOR QUÁ MẠNH
   - d_loss giảm xuống 0.16-0.17 và ổn định
   - Gap (d_real - d_fake) tăng liên tục từ ~5 lên ~10+
   - Generator không thể "đánh lừa" discriminator

2. IMBALANCED TRAINING
   - lr_discriminator (5e-5) gấp 5x lr_generator (1e-5)
   - d_steps=1, g_steps=2 nhưng không đủ bù đắp

3. DIVERSITY PENALTY QUÁ THẤP
   - diversity_weight=0.2 không đủ mạnh
   - Generator tìm được 1-2 pattern "an toàn" (P, A) và bám vào

4. THIẾU ENTROPY REGULARIZATION
   - Không có loss term nào khuyến khích đa dạng amino acid
""")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED FIXES")
    print("=" * 70)
    print("""
1. REBALANCE TRAINING:
   - lr_discriminator: 5e-5 → 1e-5 (bằng generator)
   - g_steps: 2 → 3-4
   - label_smoothing: 0.1 → 0.2

2. TĂNG DIVERSITY:
   - diversity_weight: 0.2 → 0.5
   - Thêm entropy_weight: 0.3 (regularize output distribution)

3. WEAKEN DISCRIMINATOR:
   - Thêm dropout vào discriminator (0.3)
   - Thêm Gaussian noise vào input

4. ADJUST LOSS WEIGHTS:
   - Giảm adversarial_weight: 1.0 → 0.5
   - Tăng stability_weight cho structure_evaluator

5. TRAIN LONGER với early stopping based on diversity metric
""")

if __name__ == '__main__':
    analyze_metrics()
