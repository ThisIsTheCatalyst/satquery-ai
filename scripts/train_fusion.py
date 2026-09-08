#!/usr/bin/env python3
"""
Train the optical+SAR fusion head. Encoders frozen — head only.

Time budget: this is sized to finish in roughly 10-20 minutes on a free
Colab T4, well inside a 2-hour cap. Three choices make that possible:

  1. Encoders are FROZEN ImageNet ResNet-18s. Only a 2-layer MLP head is
     trained (~200k params instead of ~22M).
  2. Features are PRECOMPUTED ONCE and cached. Each encoder runs over the
     dataset a single time, then all 15 epochs train on cached feature
     vectors. This is the big win — epochs become seconds, not minutes.
  3. The subset is capped at 500 samples at 64x64.

Also runs the 3-way ablation (optical-only / SAR-only / fused), which is
the mandated comparison and the strongest single result slide.

Usage:
    python scripts/train_fusion.py --config configs/demo.yaml
    python scripts/train_fusion.py --synthetic     # smoke test, no data
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def build_encoders(device):
    """Two frozen ImageNet ResNet-18 trunks (fc removed -> 512-d features)."""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights

    def trunk(in_ch):
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if in_ch != 3:
            # SAR has 2 channels (VV/VH); adapt conv1 by averaging RGB weights.
            old = m.conv1
            new = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True)
                                 .repeat(1, in_ch, 1, 1))
            m.conv1 = new
        m.fc = nn.Identity()
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m.to(device)

    return trunk(3), trunk(2)


class FusionHead:
    """2-layer MLP over concatenated features. Kept tiny on purpose."""

    @staticmethod
    def build(in_dim, n_classes, device):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        ).to(device)


def precompute_features(enc_opt, enc_sar, optical, sar, device, batch=64):
    """Run each frozen encoder over the whole dataset exactly once."""
    import torch
    feats_o, feats_s = [], []
    with torch.no_grad():
        for i in range(0, len(optical), batch):
            o = torch.from_numpy(optical[i:i + batch]).float().to(device)
            s = torch.from_numpy(sar[i:i + batch]).float().to(device)
            feats_o.append(enc_opt(o).cpu().numpy())
            feats_s.append(enc_sar(s).cpu().numpy())
    return np.concatenate(feats_o), np.concatenate(feats_s)


def train_head(Xtr, ytr, Xva, yva, n_classes, device, epochs, lr, wd, tag):
    """Train one MLP head on cached features. Returns (head, best_macro_f1)."""
    import torch
    import torch.nn as nn

    head = FusionHead.build(Xtr.shape[1], n_classes, device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    Xva_t = torch.from_numpy(Xva).float().to(device)

    best_f1, best_state = 0.0, None
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        tot = 0.0
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = lossf(head(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss)

        head.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(head(Xva_t)).cpu().numpy() > 0.5).astype(int)
        f1 = macro_f1(pred, yva)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        print(f"  [{tag}] epoch {ep+1:>2}/{epochs}  loss={tot:.3f}  macroF1={f1:.3f}")

    if best_state:
        head.load_state_dict(best_state)
    return head, best_f1


def macro_f1(pred, true):
    f1s = []
    for c in range(true.shape[1]):
        tp = int(np.sum((pred[:, c] == 1) & (true[:, c] == 1)))
        fp = int(np.sum((pred[:, c] == 1) & (true[:, c] == 0)))
        fn = int(np.sum((pred[:, c] == 0) & (true[:, c] == 1)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s))


def load_data(cfg, synthetic, size=64):
    """Load the fusion subset, or generate a synthetic stand-in.

    Synthetic mode exists so the training path can be smoke-tested before the
    real optical+SAR subset is in place. It is clearly labelled in the output
    and its numbers must never be reported as results.
    """
    root = Path(cfg.get("datasets", {}).get("fusion", {}).get("root",
                                                              "data/demo/fusion"))
    n_max = int(cfg.get("training", {}).get("max_train_samples", 500))

    if not synthetic and (root / "manifest.json").exists():
        man = json.loads((root / "manifest.json").read_text())
        from PIL import Image as PILImage
        opt, sar, lab = [], [], []
        classes = man["classes"]
        for rec in man["samples"][:n_max]:
            o = np.array(PILImage.open(root / rec["optical"]).convert("RGB")
                         .resize((size, size)))
            s = np.array(PILImage.open(root / rec["sar"]).convert("L")
                         .resize((size, size)))
            opt.append(o.transpose(2, 0, 1) / 255.0)
            sar.append(np.stack([s, s]) / 255.0)
            y = np.zeros(len(classes), dtype=np.float32)
            for c in rec["labels"]:
                if c in classes:
                    y[classes.index(c)] = 1.0
            lab.append(y)
        return (np.array(opt, dtype=np.float32),
                np.array(sar, dtype=np.float32),
                np.array(lab, dtype=np.float32), classes, False)

    print("  [synthetic data — smoke test only, DO NOT report these numbers]")
    rng = np.random.default_rng(42)
    classes = ["water", "vegetation", "built_up", "bare_soil"]
    n = min(n_max, 400)
    y = (rng.random((n, len(classes))) > 0.65).astype(np.float32)
    y[y.sum(1) == 0, 0] = 1.0
    opt = rng.random((n, 3, size, size)).astype(np.float32) * 0.4
    sar = rng.random((n, 2, size, size)).astype(np.float32) * 0.4
    for i in range(n):                      # make labels weakly recoverable
        for c in range(len(classes)):
            if y[i, c]:
                opt[i, c % 3] += 0.35
                sar[i, c % 2] += 0.25
    return np.clip(opt, 0, 1), np.clip(sar, 0, 1), y, classes, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    import yaml
    import torch

    cfg = yaml.safe_load(open(args.config))
    tcfg = cfg.get("training", {})
    epochs = args.epochs or int(tcfg.get("epochs", 15))
    lr = float(tcfg.get("lr", 1e-3))
    wd = float(tcfg.get("weight_decay", 1e-4))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    t_start = time.time()

    optical, sar, labels, classes, is_syn = load_data(cfg, args.synthetic)
    print(f"Samples: {len(optical)}  classes: {len(classes)}")

    split = int(0.8 * len(optical))
    idx = np.random.default_rng(42).permutation(len(optical))
    tr, va = idx[:split], idx[split:]

    print("\nPrecomputing frozen-encoder features (one pass)...")
    enc_o, enc_s = build_encoders(device)
    t0 = time.time()
    Fo, Fs = precompute_features(enc_o, enc_s, optical, sar, device)
    print(f"  done in {time.time()-t0:.1f}s  optical={Fo.shape} sar={Fs.shape}")

    variants = {
        "optical_only": (Fo, "optical"),
        "sar_only": (Fs, "SAR"),
        "fused": (np.concatenate([Fo, Fs], axis=1), "fused"),
    }

    print("\n3-way ablation")
    print("-" * 46)
    results, heads = {}, {}
    for tag, (F, _) in variants.items():
        head, f1 = train_head(F[tr], labels[tr], F[va], labels[va],
                              len(classes), device, epochs, lr, wd, tag)
        results[tag] = f1
        heads[tag] = head

    print("\n" + "=" * 46)
    print("Ablation results (macro-F1, validation)")
    print("=" * 46)
    for tag in ("optical_only", "sar_only", "fused"):
        print(f"  {tag:<16} {results[tag]:.3f}")
    best_single = max(results['optical_only'], results['sar_only'])
    delta = results["fused"] - best_single
    print(f"\n  fusion gain over best single modality: {delta:+.3f}")
    if delta <= 0:
        print("  NOTE: fusion did not beat the best single modality here.")
        print("  Report this honestly — a negative ablation stated plainly is")
        print("  more credible than a positive one that cannot be reproduced.")
    print("=" * 46)

    ck = Path(cfg.get("models", {}).get("fusion", {}).get(
        "checkpoint", "models/checkpoints/fusion/fusion_head.pt"))
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": heads["fused"].state_dict(),
                "classes": classes,
                "in_dim": variants["fused"][0].shape[1],
                "synthetic": is_syn}, ck)
    print(f"\nCheckpoint: {ck}")

    out = Path("outputs/eval_results/fusion_ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "synthetic_data": is_syn,
        "n_samples": int(len(optical)),
        "classes": classes,
        "epochs": epochs,
        "macro_f1": results,
        "fusion_gain_over_best_single": delta,
        "wall_clock_seconds": time.time() - t_start,
    }, indent=2))
    print(f"Results:    {out}")
    print(f"Total time: {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
