#!/usr/bin/env python
"""Print (and assert) parameter counts for every model-size YAML.

Models are instantiated on the meta device so even 12B costs no memory.
Exits non-zero if any named size is more than 10% off its nominal count.
JetFormer flows are excluded from that comparison: they are a separately sized
tokeniser (docs/jetformer-flow.md), reported in their own column.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astropt3 import AstroPT3Model
from astropt3.config_io import load_model_config


def count_params(config) -> tuple[int, int, int]:
    with torch.device("meta"):
        model = AstroPT3Model(config)
    total = sum(p.numel() for p in model.parameters())
    body = sum(p.numel() for p in model.model.parameters())
    flows = sum(p.numel() for p in model.flows.parameters())
    return total - flows, body, flows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        default=Path(__file__).resolve().parents[1] / "configs" / "model",
        type=Path,
    )
    parser.add_argument("--tolerance", default=0.10, type=float)
    args = parser.parse_args()

    failures = []
    print(f"{'config':<16} {'total':>15} {'body':>15} {'extras':>12} {'flows':>12} {'nominal':>15} {'off':>7}")
    for path in sorted(args.config_dir.glob("*.yaml")):
        config, meta = load_model_config(path)
        total, body, flows = count_params(config)
        nominal = meta.get("nominal_params")
        if nominal:
            off = (total - nominal) / nominal
            offs = f"{off:+.1%}"
            if abs(off) > args.tolerance:
                failures.append((meta.get("name", path.stem), offs))
        else:
            offs = "-"
        print(
            f"{meta.get('name', path.stem):<16} {total:>15,} {body:>15,} "
            f"{total - body:>12,} {flows:>12,} {nominal or 0:>15,} {offs:>7}"
        )

    if failures:
        print(f"\nFAIL: sizes off by more than {args.tolerance:.0%}: {failures}")
        return 1
    print("\nAll named sizes within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
