"""Sample a converted checkpoint using the same configured catalogs as training.

Template records come from one draw of the explicit recipe's catalog_stream.
Rows with more active sources come first; --rows selects positions in that
ordering. Each image payload counts once, regardless of filter count.
A split-aware recipe draws validation templates unless --split train is set.

Usage:
    uv run python scripts/generate.py --checkpoint <hf_dir> \
        --config-file <training.yaml> [--mode spectra-to-images] [--band hsc_g] [--n 4] \
        [--rows 0,1,2] [--stream-seed 0] [--seed 0] [--out generated] \
        [--wandb] [--wandb-run-id <id>]

Outputs are PNGs. Optional W&B logging either creates a generation run or
appends to --wandb-run-id; catalog configuration stays local.
"""

import argparse
import importlib
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import yaml

from astropt3.data.catalog_stream import CatalogStream
from astropt3.data.packing import has_modality


def _draw_live_records(config, catalog_stream: dict, rows: list[int], stream_seed: int) -> list[dict]:
    """Select templates from one upstream draw, closing its client afterward."""
    if not rows or any(index < 0 for index in rows):
        raise ValueError("--rows must contain nonnegative indices")
    stream = CatalogStream(catalog_stream, config)
    with closing(stream.draws(seed=stream_seed, rank=0, worker_id=0)) as draws:
        _, records = next(draws)
        records = list(records)
    modalities = config.modality_registry().modalities.values()

    def active_sources(record):
        return len({("image", m.record_keys[0]) if m.family == "image" else (m.family, m.name)
                    for m in modalities if has_modality(m, record)})

    records.sort(key=active_sources, reverse=True)
    if max(rows) >= len(records):
        raise ValueError(f"drew {len(records)} usable rows, but --rows asked for index {max(rows)}")
    return [records[index] for index in rows]


def main():
    from astropt3.eval.samples import BAND_MODES, MODES, sample_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="HF checkpoint dir")
    parser.add_argument("--config-file", required=True, type=Path, help="training YAML with explicit catalog_stream")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="default: unconditional with --band; otherwise all supported sampling modes",
    )
    parser.add_argument("--band", help="generate or condition on just this calibrated band, e.g. hsc_g")
    parser.add_argument("--n", type=int, default=4, help="samples to draw per mode")
    parser.add_argument("--temperature", type=float, default=1.0, help="scales GMM sigma")
    parser.add_argument("--seed", type=int, default=0, help="sampling RNG seed")
    parser.add_argument("--stream-seed", type=int, default=0, help="catalog draw seed")
    splits = ("train", "validation")
    parser.add_argument("--split", choices=splits, help="override the template draw split")
    parser.add_argument("--rows", default="0", help="comma-separated nonnegative template indices")
    parser.add_argument("--out", default="generated", help="output directory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true", help="log sample figures to W&B")
    parser.add_argument("--wandb-run-id", default=None, help="append to an existing W&B run")
    args = parser.parse_args()
    if args.band and args.mode is not None and args.mode not in BAND_MODES:
        parser.error(f"--band requires one of: {', '.join(BAND_MODES)}")

    from transformers import AutoConfig

    try:
        rows = [int(index) for index in args.rows.split(",")]
        recipe = yaml.safe_load(args.config_file.read_text())
        stages = recipe["data_stages"]
        if len(stages) != 1:
            raise ValueError("generation requires exactly one data stage")
        dataset = stages[0]["data"]["dataset"]
        if "crossmatch_desi" in dataset:
            raise ValueError("crossmatch_desi is removed; use an explicit catalog_stream")
        spec = dataset["catalog_stream"]
        if "split" in spec and spec["split"] not in splits:
            raise ValueError("split must be train or validation")
        if args.split is not None or "split" in spec:
            spec = {**spec, "split": args.split or "validation"}
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(f"Invalid rows or catalog configuration: {error}")

    config = AutoConfig.from_pretrained(args.checkpoint)
    if args.band:
        band = config.modality_registry().modalities.get(args.band)
        if band is None or band.band is None:
            parser.error(f"checkpoint has no image band {args.band!r}")
    records = _draw_live_records(config, spec, rows, args.stream_seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = sample_checkpoint(
        args.checkpoint,
        records,
        modes=[args.mode] if args.mode else None,
        band=args.band,
        n=args.n,
        temperature=args.temperature,
        seed=args.seed,
        out_dir=out_dir,
        device=args.device,
    )

    if args.wandb:
        wandb_module = cast(Any, importlib.import_module("wandb"))
        wandb_run = wandb_module.init(
            project="astropt3",
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            name=None if args.wandb_run_id else "generate",
            job_type="generation",
            config={k: v for k, v in vars(args).items() if k not in ("wandb", "wandb_run_id", "config_file")},
        )
        wandb_run.log({f"generation/{key}": wandb_module.Image(png) for key, png in pngs.items()})
        wandb_run.finish()

    for key, png in pngs.items():
        print(f"wrote {key}: {png}")


if __name__ == "__main__":
    main()
