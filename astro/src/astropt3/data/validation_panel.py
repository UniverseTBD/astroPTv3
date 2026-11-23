"""A replayable held-out panel; only its ID manifest is written to disk."""

import json
from contextlib import closing
from itertools import islice
from pathlib import Path

import yaml

from .catalog_stream import CatalogStream
from .nanotron_loader import flatten_packed_batch, hf_config_from_modalities
from .packing import ObjectSequencer, PackedCollator

PANEL_FILENAME = "validation-panel.json"
PANEL_SEED = 76
_ROWS_PER_DRAW = 16


def load_validation_panel(model_config, objects, seq_len, batch_size, dp_size):
    """Load one fixed sample and pack all rows, including the last partial batch."""
    config = hf_config_from_modalities(
        model_config.modalities,
        spiral=model_config.spiral,
        image_norm_divisor=model_config.image_norm_divisor,
        spectra_norm_divisor=model_config.spectra_norm_divisor,
    )
    spec = yaml.safe_load(Path(__file__).with_name("validation_catalogs.yaml").read_text())
    stream = CatalogStream(spec, config)
    sequencer = ObjectSequencer(config)
    panel = {}
    # Bound the scan if the footprint is sparse or objects repeat across draws.
    with closing(stream.draws(seed=PANEL_SEED, rank=0, worker_id=0, rows_per_draw=_ROWS_PER_DRAW)) as draws:
        for _, records in islice(draws, objects):
            for record in records:
                object_id = record["object_id"]
                if object_id not in panel:
                    panel[object_id] = sequencer.build(record, epoch=PANEL_SEED)
                if len(panel) == objects:
                    break
            if len(panel) == objects:
                break
    if len(panel) != objects:
        raise ValueError(f"Validation panel found only {len(panel)} of {objects} unique objects")

    rows = PackedCollator(config, seq_len).pack_rows(list(panel.values()))
    shards = []
    for rank in range(dp_size):
        local_rows = rows[rank::dp_size]
        batches = []
        for start in range(0, len(local_rows), batch_size):
            batch_rows = local_rows[start:start + batch_size]
            length = max(sum(map(len, row)) for row in batch_rows)
            packed = PackedCollator(config, length).collate_rows(batch_rows)
            batches.append(flatten_packed_batch(packed, config, length))
        shards.append(batches)
    return list(panel), shards


def save_validation_manifest(path, object_ids, *, reference=None):
    """Fail on changed membership/order rather than comparing different panels."""
    manifest = {"seed": PANEL_SEED, "object_ids": object_ids}
    for existing in (reference, path if path.exists() else None):
        if existing is None:
            continue
        try:
            previous = json.loads(existing.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read validation panel manifest {existing}") from error
        if previous != manifest:
            raise ValueError(f"Validation panel does not match {existing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
