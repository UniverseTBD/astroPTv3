"""Pack YAML-configured LSDB catalog streams into nanotron micro-batches.

Workers delegate source iteration to ``CatalogStream``. Batches are flattened
because nanotron only moves top-level tensors to the device.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Mapping
from typing import Any

import httpx
import numpy as np
import pandas as pd
import torch

from ..configuration_astropt3 import AstroPT3Config
from .band_registry import _DIV_FACTOR
from .echo import echo_draw, validate_echo_factor
from .packing import ObjectSequencer, PackedCollator
from .spectral import _DIV_FACTOR as _SPECTRA_DIV_FACTOR
from .telemetry import install_byte_probe, instrument

IMAGE_SHAPE = (3, 152, 152)
_MAX_NET_RETRIES = 60
_MAX_NET_RETRY_WAIT = 120


def hf_config_from_modalities(modalities, **extra) -> AstroPT3Config:
    """Build the HF-side config used by the shared sequencer and collator."""
    return AstroPT3Config(
        modalities=[dict(modality) for modality in modalities],
        **extra,
    )


def flatten_packed_batch(batch: dict, config: AstroPT3Config, seq_len: int) -> dict:
    """Convert ``PackedCollator`` output to nanotron's flat input contract."""
    registry = config.modality_registry()
    batch_size = batch["input_ids"].shape[0]
    flat = {
        "input_ids": batch["input_ids"],
        "position_ids": batch["position_ids"],
    }
    for name in registry.names():
        modality = registry.get_config(name)
        if name in batch["modality_masks"]:
            flat[f"{name}_mask"] = batch["modality_masks"][name]
            flat[f"{name}_values"] = batch["modality_values"][name]
            flat[f"{name}_positions"] = batch["modality_positions"][name]
        else:
            flat[f"{name}_mask"] = torch.zeros(
                (batch_size, seq_len), dtype=torch.bool
            )
            flat[f"{name}_values"] = torch.empty(
                (0, modality.input_size), dtype=torch.float32
            )
            if modality.pos_type == "index":
                flat[f"{name}_positions"] = torch.empty((0,), dtype=torch.long)
            else:
                flat[f"{name}_positions"] = torch.empty(
                    (0, modality.pos_input_size), dtype=torch.float32
                )
    return flat


def regroup_micro_batch(flat: dict, names) -> dict:
    """Convert a flat nanotron micro-batch to HF model keyword arguments."""
    return {
        "input_ids": flat["input_ids"],
        "position_ids": flat["position_ids"],
        "modality_values": {
            name: flat[f"{name}_values"]
            for name in names
            if flat[f"{name}_values"].shape[0]
        },
        "modality_masks": {
            name: flat[f"{name}_mask"]
            for name in names
            if flat[f"{name}_mask"].any()
        },
        "modality_positions": {
            name: flat[f"{name}_positions"]
            for name in names
            if flat[f"{name}_values"].shape[0]
        },
    }


def _as_mapping(value: Any) -> Mapping:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
        if isinstance(value, Mapping):
            return value
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict(orient="list")
        except TypeError:
            value = value.to_dict()
        if isinstance(value, Mapping):
            return value
    raise ValueError(f"image payload has unsupported type {type(value).__name__}")


def _stack_nested(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        return np.stack([_stack_nested(item) for item in array])
    return array


def _finite_scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _row_from_map_rows(mapped: Mapping, nested: dict[str, tuple[str, ...]]) -> dict:
    """Reassemble dotted-key arrays without per-record pandas DataFrames.

    ``map_rows`` exposes nested Arrow sub-columns as NumPy arrays; absent
    nested values arrive as empty arrays and become ``None`` here.
    """
    row = dict(mapped)
    for name, subfields in nested.items():
        sub: dict = {}
        present = False
        for field in subfields:
            key = f"{name}.{field}"
            if key not in row:
                continue
            value = row.pop(key)
            sub[field] = value
            if isinstance(value, np.ndarray):
                present = present or bool(value.size)
            else:
                present = present or value is not None
        row[name] = sub if present else None
    return row


def decode_legacy_row(row: Mapping) -> dict:
    """Decode a LegacySurvey payload, not a separate source-selection mode."""
    image = _as_mapping(row.get("image"))
    bands = [str(band) for band in image.get("band", ())]
    flux = _stack_nested(image.get("flux")).astype(np.float32, copy=False)
    if flux.shape != IMAGE_SHAPE:
        raise ValueError(f"image flux has shape {flux.shape}, expected {IMAGE_SHAPE}")
    if len(bands) != IMAGE_SHAPE[0]:
        raise ValueError(f"image has {len(bands)} bands, expected {IMAGE_SHAPE[0]}")

    object_id = row.get("object_id")
    if object_id is None:
        raise ValueError("LegacySurvey row has no object_id")
    record = {
        "object_id": str(object_id),
        "image": {"flux": flux, "band": bands},
    }
    for key, value in row.items():
        if key in {"object_id", "image"}:
            continue
        number = _finite_scalar(value)
        if number is not None:
            record[str(key)] = number

    fwhm = image.get("psf_fwhm")
    if fwhm is not None and len(fwhm) == len(bands):
        for band, value in zip(bands, fwhm, strict=True):
            number = _finite_scalar(value)
            if number is not None and 0 < number < 5:
                record[f"psf_fwhm_{band}"] = number
    return record


def _decode_spectrum(value: Any) -> dict:
    if isinstance(value, pd.DataFrame):
        # Column access avoids boxing every cell of a 7781-bin spectrum.
        return {
            "flux": value["flux"].to_numpy(dtype=np.float32, copy=False),
            "lambda": value["lambda"].to_numpy(dtype=np.float32, copy=False),
            "mask": value["mask"].to_numpy(dtype=bool, copy=False),
        }
    spec = _as_mapping(value)
    return {
        "flux": _stack_nested(spec.get("flux")).astype(np.float32, copy=False),
        "lambda": _stack_nested(spec.get("lambda")).astype(np.float32, copy=False),
        "mask": _stack_nested(spec.get("mask")).astype(bool, copy=False),
    }


def consumer_seed(seed: int, rank: int, worker: int, retry_generation: int) -> int:
    """Stable independent seed for one DP-rank/worker/fresh-stream consumer."""
    return zlib.crc32(f"{seed}:{rank}:{worker}:{retry_generation}".encode())


_EXCEPTION_AS_DATA_SUBSCRIPT = re.compile(
    r"'(\w*(Error|Exception))' object is not subscriptable"
)


def _retryable(error: Exception) -> bool:
    if isinstance(error, (httpx.HTTPError, OSError, TimeoutError)):
        return True
    if isinstance(error, RuntimeError) and "client has been closed" in str(error):
        return True
    # fsspec can swallow a transport exception into parquet range bytes.
    if isinstance(error, TypeError):
        message = str(error)
        if "can't concat" in message and "to bytes" in message:
            return True
        return bool(_EXCEPTION_AS_DATA_SUBSCRIPT.search(message))
    return False


class PackedMicroBatches(torch.utils.data.IterableDataset):
    """Endless configured LSDB records packed into nanotron micro-batches."""

    def __init__(
        self,
        config: AstroPT3Config,
        micro_batch_size: int,
        seq_len: int,
        *,
        catalog_stream: dict,
        rank: int = 0,
        seed: int = 0,
        echo_factor: int = 1,
    ):
        from .catalog_stream import CatalogStream

        super().__init__()
        validate_echo_factor(echo_factor)
        self.echo_factor = echo_factor
        self.config = config
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.seed = seed
        self.catalog_stream = CatalogStream(catalog_stream, config)
        self._epoch = 0  # InfiniteStream partition-draw nonce for span ordering
        self.sequencer = ObjectSequencer(config)
        self.collator = PackedCollator(config, seq_len=seq_len)

    def _open_records(self, worker):
        for records in self._open_draws(worker):
            yield from records

    def _open_draws(self, worker):
        """Yield finite record iterators, one per upstream catalog draw."""
        for epoch, records in self.catalog_stream.draws(
            seed=self.seed, rank=self.rank, worker_id=worker.id if worker else 0,
        ):
            self._epoch = epoch
            yield records

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        install_byte_probe(self.rank, worker.id if worker else None)
        if self.echo_factor == 1:
            yield from self._pack_records(self._open_records(worker))
            return
        for records in self._open_draws(worker):
            batches = list(self._pack_records(records, flush_tail=True))
            yield from echo_draw(
                batches, self.config, echo_factor=self.echo_factor,
                seed=consumer_seed(self.seed, self.rank, worker_id, self._epoch),
            )
            del batches

    def _pack_records(self, records, *, flush_tail=False):
        rows: list[list] = [[] for _ in range(self.micro_batch_size)]
        used = [0] * self.micro_batch_size

        for record in records:
            obj = self.sequencer.build(record, epoch=self._epoch)
            if len(obj) > self.seq_len:
                raise ValueError(
                    f"object of length {len(obj)} exceeds seq_len {self.seq_len}"
                )
            row_index = min(range(len(used)), key=lambda row: (used[row], row))
            if used[row_index] + len(obj) > self.seq_len:
                batch = self.collator.collate_rows(rows)
                expected_shape = (self.micro_batch_size, self.seq_len)
                if batch["input_ids"].shape != expected_shape:
                    raise AssertionError(
                        f"packing mismatch: {batch['input_ids'].shape} != {expected_shape}"
                    )
                yield flatten_packed_batch(batch, self.config, self.seq_len)
                rows = [[] for _ in range(self.micro_batch_size)]
                used = [0] * self.micro_batch_size
                row_index = 0
            rows[row_index].append(obj)
            used[row_index] += len(obj)
        if flush_tail and any(used):
            yield flatten_packed_batch(self.collator.collate_rows(rows), self.config, self.seq_len)


def build_astropt3_dataloader(
    dataset_args,
    model_config,
    micro_batch_size: int,
    sequence_length: int,
    dp_rank: int,
    dp_size: int,
    num_workers: int = 0,
    seed: int = 0,
    multiprocessing_context: str | None = None,
    batch_accumulation_per_replica: int = 1,
    world_size: int = 1,
) -> Any:
    """Build worker-side draw echoing, then main-process telemetry/device inputs."""
    echo_factor = getattr(dataset_args, "echo_factor", 1)
    validate_echo_factor(echo_factor)
    if echo_factor > 1:
        if batch_accumulation_per_replica != 1:
            raise ValueError("echo-factor > 1 requires batch_accumulation_per_replica=1")
        if world_size != 1 or dp_size != 1:
            raise ValueError("echo-factor > 1 currently requires world_size=dp_size=1")
    config = hf_config_from_modalities(
        model_config.modalities,
        **{
            field: getattr(model_config, field, default)
            for field, default in [
                ("jetformer_flow_steps", 4),
                ("jetformer_flow_hidden", 128),
                ("jetformer_gmm_k", 4),
                ("jetformer_noise_max", 0.1),
                ("jetformer_noise_min", 0.0),
                ("scalar_gmm_k", 5),
                ("image_norm_divisor", _DIV_FACTOR),
                ("spectra_norm_divisor", _SPECTRA_DIV_FACTOR),
                ("spiral", True),
            ]
        },
    )
    catalog_stream = getattr(dataset_args, "catalog_stream", None)
    if not isinstance(catalog_stream, dict):
        raise ValueError("catalog_stream is required; migrate legacy dataset YAML")
    dataset = PackedMicroBatches(
        config,
        micro_batch_size,
        sequence_length,
        rank=dp_rank,
        seed=seed,
        echo_factor=echo_factor,
        catalog_stream=catalog_stream,
    )
    if num_workers and multiprocessing_context is None:
        multiprocessing_context = "spawn"
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context=multiprocessing_context,
    )
    return instrument(loader, echo_factor=echo_factor)
