"""Bounded, local HF tiny-model experiment for #58; see docs/dual-anchor-pilot.md."""

import argparse
import hashlib
import os
import resource
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq
import torch
from huggingface_hub import HfFileSystem

from astropt3.config_io import load_model_config
from astropt3.configuration_astropt3 import (
    DEFAULT_MODALITIES,
    LEGACY_SOUTH_MODALITIES,
    AstroPT3Config,
)
from astropt3.data import telemetry
from astropt3.data.nanotron_loader import (
    decode_legacy_image,
    decode_legacy_south_image,
    flatten_packed_batch,
)
from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.modeling_astropt3 import AstroPT3Model

# Sparse partitions, selected by file size, not by loss. Never enumerate the full catalogs.
SAMPLE = {
    'north': ('UniverseTBD/mmu_ssl_legacysurvey_north@f634744d3c44dd4fde0dee3172d4887c5e3c31c0', 945, 1),
    'south': ('hugging-science/mmu_legacysurvey_dr10_south_21@982f321784d8daf2a5f983eb3ec560f31b90d667', 2069, 0),
}


def prepare(out):
    # Probe only this process; the replay never needs the hub.
    fs = HfFileSystem()
    for side, (repo, pixel, group) in SAMPLE.items():
        name = repo.split('@')[0].split('/')[-1]
        path = f'datasets/{repo}/{name}/dataset/Norder=4/Dir=0/Npix={pixel}.parquet'
        cols = ['object_id', 'ra', 'dec', 'image.band', 'image.flux']
        start = perf_counter()
        with fs.open(path, cache_type='none', block_size=65536) as f:
            pf = pq.ParquetFile(f)
            rg = pf.metadata.row_group(group)
            assert rg.num_rows <= 40 and rg.total_byte_size < 32 * 2**20
            rows = pf.read_row_group(group, columns=cols, use_threads=False)
        pq.write_table(rows, out / f'{side}.parquet')
        print(side, path, group, rows.num_rows, 'fetch+stage_s', perf_counter() - start, flush=True)


def train(out, arm):
    torch.set_num_threads(2)
    torch.manual_seed(58)
    tiny, _ = load_model_config(Path(__file__).resolve().parents[1] / 'configs/model/test-tiny.yaml')
    arch = tiny.to_dict()
    north_bands = DEFAULT_MODALITIES[:3]
    arch.update(modalities=north_bands + LEGACY_SOUTH_MODALITIES, vocab_size=None)
    config = AstroPT3Config.from_dict(arch)
    model = AstroPT3Model(config)
    digest = hashlib.sha256()
    for tensor in model.state_dict().values():
        digest.update(tensor.numpy().tobytes())
    print('initial_weights_sha256', digest.hexdigest(), 'parameters', sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    sequencer = ObjectSequencer(config)
    pack = PackedCollator(config, seq_len=640)
    objects = {}
    began = perf_counter()
    for side in SAMPLE:
        rows = pq.read_table(out / f'{side}.parquet').slice(0, 10).to_pylist()
        north = side == 'north'
        modality = north_bands[0] if north else LEGACY_SOUTH_MODALITIES[0]
        source = modality['source']
        key, = modality['record_keys']
        decode = decode_legacy_image if north else decode_legacy_south_image
        objects[side] = [sequencer.build({'object_id': f"{source}:{r['object_id']}", key: decode(r['image'])}) for r in rows]
        assert len({o.object_id for o in objects[side]}) == 10
    print('read_decode_sequence_both_s', perf_counter() - began)

    @torch.no_grad()
    def score():
        model.eval()
        return {side: float(model(**pack(pool[8:])).loss) for side, pool in objects.items()}

    print('diagnostic_before', score())
    model.train()
    began = perf_counter()
    for step in range(8):
        if arm == 'mixed':
            draw = [objects[side][step] for side in SAMPLE]
        else:
            i = (2 * step) % 8
            draw = objects[arm][i:i+2]
        start = perf_counter()
        batch = pack(draw)
        wait = perf_counter() - start
        telemetry.observe_micro_batch(flatten_packed_batch(batch, config, pack.seq_len), wait)
        metrics = telemetry.drain_step()
        assert metrics['rows'] == 2 and metrics['tokens_nonpad'] == sum(map(len, draw))
        output = model(**batch)
        loss = output.loss
        assert loss is not None and torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.update(loss=float(loss.detach()), losses={m: float(v.detach()) for m,v in output.modality_losses.items()},
                       ids=[o.object_id for o in draw], lengths=list(map(len, draw)))
        telemetry.write_step(step + 1, metrics)
    print('train_wall_s', perf_counter() - began, 'diagnostic_after', score())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=['prepare', 'north', 'south', 'mixed'])
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    out = args.directory
    log = out / args.arm
    log.mkdir(parents=True, exist_ok=False)
    os.environ[telemetry.TELEMETRY_DIR_ENV] = str(log)
    if args.arm == 'prepare':
        if not telemetry.install_byte_probe():
            raise RuntimeError('HF byte probe unavailable')
        prepare(out)
    else:
        train(out, args.arm)
    print('peak_process_RSS_MiB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


if __name__ == '__main__':
    main()
