"""Transport hygiene and bounded concurrency for Hugging Face Hub fsspec reads."""

from __future__ import annotations

import io
import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import Lock
from typing import cast

import fsspec.parquet
import nested_pandas as npd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fsspec.utils import merge_offset_ranges
from huggingface_hub import HfFileSystem
from huggingface_hub.hf_file_system import HfFileSystemFile

# Split single large range GETs (> 32 MiB) into bounded concurrent sub-ranges to avoid
# ~95 MB/s single-TCP-stream stalls on 200-250 MiB Parquet row groups (ADR 0014 §10a).
RANGE_CHUNK_BYTES = 32 * 1024 * 1024
MAX_CAT_RANGE_WORKERS = 4
MAX_CHUNK_WORKERS = 8

_FOOTER_SAMPLE_BYTES = 256_000
_PREFETCH_MAX_GAP = 64_000
_PREFETCH_MAX_BLOCK = 32_000_000

_pool_lock = Lock()
_pool_pid: int | None = None
_cat_pool: ThreadPoolExecutor | None = None
_chunk_pool: ThreadPoolExecutor | None = None


def _pools() -> tuple[ThreadPoolExecutor, ThreadPoolExecutor]:
    """Return distinct bounded pools for ``cat_ranges`` and sub-range splitting.

    Keeping the two pools separate prevents nested submission deadlocks when
    ``cat_ranges`` invokes ``cat_file`` -> ``_fetch_range`` on a >32 MiB range.
    """
    global _pool_pid, _cat_pool, _chunk_pool
    pid = os.getpid()
    if _pool_pid == pid and _cat_pool is not None and _chunk_pool is not None:
        return _cat_pool, _chunk_pool
    with _pool_lock:
        if _pool_pid != pid or _cat_pool is None or _chunk_pool is None:
            _cat_pool = ThreadPoolExecutor(
                max_workers=MAX_CAT_RANGE_WORKERS, thread_name_prefix="hf-cat-range"
            )
            _chunk_pool = ThreadPoolExecutor(
                max_workers=MAX_CHUNK_WORKERS, thread_name_prefix="hf-range-chunk"
            )
            _pool_pid = pid
        return _cat_pool, _chunk_pool


def disable_hf_cat_readahead() -> None:
    """Avoid fsspec's 5 MiB ReadAheadCache over-fetch on bounded ``cat_file`` reads.

    ``HfFileSystem`` inherits ``AbstractFileSystem.cat_file()``, which opens a temporary
    handle with the default ``cache_type="readahead"`` and 5 MiB block size. Every
    Parquet column range fetched through ``fsspec.parquet`` therefore over-fetches 5 MiB
    past ``end`` and immediately discards it when the handle closes. Passing
    ``cache_type="none"`` keeps range reads exact while still routing through
    ``HfFileSystemFile._fetch_range`` (and its byte telemetry probe).
    """
    current = HfFileSystem.cat_file
    if getattr(current, "_astropt3_no_readahead", False):
        return

    @wraps(current)
    def cat_file(self, path, start=None, end=None, **kwargs):
        if start is not None or end is not None:
            kwargs.setdefault("cache_type", "none")
        return current(self, path, start=start, end=end, **kwargs)

    cat_file._astropt3_no_readahead = True  # type: ignore[attr-defined]
    HfFileSystem.cat_file = cat_file


def enable_hf_concurrent_ranges(
    *, chunk_bytes: int = RANGE_CHUNK_BYTES, max_range_workers: int = MAX_CAT_RANGE_WORKERS
) -> None:
    """Parallelize ``HfFileSystem.cat_ranges`` and split large ``_fetch_range`` GETs.

    Works for both ``fsspec.parquet`` (which calls ``cat_ranges``) and direct
    ``pyarrow.parquet.ParquetFile`` reads (which call ``HfFileSystemFile._fetch_range``).
    Preserves exact byte accounting when combined with ``telemetry.install_byte_probe``.
    """
    disable_hf_cat_readahead()

    current_cat_ranges = HfFileSystem.cat_ranges
    if not getattr(current_cat_ranges, "_astropt3_concurrent_ranges", False):
        @wraps(current_cat_ranges)
        def cat_ranges(self, paths, starts, ends, max_gap=None, on_error="return", **kwargs):
            if max_gap is not None:
                raise NotImplementedError
            if not isinstance(paths, list):
                raise TypeError
            if not isinstance(starts, list):
                starts = [starts] * len(paths)
            if not isinstance(ends, list):
                ends = [ends] * len(paths)
            if len(starts) != len(paths) or len(ends) != len(paths):
                raise ValueError
            if len(paths) <= 1 or max_range_workers <= 1:
                return current_cat_ranges(
                    self, paths, starts, ends, max_gap=max_gap, on_error=on_error, **kwargs
                )

            def fetch_one(item):
                p, s, e = item
                try:
                    return self.cat_file(p, s, e, **kwargs)
                except Exception as exc:
                    if on_error == "return":
                        return exc
                    raise

            cat_pool, _ = _pools()
            return list(cat_pool.map(fetch_one, zip(paths, starts, ends)))

        cat_ranges._astropt3_concurrent_ranges = True  # type: ignore[attr-defined]
        HfFileSystem.cat_ranges = cat_ranges

    current_fetch = HfFileSystemFile._fetch_range
    if not getattr(current_fetch, "_astropt3_concurrent_fetch", False):
        def _fetch_range(self, start: int, end: int) -> bytes:
            if chunk_bytes <= 0 or (end - start) <= chunk_bytes:
                return current_fetch(self, start, end)
            spans = [
                (offset, min(offset + chunk_bytes, end))
                for offset in range(start, end, chunk_bytes)
            ]
            _, chunk_pool = _pools()
            parts = list(chunk_pool.map(lambda span: current_fetch(self, span[0], span[1]), spans))
            return b"".join(parts)

        _fetch_range.__name__ = current_fetch.__name__
        _fetch_range.__doc__ = current_fetch.__doc__
        _fetch_range._astropt3_concurrent_fetch = True  # type: ignore[attr-defined]
        # Intentionally do NOT set __wrapped__ so telemetry.install_byte_probe does not unwrap it.
        HfFileSystemFile._fetch_range = _fetch_range


class _CoalescedReadUnsupported(Exception):
    """Raised when a Parquet read should fall back to HATS / nested-pandas."""


def _normalize_dnf_filters(
    filters: list[tuple[str, str, object]] | list[list[tuple[str, str, object]]],
) -> list[list[tuple[str, str, object]]]:
    """Normalize pyarrow list-of-tuples filters to DNF (list of conjunctions)."""
    if not isinstance(filters, list):
        raise _CoalescedReadUnsupported("non-list filters")
    if not filters:
        return []
    first = filters[0]
    if isinstance(first, tuple):
        conjunctions = [cast(list[tuple[str, str, object]], filters)]
    elif isinstance(first, list):
        conjunctions = cast(list[list[tuple[str, str, object]]], filters)
    else:
        raise _CoalescedReadUnsupported("invalid filters structure")
    for conj in conjunctions:
        if not isinstance(conj, list):
            raise _CoalescedReadUnsupported("invalid conjunction in filters")
        for term in conj:
            if not isinstance(term, tuple) or len(term) != 3:
                raise _CoalescedReadUnsupported("invalid predicate tuple in filters")
    return conjunctions


def _predicate_matches_row_group(
    rg_meta: pq.RowGroupMetaData, col_indices: dict[str, int], col: str, op: str, val: object
) -> bool:
    """Conservatively check whether any row in ``rg_meta`` can satisfy ``(col, op, val)``."""
    col_idx = col_indices.get(col)
    if col_idx is None:
        return True
    stats = rg_meta.column(col_idx).statistics
    if stats is None or val is None or (isinstance(val, float) and math.isnan(val)):
        return True
    if not stats.has_min_max:
        return True
    if stats.has_null_count and stats.null_count == rg_meta.num_rows:
        return False
    smin, smax = stats.min, stats.max
    try:
        if op in ("==", "="):
            return bool(smin <= val <= smax)
        if op == "!=":
            return not (smin == val and smax == val)
        if op == "<":
            return bool(smin < val)
        if op == "<=":
            return bool(smin <= val)
        if op == ">":
            return bool(smax > val)
        if op == ">=":
            return bool(smax >= val)
        if op == "in":
            if not isinstance(val, (list, tuple, set, frozenset)):
                return True
            return any(
                v is not None and not (isinstance(v, float) and math.isnan(v)) and smin <= v <= smax
                for v in val
            )
    except TypeError:
        return True
    return True


def _select_row_groups(
    metadata: pq.FileMetaData, dnf: list[list[tuple[str, str, object]]]
) -> list[int]:
    """Return row-group indices that may satisfy ``dnf`` based on Parquet min/max stats."""
    if not dnf or metadata.num_row_groups == 0:
        return list(range(metadata.num_row_groups))
    rg0 = metadata.row_group(0)
    col_indices = {rg0.column(i).path_in_schema: i for i in range(rg0.num_columns)}
    selected: list[int] = []
    for rg_idx in range(metadata.num_row_groups):
        rg = metadata.row_group(rg_idx)
        if any(
            all(_predicate_matches_row_group(rg, col_indices, col, op, val) for col, op, val in conj)
            for conj in dnf
        ):
            selected.append(rg_idx)
    return selected


def _is_list_arrow_type(dtype: pa.DataType) -> bool:
    return (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
    )


def _read_coalesced_parquet_to_nestedframe(
    filesystem,
    path: str,
    *,
    columns: list[str],
    filters=None,
    schema: pa.Schema | None = None,
    use_pandas_metadata: bool = True,
) -> npd.NestedFrame:
    """Read a remote Parquet file via coalesced leaf/row-group ``cat_ranges`` into a NestedFrame."""
    if isinstance(filters, pc.Expression):
        raise _CoalescedReadUnsupported("PyArrow Expression filters")
    dnf = _normalize_dnf_filters(filters) if filters is not None else None
    filter_expr = pq.filters_to_expression(filters) if dnf else None
    filter_columns = list(dict.fromkeys(col for conj in dnf for col, _, _ in conj)) if dnf else []
    read_columns = list(dict.fromkeys([*columns, *filter_columns]))

    file_size = int(filesystem.size(path))
    if file_size < 12:
        raise _CoalescedReadUnsupported("parquet file too small")
    footer_start = max(0, file_size - _FOOTER_SAMPLE_BYTES)
    footer = filesystem.cat_file(path, start=footer_start, end=file_size)
    footer_len = int.from_bytes(footer[-8:-4], "little")
    real_footer_start = file_size - (footer_len + 8)
    if real_footer_start < 0:
        raise _CoalescedReadUnsupported("invalid parquet footer length")
    if real_footer_start < footer_start:
        prefix = filesystem.cat_file(path, start=real_footer_start, end=footer_start)
        footer = prefix + footer
        footer_start = real_footer_start

    meta_pf = pq.ParquetFile(io.BytesIO(footer))
    file_schema = meta_pf.schema_arrow
    if schema is not None and not schema.equals(file_schema, check_metadata=False):
        raise _CoalescedReadUnsupported("schema differs from file schema")

    # Validate requested columns and group dotted struct leaf requests by parent struct.
    struct_leaves: dict[str, list[str]] = {}
    for col in columns:
        if col in file_schema.names:
            continue
        name, _, field = col.partition(".")
        if (
            "." in field
            or name not in file_schema.names
            or not pa.types.is_struct(struct_type := file_schema.field(name).type)
            or struct_type.get_field_index(field) < 0
        ):
            raise _CoalescedReadUnsupported(f"{col!r} is not a direct field of a struct column")
        struct_leaves.setdefault(name, []).append(field)
    for col in filter_columns:
        if col not in file_schema.names:
            raise _CoalescedReadUnsupported(f"filter column {col!r} is not top-level")
    # Reject mixed full + partial requests for the same struct column (let nested-pandas raise).
    if any(name in columns for name in struct_leaves):
        raise _CoalescedReadUnsupported("mixed full and partial struct column request")

    md = meta_pf.metadata
    row_groups = _select_row_groups(md, dnf) if dnf else list(range(md.num_row_groups))
    blocks: dict[tuple[int, int], bytes] = {(footer_start, file_size): footer}

    if row_groups:
        starts: list[int] = []
        ends: list[int] = []
        for rg_idx in row_groups:
            rg = md.row_group(rg_idx)
            for c in range(rg.num_columns):
                col_meta = rg.column(c)
                leaf_path = col_meta.path_in_schema
                if any(leaf_path == req or leaf_path.startswith(req + ".") for req in read_columns):
                    d = col_meta.to_dict()
                    offset0 = min(
                        v
                        for v in (
                            d.get("dictionary_page_offset"),
                            d.get("data_page_offset"),
                            d.get("index_page_offset"),
                        )
                        if v is not None
                    )
                    if offset0 < footer_start:
                        starts.append(offset0)
                        ends.append(min(offset0 + d["total_compressed_size"], footer_start))
        if starts:
            m_paths, m_starts, m_ends = merge_offset_ranges(
                [path] * len(starts),
                starts,
                ends,
                max_gap=_PREFETCH_MAX_GAP,
                max_block=_PREFETCH_MAX_BLOCK,
                sort=True,
            )
            buffers = filesystem.cat_ranges(m_paths, m_starts, m_ends, on_error="raise")
            for s, e, data in zip(m_starts, m_ends, buffers, strict=True):
                blocks[(s, e)] = data

    if not any(s == 0 and e >= 4 for s, e in blocks):
        blocks[(0, 4)] = b"PAR1"

    buffered = fsspec.parquet.AlreadyBufferedFile(
        fs=None, path=path, mode="rb", cache_type="parts", cache_options={"data": blocks}, size=file_size
    )
    table = pq.ParquetFile(buffered).read_row_groups(row_groups, columns=read_columns)
    if filter_expr is not None and table.num_rows > 0:
        table = table.filter(filter_expr)

    # Assemble projected output table matching nested-pandas's read_parquet output shape:
    # - top-level columns (and non-list struct leaves) come first in requested order
    # - struct-of-list sub-columns are appended at the end as pruned structs (NestedDtype)
    arrays: list[pa.ChunkedArray] = []
    names: list[str] = []
    nested_arrays: list[pa.ChunkedArray] = []
    nested_names: list[str] = []
    emitted_structs: set[str] = set()
    for col in columns:
        if col in file_schema.names:
            arrays.append(table[col])
            names.append(col)
            continue
        name, field = col.split(".")
        fields = struct_leaves[name]
        struct_col = table[name]
        struct_type = struct_col.type
        if all(_is_list_arrow_type(struct_type.field(f).type) for f in fields):
            if name in emitted_structs:
                continue
            emitted_structs.add(name)
            out_type = pa.struct([struct_type.field(f) for f in fields])
            repacked_chunks = [
                pa.StructArray.from_arrays(
                    [pc.struct_field(chunk, f) for f in fields],
                    names=fields,
                )
                for chunk in struct_col.chunks
            ]
            nested_arrays.append(pa.chunked_array(repacked_chunks, type=out_type))
            nested_names.append(name)
        else:
            arrays.append(pc.struct_field(struct_col, field))
            names.append(field)

    metadata = schema.metadata if schema is not None else file_schema.metadata
    out_table = pa.Table.from_arrays(
        [*arrays, *nested_arrays], names=[*names, *nested_names], metadata=metadata
    )
    return npd.from_pyarrow(out_table, use_pandas_metadata=use_pandas_metadata)


def enable_hf_coalesced_parquet_reads() -> None:
    """Coalesce leaf-struct and DNF row-group range reads for remote ``hf://`` HATS partitions.

    TODO(upstream): Periodically check whether BOTH upstream ``nested-pandas`` PRs have been
    merged and released:
      - https://github.com/lincc-frameworks/nested-pandas/pull/586
      - https://github.com/lincc-frameworks/nested-pandas/pull/589
    Once both land in a released ``nested-pandas`` version required by ``astro/pyproject.toml``,
    delete ``enable_hf_coalesced_parquet_reads()`` (and its helpers above) and remove its call
    site in ``astropt3.data.catalog_stream``.
    """
    import hats.io.file_io as hats_file_io_pkg
    import hats.io.file_io.file_io as hats_file_io_mod

    disable_hf_cat_readahead()
    current = hats_file_io_mod.read_parquet_file_to_pandas
    if getattr(current, "_astropt3_coalesced_parquet", False):
        return

    _ALLOWED_KWARGS = {"columns", "filters", "schema", "partitioning", "use_pandas_metadata"}

    @wraps(current)
    def read_parquet_file_to_pandas(file_pointer, is_dir=None, **kwargs):
        columns = kwargs.get("columns")
        filters = kwargs.get("filters")
        if (
            is_dir is False
            and isinstance(columns, list)
            and (any("." in col for col in columns) or filters is not None)
            and not any(v is not None for k, v in kwargs.items() if k not in _ALLOWED_KWARGS)
        ):
            upath = hats_file_io_mod.get_upath(file_pointer)
            if upath.protocol == "hf" and hasattr(upath.fs, "cat_ranges"):
                try:
                    return _read_coalesced_parquet_to_nestedframe(
                        upath.fs,
                        upath.path,
                        columns=columns,
                        filters=filters,
                        schema=kwargs.get("schema"),
                        use_pandas_metadata=kwargs.get("use_pandas_metadata", True),
                    )
                except _CoalescedReadUnsupported:
                    pass
        return current(file_pointer, is_dir=is_dir, **kwargs)

    read_parquet_file_to_pandas._astropt3_coalesced_parquet = True  # type: ignore[attr-defined]
    hats_file_io_mod.read_parquet_file_to_pandas = read_parquet_file_to_pandas
    hats_file_io_pkg.read_parquet_file_to_pandas = read_parquet_file_to_pandas
