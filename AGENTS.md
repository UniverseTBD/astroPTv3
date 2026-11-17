# AGENTS.md

This file provides guidance to Agents when working with code in this repository.

## What this repo is

AstroPTv3: a from-scratch suite of multimodal astronomical foundation models
(70M–12B, Pythia-mirrored) — a SmolLM3 decoder body fed continuous image/
spectra patch tokens with per-modality regression heads, pretrained on the
Multimodal Universe. The repo is a fork of `huggingface/smollm`: `text/`,
`vision/`, `tools/` are **read-only upstream reference**; all project code
lives in `astro/`, and `nanotron/` is a git submodule pointing at the
pretraining fork (`Smith42/nanotron`, branch `astropt3`).

**Before working in `astro/`, read [`astro/AGENTS.md`](astro/AGENTS.md)** —
it carries the commands, hard constraints, and the architecture contract
for the project. The lab book — charter/roadmap (MMU Streaming × AstroPT),
phase history, and experiment log — is
[`astro/EXPERIMENTS.md`](astro/EXPERIMENTS.md);
the project README is [`astro/README.md`](astro/README.md).

Repo-wide constraints: this repo runs on several machines — do not assume
GPU count, memory, or scheduler from any one of them. The CPU suite is
the fast gate and must pass with no GPUs present; GPU-marked tests and
real training runs execute wherever a suitable GPU environment exists,
and multi-day runs belong on the training cluster by preference.
Dependencies are managed only through uv in `astro/pyproject.toml`.

Change policy: breaking changes are allowed. Do not add compatibility shims,
deprecation paths, migration notes, or support for old checkpoints and YAMLs;
when something changes, update every caller and recipe and retrain.
Documentation describes the current code only — no "removed", "retired" or
"formerly" notes. Git history is the change record; the lab book records
experiments and results, not code changes.
