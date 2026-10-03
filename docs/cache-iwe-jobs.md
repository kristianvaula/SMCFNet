# SMCM IWE Cache Jobs

Offline IWE caching is the safe default when a frozen, stateful SMCM module is
used by detector training. It preserves sequence order during SMCM rollout and
then lets detector training consume a normal shuffled cache.

## Main Idea

The cache stores the detector-facing IWE tensor that live SMCM would otherwise
produce online:

```text
events_voxel: [B, 3, 480, 640]
```

Channel `0` is intentionally zero. Positive and negative IWE activity is stored
in channels `1` and `2`.

## Recommended Entry Points

- `tools/precompute_smcm_iwe.py`: direct cache generation tool.
- `slurmjobs/cache/run_precompute_smcm_iwe_template.slurm`: cluster template for reproducible cache jobs.

## Typical Cache Layout

```text
outputs/cache/smcm_iwe/<variant>/<split>/<sequence>.h5
outputs/cache/smcm_iwe/<variant>/<split>/manifest.json
outputs/cache/smcm_iwe/<variant>/slurm_logs/stdout_<job>.log
outputs/cache/smcm_iwe/<variant>/slurm_logs/stderr_<job>.log
```

Each sequence file stores:

```text
iwe: [num_frames, 3, 480, 640]
chunks: (1, 3, 480, 640)
```

## Good Practices

- Use a stable variant name that identifies the exact checkpoint.
- Keep `model.mcm.pretrained_weights` in the detector config even when using
  cached IWEs so checkpoint provenance can still be validated.
- Smoke-test a few sequences before launching a full-cache job.
- Keep cache generation and detector training on the same dataset split,
  rectification policy, and checkpoint.

## Training From Cache

Detector configs typically point at a cache directory like this:

```yaml
data:
  precomputed_iwe_dir: outputs/cache/smcm_iwe/<variant>/all
  precomputed_iwe_key: iwe
  precomputed_iwe_dtype: fp16
```

In practice, it is often easiest to create an `all/` directory containing
symlinks to the train and val split files. That avoids failures when training
and evaluation loaders share one cache provider but read different sequences.

## Validation Checklist

- manifest sequence counts match the dataset YAML
- checkpoint hash matches the intended pretrained weights
- HDF5 tensors are shaped `[N, 3, 480, 640]`
- coordinate-frame metadata matches the detector config
- later sequence frames contain non-empty activity

If cached and live outputs disagree, the usual cause is a mismatch between the
training config and the config used during precompute.