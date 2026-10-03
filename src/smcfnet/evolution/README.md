# Evolution Search

This package contains the generic evolutionary search scaffolding for MCM/OFE
experiments. The current FireNet-LIF path is usable for trainer-backed smoke
evaluations, but it is not yet a complete long-running EA system.

## Current Architecture

The code is split into generic EA pieces and target-specific adapters:

- `framework.py` owns the EA loop, candidates, fitness values, Pareto selection,
  crossover, mutation hooks, and DDP-aware context helpers.
- `search_spaces.py` owns nested discrete search-space expansion and random
  sampling.
- `targets.py` translates a genome into a concrete `Config`.
- `evaluator.py` runs the normal training entrypoint in a subprocess and turns
  `train.log` metrics into a `Fitness`.
- `metrics.py` parses MCM logs and builds convergence-oriented objectives.
- `archive.py` writes candidate-level and generation-level JSONL/CSV/Markdown
  archives and optional W&B logs.
- `plotting.py` generates the standard EA monitoring figures from
  `candidates.jsonl`.
- `state.py` serializes population state and RNG state.
- `runner.py` provides the command-line entrypoint.
- `firenet/search_space.py` contains Xu-style FireNet-LIF search-space presets
  and FireNet-specific guardrails.
- `hagenaars_firenet/search_space.py` contains the Hagenaars/Paredes-Valles
  full-resolution FireNet-LIF presets and guardrails.
- `cuadrado_unet/search_space.py` contains Cuadrado U-Net presets and search
  helpers for architecture sweeps.
- `ssnn_net/search_space.py` contains the streamlined SSNN Net presets and
  search helpers.

The intended boundary is:

- the EA loop should not know anything about FireNet, DSEC, SOFE, or MCM;
- the target adapter should know how to normalize a genome and apply it to a
  config;
- the evaluator should know how to run a trainer and score logs, but not how to
  interpret architecture-specific genome fields;
- FireNet-specific constraints should stay in the target-specific search-space
  folder or target helper.

## FireNet-LIF Genome

The FireNet-LIF target uses a structured genome:

```yaml
mcm:
  events_per_step_target: 15000
  bptt_steps: 5
  flow_scaling: 4.0
  smooth_weight: 1.0
  contrast_weight: 1.0
  loss_weight: 0.1
sofe_architecture:
  family: firenet_lif
  count_scale: 1.0
  firenet_downsample_mode: sum
  firenet_spike_decay_mode: hold
  flow_head_init_std: 0.01
  surrogate_alpha: 10.0
  detach_reset: true
  spiking_neuron:
    leak: [-2.0, 0.1]
    thresh: [0.8, 0.1]
    learn_leak: true
    learn_thresh: true
    hard_reset: true
```

`FireNetLIFTarget.normalize_genome()` merges candidate genomes with this default
shape, normalizes the SOFE architecture spec, applies FireNet-LIF guardrails, and
normalizes the SOFE spec again after repair.

`FireNetLIFTarget.apply_to_config()` then enforces the streaming MCM setup:

- `model.mcm.enabled = True`
- `model.mcm.architecture = sofe`
- `model.mcm.estimator_representation = streaming_events`
- `model.mcm.input_representation = streaming_events`
- `model.mcm.prefetcher_type = streaming`
- `training.streaming_step_policy = chunk_backward`

The target also applies the sampled MCM and SOFE fields to the config.

## Hagenaars FireNet-LIF Genome

The Hagenaars target is selected with `--target hagenaars_firenet_lif`. It uses
the same generic EA loop, evaluator, grouped crossover, and sensitivity-aware
mutation as `firenet_lif`, but its schema is separate because the full-resolution
implementation does not use the Xu-style `firenet_input_downsample`,
`spiking_neuron.leak`, or two-element `spiking_neuron.thresh` genes as primary
architecture controls.

The target-specific files are:

- `hagenaars_firenet/search_space.py`: search-space values, gene specs,
  crossover groups, blend fields, and guardrails.
- `targets.py::HagenaarsFireNetLIFTarget`: config adapter for
  `family: hagenaars_firenet_lif`.

The default Hagenaars genome includes:

```yaml
mcm:
  events_per_step_target: 15000
  bptt_steps: 5
  decay_tau_us: 10000.0
  flow_scaling: 4.0
  smooth_weight: 1.0
  contrast_weight: 1.0
  loss_weight: 0.1
sofe_architecture:
  family: hagenaars_firenet_lif
  count_encoding: raw
  count_scale: 1.0
  base_channels: 32
  kernel_size: 3
  spiking_v_threshold: 1.0
  surrogate_function: atan
  surrogate_alpha: 1.0
  detach_reset: true
  firenet_spike_decay_mode: decay
  flow_activation: softsign
  flow_head_init_std: 0.01
  hagenaars_use_bias: true
  hagenaars_pred_kernel_size: 3
```

The main difference from the Xu-style target is that Hagenaars FireNet is
full-resolution, so `base_channels`, `events_per_step_target`, and `bptt_steps`
are more memory-sensitive. The guardrails cap obviously unsafe chunk/BPTT
combinations before launching candidate training, while leaving runtime OOMs to
the evaluator failure handling.

## Repairing FireNet-LIF Genomes

Repair is target-specific. It happens during FireNet-LIF genome normalization via
`apply_firenet_lif_guardrails()`.

Current repair and validation rules:

- reject non-positive `events_per_step_target`;
- reject `bptt_steps < 1`;
- reject non-positive `flow_scaling`;
- cap `bptt_steps` when `events_per_step_target * bptt_steps > 500000`;
- raise the threshold mean to `0.6` when `count_scale >= 4.0` and threshold mean
  is at or below `0.5`;
- warn when `count_scale >= 4.0` and `events_per_step_target >= 50000`.

These repairs are only cheap static guardrails. They cannot eliminate all unsafe
genomes. Runtime CUDA OOM, non-finite loss, invalid tensor shapes, or bad data
windows must be handled by the evaluator as invalid candidates.

The trainer-backed evaluator currently does this by returning
`Fitness(valid=False, objectives=(inf, inf, inf, inf))` when a candidate times
out, exits with an error, produces no MCM log rows, or looks like an OOM or
non-finite failure.

## Fitness

The EA uses a staged multi-objective minimization fitness. Invalid candidates
are first removed by hard feasibility checks for NaNs, OOMs, timeouts, missing
MCM logs, and non-finite trainer failures. Feasible candidates are scored with:

- `f_quality`: robust late-window optical-flow quality,
  `median_late(mcm_epe) + 0.1 * IQR_late(mcm_epe)`.
- `f_flow_structure`: a penalty for under-scaled, flat, or globally collapsed
  flow using flow magnitude, centered-flow magnitude, mean-flow dominance, and
  spatial flow variance.
- `f_snn_stability`: a penalty for silent/saturated spiking dynamics, voltage
  excursions when logged, and noisy late-window EPE.
- `f_compute_cost`: a compute proxy using peak CUDA memory, streaming step
  count, readout count, wall-clock time per iteration, and spike activity.

Short-budget evaluations use relaxed flow-structure thresholds:

```text
<= 3000 iters:   flow_mag >= 1.0, spatial_std >= 0.5
<= 12000 iters:  flow_mag >= 3.0, spatial_std >= 1.0
> 12000 iters:   flow_mag >= 5.0, spatial_std >= 3.0
```

The fitness also records promotion metadata such as `fitness/epe_improvement`,
`fitness/centered_ratio_late`, and `fitness/flow_mean_flow_ratio_late`. This is
intentional for FireNet-LIF: a candidate that still has mediocre early EPE can
remain valuable if the late window shows emerging non-global, spatially varying
flow.

For fast, repeatable supervised DSEC-flow convergence tests, the base config or
runner can restrict the flow-aligned index before sampling:

```bash
--flow-index-start 0 \
--flow-index-count 64 \
--flow-index-stride 1
```

These values map to `data.flow_index_start`, `data.flow_index_count`, and
`data.flow_index_stride`. They are applied per selected sequence before the
infinite training sampler is constructed, which keeps candidate comparisons
deterministic across DataLoader workers and DDP ranks when the sampler seed is
fixed.

Generation 0 can also be seeded around a known working FireNet-LIF run config:

```bash
--config outputs/debug/firenet_lif_local/firenet_lif_local_15/config.yaml \
--initial-population-source config
```

With `config` seeding, candidate `g0_0` is the normalized genome extracted from
the config's `model.mcm` and `model.mcm.sofe_architecture` fields. The rest of
the initial population is filled with sensitivity-aware mutation variants around
that seed. This is preferred when a manual run has shown delayed but promising
flow-structure emergence, because generation 0 explores a local neighborhood
instead of restarting from the broad default search space.

## Crossover

There are two crossover operators:

- `RandomCrossover`: legacy top-level uniform crossover.
- `GroupedUniformCrossover`: grouped mixed-type uniform crossover over nested
  genome paths.

`GroupedUniformCrossover` is the intended default for FireNet-LIF. It is generic:
the framework only knows about nested paths, groups, and optional numeric blend
metadata. FireNet-LIF supplies the semantic groups in
`firenet/search_space.py`.

The FireNet-LIF groups are:

- `temporal_partitioning`: event partitioning fields such as
  `mcm.events_per_step_target`.
- `training_bptt`: BPTT and loss-weight fields such as `mcm.bptt_steps`,
  `mcm.smooth_weight`, `mcm.contrast_weight`, and `mcm.loss_weight`.
- `lif_dynamics`: LIF leak, threshold, reset behavior, surrogate slope, detach
  reset, and spike decay mode.
- `architecture_capacity`: input count scaling and FireNet downsample mode.
- `flow_readout`: output/readout scale fields such as `mcm.flow_scaling` and
  `sofe_architecture.flow_head_init_std`.

The operator works as follows:

1. Start with a deep copy of the left parent genome.
2. With probability `crossover_probability`, perform crossover. Otherwise keep
   the left parent unchanged.
3. For each semantic group, choose group-level or gene-level mixing.
4. In group-level mode, inherit the whole semantic block from one parent.
5. In gene-level mode, mix individual paths inside the group.
6. For selected numeric paths, use arithmetic or log-space blend crossover.
7. The resulting child is normalized by the target, which applies FireNet-LIF
   repair before evaluation.

The default CLI probabilities are:

```text
crossover_probability = 0.6
group_swap_probability = 0.5
gene_swap_probability = 0.5
```

These can be changed with:

```bash
--crossover-probability 0.6 \
--group-swap-probability 0.5 \
--gene-swap-probability 0.5
```

FireNet-LIF uses log-space blend crossover for positive scale and loss fields:

- `mcm.flow_scaling`
- `mcm.smooth_weight`
- `mcm.contrast_weight`
- `mcm.loss_weight`
- `sofe_architecture.flow_head_init_std`
- `sofe_architecture.surrogate_alpha`

It uses arithmetic blend crossover for signed or paired LIF parameters:

- `sofe_architecture.spiking_neuron.leak`
- `sofe_architecture.spiking_neuron.thresh`

Categorical and integer fields use parent-choice uniform crossover.

This is intentionally not single-point crossover. The FireNet-LIF genome is a
mixed set of temporal, spiking, architectural, training, and output-scaling
hyperparameters. Gene order is arbitrary, and useful interactions are not only
local in a linear chromosome. Grouped uniform crossover preserves semantic
blocks while still allowing useful traits from both parents to recombine.

## Mutation

There are two mutation operators:

- `RandomMutation`: legacy no-op mutation that returns a deep copy of the input
  genome.
- `SensitivityAwareMutation`: generic type-aware, sensitivity-aware local
  mutation over `GeneSpec` entries.

`SensitivityAwareMutation` is the intended default for FireNet-LIF. It is
generic: the EA framework knows how to walk nested paths, mutate values by type,
and record metadata. FireNet-LIF supplies the schema in
`firenet/search_space.py` through `FIRENET_LIF_GENE_SPECS`.

Each `GeneSpec` defines:

```python
GeneSpec(
    path=("mcm", "events_per_step_target"),
    kind="categorical",
    values=[5000, 10000, 15000, 25000, 50000],
    sensitivity="heavy",
    ordered=True,
)
```

Supported gene kinds are:

- `categorical`
- `int`
- `float`
- `log_float`
- `bool`

The default FireNet-LIF mutation probabilities are:

```text
heavy_p = 0.05
medium_p = 0.12
light_p = 0.20

heavy_random_jump_p = 0.05
medium_random_jump_p = 0.20
light_random_jump_p = 0.70

min_mutations_per_child = 1
max_mutations_per_child = 3
```

These can be changed with:

```bash
--mutation-heavy-p 0.05 \
--mutation-medium-p 0.12 \
--mutation-light-p 0.20 \
--mutation-heavy-random-jump-p 0.05 \
--mutation-medium-random-jump-p 0.20 \
--mutation-light-random-jump-p 0.70 \
--min-mutations-per-child 1 \
--max-mutations-per-child 3
```

Sensitivity controls how aggressively a field changes:

- heavy fields: expensive or sensitive parameters that should change rarely and
  preferably by adjacent options, such as `events_per_step_target`, `bptt_steps`,
  `count_scale`, leak, and threshold;
- medium fields: important but less explosive fields, such as `flow_scaling`,
  `smooth_weight`, `flow_head_init_std`, and spike decay mode;
- light fields: safer fields that can mutate more frequently, such as
  `surrogate_alpha`, `detach_reset`, and `hard_reset`.

Mutation behavior:

- ordered categorical heavy genes usually move to an adjacent option;
- medium categorical genes mostly move locally but can jump more often;
- light categorical genes can make broader random flips;
- booleans flip directly;
- bounded integer genes use adjacent step moves unless a jump is sampled;
- bounded float genes use additive Gaussian perturbation;
- bounded log-float genes use multiplicative perturbation
  `old * exp(normal(0, sigma))`.

For the current FireNet-LIF schema, most fields are discrete categorical choices
from the search space. LIF `leak` and `thresh` are categorical two-element lists,
so mutation preserves the exact pair structure expected by the FireNet-LIF
constructor.

The mutation operator returns metadata in addition to the mutated genome.
`EvolutionarySearch._breed()` stores it under `Candidate.metadata["mutation"]`:

```json
{
  "mutation_op": "sensitivity_aware_type_mutation",
  "num_mutations": 1,
  "mutations": [
    {
      "path": "mcm.bptt_steps",
      "old": 5,
      "new": 10,
      "kind": "categorical",
      "sensitivity": "heavy"
    }
  ]
}
```

Target repair remains outside the generic mutation operator. After mutation, the
child genome is passed through `target.normalize_genome()`, which applies
FireNet-LIF guardrails before evaluation.

## Running

Minimal FireNet-LIF smoke search:

```bash
PYTHONPATH=src /cluster/work/kristvje/miniconda3/envs/smcm-mcfnet/bin/python -m smcfnet.evolution.runner \
  --config SMCM-MCFNet/src/smcfnet/config/runs/sofe/firenet_lif/_base_firenet_lif.yaml \
  --target firenet_lif \
  --population-size 4 \
  --generations 1 \
  --max-steps-per-candidate 100 \
  --device cuda \
  --num-gpus 1
```

Multi-fidelity rung search:

```bash
PYTHONPATH=src /cluster/work/kristvje/miniconda3/envs/smcm-mcfnet/bin/python -m smcfnet.evolution.runner \
  --config outputs/debug/firenet_lif_local/firenet_lif_local_15/config.yaml \
  --target firenet_lif \
  --initial-population-source config \
  --population-size 8 \
  --generations 2 \
  --elite-size 2 \
  --offspring-size 6 \
  --rung-budgets 300,3000,12000 \
  --rung-promote-fraction 0.5 \
  --min-rung-survivors 2 \
  --final-rung-validation-fitness \
  --evaluator-max-retries 1 \
  --evaluator-retry-backoff-sec 30 \
  --flow-index-start 0 \
  --flow-index-count 64 \
  --flow-index-stride 1 \
  --device cuda \
  --num-gpus 1
```

When `--rung-budgets` is set, each generation is evaluated with successive
halving: rung 0 evaluates the full population at the smallest budget, each
intermediate rung promotes a Pareto-selected subset, and breeding uses the
final-rung survivors. Candidate records include `rung_index`, per-row
`budget_iters`, and `metadata.rung`. `--final-rung-validation-fitness` forces
MCM validation on the final rung and replaces `f_quality` with validation EPE
when `[MCM][VAL]` rows are available. The deterministic DSEC-flow window
settings apply to both train and validation loaders, so the final validation
fitness is computed on a fixed validation subset.

Evaluator retries are intentionally narrow. By default only
`dataloader_transient` and `filesystem_transient` failures are retried; OOM,
non-finite losses, missing metrics, and generic trainer failures are recorded as
invalid candidates. Retry behavior can be configured with
`evolution.evaluator_max_retries`, `evolution.evaluator_retry_backoff_sec`, and
`evolution.evaluator_retry_failure_reasons`, or with the matching runner flags.

Outputs are written under:

```text
<cfg.log.output_dir>/<cfg.log.experiment_name>/evolution/
```

Each candidate gets:

- `candidate_config.yaml`
- `evaluator_subprocess.log`
- `train.log`
- `metrics.json`

The EA root also gets structured search archives:

- `candidates.jsonl`: one row per evaluated candidate, including genome,
  parents, operator, mutation paths, status/failure reason, objectives, late
  metrics, notes, and metadata.
- `candidates.csv`: flat mirror of `candidates.jsonl` for quick spreadsheet or
  notebook inspection.
- `generation_summary.jsonl`: one row per generation with completed/failed
  counts, failure reasons, objective best/median/worst, survivor ids, and best
  candidate id.
- `generation_summary.csv`: flat mirror of `generation_summary.jsonl`.
- `generation_summary.md`: human-readable generation summaries for logs.
- `plots/`: monitoring figures refreshed after each generation when Matplotlib
  is available.

Search state is written incrementally to:

```text
<evolution_root>/state.json
```

It is updated after each candidate evaluation, after each generation, when the
next generation is ready, and at completion. Resume with:

```bash
--resume
```

or from an explicit state path:

```bash
--resume-from <path/to/state.json>
```

On resume, candidates that already have `fitness` in `state.json` are kept and
not re-evaluated; only pending candidates are run. The serialized RNG state is
restored so future mutation/crossover choices continue from the saved search
state.

The plot suite can also be regenerated manually:

```bash
PYTHONPATH=src /cluster/work/kristvje/miniconda3/envs/smcm-mcfnet/bin/python tools/ea/plot_ea_progress.py \
  --results <evolution_root>/candidates.jsonl \
  --out <evolution_root>/plots
```

Completed archives can be analyzed offline for objective correlations and
model-free hyperparameter importance:

```bash
PYTHONPATH=src /cluster/work/kristvje/miniconda3/envs/smcm-mcfnet/bin/python tools/ea/analyze_ea_archive.py \
  --results <evolution_root>/candidates.jsonl \
  --out <evolution_root>/analysis
```

The older `scripts/ea/*.py` paths remain as compatibility wrappers.

Generated figures:

- `01_objectives_by_generation.png`
- `02_pareto_epe_vs_cost.png`
- `02b_pareto_epe_vs_structure.png`
- `03_flow_structure_by_generation.png`
- `04_parallel_coordinates_elites.png`
- `05_gene_distributions.png`
- `06_failure_modes.png`
- `07_qualitative_warp_iwe_metrics.png`
- `08_mutation_impact.png`
- `09_lineage_quality.png`

Analysis artifacts:

- `analysis/objective_correlations.csv`
- `analysis/hyperparameter_importance.csv`
- `analysis/analysis_summary.md`
- `analysis/analysis_summary.json`
- `analysis/07_objective_correlation_heatmap.png`
- `analysis/08_hyperparameter_importance.png`

## Analyzing Results

The EA output should be read in this order:

1. Check run health and failure modes.
2. Identify the final-rung completed candidates.
3. Compare candidates on the Pareto plots and objective values.
4. Check flow-structure metrics to reject global-flow or no-flow failures.
5. Check SNN stability and compute cost as secondary filters.
6. Inspect the winning genomes/configs before deciding what to continue.

The best candidate is not necessarily the one with the lowest training EPE.
For FireNet-LIF, prefer a candidate with slightly worse EPE but clearly better
spatially structured flow over a candidate with low-ish EPE and global one-way
flow. The main question is whether the candidate is learning an optical-flow
field or exploiting a weak global displacement shortcut.

### Health Check

Start with:

```text
<evolution_root>/generation_summary.md
<evolution_root>/generation_summary.csv
<evolution_root>/plots/06_failure_modes.png
```

A usable search should have most final-rung candidates marked `completed`.
Occasional OOM or timeout failures are acceptable and expected in FireNet-LIF
searches. A high rate of `no_mcm_metrics`, `non_finite`, or generic
`trainer_failed` means the search space or evaluator needs more guardrails
before launching a full job.

Use `generation_summary.md` for a quick live summary. It lists completed/failed
counts, survivor ids, best quality candidate, and best structure candidate per
generation.

### Candidate Tables

Use `candidates.csv` for quick sorting and `candidates.jsonl` when nested genome
or metadata fields matter.

Important columns in `candidates.csv`:

- `candidate_id`: candidate identifier. Rung candidates end in `_r0`, `_r1`,
  etc.
- `generation`: EA generation.
- `status`: `completed` or `failed`.
- `failure_reason`: failure class for invalid candidates.
- `rung_index`: successive-halving rung index, if rung scheduling was used.
- `budget_iters`: candidate budget for this evaluation row.
- `objectives.f_quality`: lower is better. Uses validation quality on final
  rung when `--final-rung-validation-fitness` produced validation rows.
- `objectives.f_flow_structure`: lower is better. Penalizes weak, flat, or
  globally collapsed flow.
- `objectives.f_snn_stability`: lower is better. Penalizes silent/saturated
  spike behavior and noisy EPE.
- `objectives.f_compute_cost`: lower is better. Used as a tiebreaker early in
  FireNet-LIF search.
- `metrics_late.mcm_epe_median`: late-window train EPE.
- `metrics_late.validation_mcm_epe_median`: late validation EPE if available.
- `metrics_late.flow_mag_mean_median`: late predicted flow magnitude.
- `metrics_late.centered_ratio`: centered-flow magnitude divided by total flow
  magnitude.
- `metrics_late.flow_mean_flow_ratio_median`: how much of the flow is explained
  by the global mean vector.
- `metrics_late.spatial_std_median`: spatial variation in the predicted flow.
- `metrics_late.warp_structure_proxy`: cheap proxy for non-global useful flow.
- `metrics_late.gpu_peak_mb` and `metrics_late.wall_ms_per_iter`: cost signals.

When using rung scheduling, compare final-rung rows first. A candidate can look
good at rung 0 and regress at the final rung, so do not select from early-rung
records unless the search was only a smoke test.

### Final Population

`state.json` contains the final population after selection. This is the most
direct list of candidates the EA kept:

```bash
PYTHONPATH=src /cluster/work/kristvje/miniconda3/envs/smcm-mcfnet/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("<evolution_root>")
state = json.loads((root / "state.json").read_text())
for cand in state["population"]:
    fit = cand.get("fitness") or {}
    print(cand.get("identifier"), fit.get("valid"), fit.get("objectives"))
PY
```

Use the matching candidate directories to inspect the concrete config and logs:

```text
<evolution_root>/generation_XXX/<candidate_id>/candidate_config.yaml
<evolution_root>/generation_XXX/<candidate_id>/train.log
<evolution_root>/generation_XXX/<candidate_id>/metrics.json
```

For rung candidates, the archived `candidate_id` may be `g1_0_r1`, while the
base candidate id is stored in metadata as `metadata.base_candidate_id`.

### Plot Guide

`01_objectives_by_generation.png`
: Shows best, median, and interquartile range for the four objectives. A healthy
search should move the best and median `f_quality` or `f_flow_structure`
downward over generations. If only `f_compute_cost` improves, the EA is learning
to be cheap rather than learning flow.

`02_pareto_epe_vs_cost.png`
: Shows the tradeoff between late EPE and compute cost. Use this to find cheap
candidates that are close to the quality front. Do not pick the cheapest point
unless its flow-structure metrics are also acceptable.

`02b_pareto_epe_vs_structure.png`
: This is the most important FireNet-LIF selection plot. Prefer points near the
bottom-left: low EPE and low flow-structure penalty. A point with low EPE but
high structure penalty is likely producing weak or globally biased flow.

`03_flow_structure_by_generation.png`
: Tracks late flow magnitude, centered magnitude, centered ratio, mean-flow
ratio, and spatial standard deviation. Promising FireNet-LIF candidates should
increase flow magnitude and spatial variation while keeping mean-flow ratio
low. Bad global-flow candidates often have high total flow magnitude but poor
centered ratio or high mean-flow ratio.

`04_parallel_coordinates_elites.png`
: Shows how elite genomes differ across key genes and objectives. Use this after
identifying candidate ids from the Pareto plots. Look for repeated patterns in
good candidates, such as a stable range for `events_per_step_target`,
`bptt_steps`, `flow_scaling`, `count_scale`, threshold, or leak.

`05_gene_distributions.png`
: Shows how gene values distribute by generation and quality. Use this to spot
search collapse. If all good candidates converge to one value for a heavy gene,
that value is probably useful. If values are scattered with no quality pattern,
that gene is likely neutral at the current budget.

`06_failure_modes.png`
: Shows failed candidate reasons by generation. OOM concentrated around a
specific gene pattern means guardrails should be tightened before the full run.
Many non-finite failures usually mean flow/loss scaling ranges are too broad.

`07_qualitative_warp_iwe_metrics.png`
: Tracks cheap qualitative proxies from the training logs: IWE contrast,
smoothness, warp-structure proxy, and mean-flow ratio. Use this as a weak
sanity check, not as the primary objective. A useful candidate should not only
increase contrast; it should also improve warp-structure proxy and avoid high
mean-flow ratio.

`08_mutation_impact.png`
: Shows how mutation paths changed objectives relative to the best parent. For
each mutation path, values below zero mean the mutation improved that objective.
Use this to identify which genes are helping or hurting. Treat the plot as
suggestive unless the archive has enough repeated mutations.

`09_lineage_quality.png`
: Shows quality trajectories for selected lineages. Prefer candidates from
lineages that improve consistently across generations or rungs. A single lucky
candidate with no stable lineage is less trustworthy for a long full run.

`analysis/07_objective_correlation_heatmap.png`
: Shows objective and metric correlations. If `f_quality` and
`f_flow_structure` are strongly aligned, selection is easy. If they conflict,
choose based on the research priority: for FireNet-LIF convergence, flow
structure usually matters more than small early EPE differences.

`analysis/08_hyperparameter_importance.png`
: Shows model-free Spearman importance for genome leaves. Use this to revise
the next search space. It should not by itself select a final candidate, because
importance is correlation-based and can be misleading with small populations.

### Practical Selection Rule

For each completed final-rung candidate, apply these filters:

1. Candidate must be `status=completed` and `fitness.valid=true`.
2. Reject candidates with OOM, timeout, non-finite, or missing-metric history if
   the same genome pattern repeatedly fails.
3. Prefer final-rung validation quality when available:
   `metrics_late.validation_mcm_epe_median` or `metrics_late.validation_quality`.
4. Require credible flow structure:
   `objectives.f_flow_structure` should be low relative to the population, and
   `metrics_late.flow_mean_flow_ratio_median` should not be near `1.0`.
5. Prefer `metrics_late.centered_ratio` closer to `1.0` than `0.0`; low centered
   ratio means most flow is global mean displacement.
6. Prefer non-trivial `metrics_late.spatial_std_median`; near-zero spatial
   variation means the model is not producing directional structure.
7. Check `objectives.f_snn_stability`; avoid silent or saturated spiking
   candidates unless their flow evidence is clearly superior.
8. Use `objectives.f_compute_cost`, peak memory, and wall time as tiebreakers,
   not primary objectives in the first searches.

For early FireNet-LIF searches, the best overall continuation candidate is
usually:

```text
lowest validation/train f_quality among candidates on or near the Pareto front
AND low f_flow_structure
AND acceptable f_snn_stability
AND no repeated failure/repair pattern
```

If two candidates are close, choose the one with better flow structure and more
stable SNN dynamics, even if its EPE is slightly worse. The goal of the first EA
phase is to find an architecture that learns directional, spatially varying
flow, not merely the lowest short-budget EPE.

### Quick Ranking Script

This prints completed candidates sorted by a pragmatic FireNet-LIF continuation
score. Lower is better. It prioritizes quality and flow structure, then uses
SNN stability and compute cost as lighter terms:

```bash
PYTHONPATH=src /cluster/work/kristvje/miniconda3/envs/smcm-mcfnet/bin/python - <<'PY'
import json
import math
from pathlib import Path

root = Path("<evolution_root>")
rows = []
for line in (root / "candidates.jsonl").read_text().splitlines():
    rec = json.loads(line)
    if rec.get("status") != "completed":
        continue
    obj = rec.get("objectives") or {}
    late = rec.get("metrics_late") or {}
    if not obj:
        continue
    quality = obj.get("f_quality", math.inf)
    structure = obj.get("f_flow_structure", math.inf)
    stability = obj.get("f_snn_stability", math.inf)
    cost = obj.get("f_compute_cost", math.inf)
    score = quality + 2.0 * structure + 0.5 * stability + 0.1 * cost
    rows.append((score, rec, late))

for score, rec, late in sorted(rows, key=lambda item: item[0])[:12]:
    print(
        f"{rec['candidate_id']:>14} gen={rec['generation']} rung={rec.get('rung_index')} "
        f"score={score:.4f} quality={rec['objectives'].get('f_quality')} "
        f"structure={rec['objectives'].get('f_flow_structure')} "
        f"stability={rec['objectives'].get('f_snn_stability')} "
        f"cost={rec['objectives'].get('f_compute_cost')} "
        f"val_epe={late.get('validation_mcm_epe_median')} "
        f"flow_mag={late.get('flow_mag_mean_median')} "
        f"centered={late.get('centered_ratio')} "
        f"mean_ratio={late.get('flow_mean_flow_ratio_median')} "
        f"spatial={late.get('spatial_std_median')}"
    )
PY
```

For a publication-quality decision, do not rely on this scalar score alone.
Use it to shortlist candidates, then inspect the Pareto plots, candidate
configs, and qualitative warp/flow outputs.

### Choosing What To Continue

After selecting a candidate:

1. Copy its `candidate_config.yaml` path into your experiment notes.
2. Record the source `candidate_id`, generation, rung, objectives, and key late
   metrics.
3. Run a longer single-candidate continuation with the exact config before
   committing a full EA search around that region.
4. If the longer continuation still produces global one-direction flow, reject
   the candidate even if the short-run objective looked good.
5. If it preserves directional flow structure and EPE continues to improve,
   use its genome as `evolution.base_genome` or seed the next run with
   `--initial-population-source config`.

Good continuation candidates should show the same qualitative direction across
multiple signals: improving EPE, increasing non-global flow structure, stable
spiking, and manageable runtime/memory. Candidates that only win one metric are
useful for search-space diagnosis but are weaker choices for long training.

If `wandb.use_wandb=true` and `wandb.mode` is not `disabled`, the archive writer
logs candidate and generation summaries to W&B under `ea/*` keys. Candidate
trainer subprocesses still run with W&B disabled so the search remains one
top-level EA run rather than one W&B run per candidate.

## What Remains For A Fully EA Search

Implemented since the initial scaffold:

- Deterministic DSEC-flow subset/window selection for fast repeatable
  convergence tests:
  `data.flow_index_start`, `data.flow_index_count`, and
  `data.flow_index_stride`, with runner overrides
  `--flow-index-start`, `--flow-index-count`, and `--flow-index-stride`.
- Config-seeded generation-0 populations:
  `--initial-population-source config` extracts the FireNet-LIF genome from a
  known working config and fills the rest of the initial population with local
  sensitivity-aware mutation variants.
- `evolution.base_genome` now seeds a full initial population instead of only a
  single candidate when no explicit initial-population source is provided.
- FireNet-LIF config-to-genome extraction via
  `FireNetLIFTarget.genome_from_config()`.
- Richer staged fitness for FireNet-LIF:
  late-window EPE median/IQR quality, explicit flow-structure penalty,
  SNN-stability penalty, and compute-cost objective.
- Budget-dependent flow-structure thresholds for short, medium, and longer
  candidate evaluations.
- Multi-fidelity rung scheduling via `evolution.rung_budgets` or
  `--rung-budgets`, with Pareto promotion controlled by
  `--rung-promote-fraction` and `--min-rung-survivors`.
- Final-rung validation quality via `--final-rung-validation-fitness`; the
  evaluator parses `[MCM][VAL]` rows and uses validation EPE as `f_quality` when
  available while keeping training-derived structure, SNN, and cost objectives.
- Fitness metadata for promotion decisions, including
  `fitness/epe_improvement`, `fitness/centered_ratio_late`,
  `fitness/flow_mean_flow_ratio_late`, late flow magnitude, late spatial
  structure, spike proxy, memory, steps, readouts, and wall-clock time per
  iteration.
- Parser filtering so training fitness ignores `[MCM][VAL]` and epoch-summary
  lines by default.
- Candidate wall-clock timing from evaluator subprocess runtime:
  `candidate/elapsed_sec`, `candidate/wall_time_per_iter_sec`,
  `candidate/wall_time_per_logged_iter_sec`, and
  `candidate/wall_time_per_log_row_sec`.
- Cheap qualitative warp/IWE proxies are recorded from existing MCM logs:
  late `iwe_contrast`, raw smoothness, and a flow-structure warp proxy derived
  from magnitude, centered ratio, and mean-flow dominance.
- Evaluator-level retry/backoff for transient dataloader and filesystem
  failures, with retry attempt metadata stored in candidate notes and
  `candidate/retry_count`.
- Invalid subprocess outcomes are classified as OOM, non-finite, missing MCM
  metrics, transient dataloader/filesystem failures, timeout, or trainer failed.
- Repair warnings are copied into `Candidate.metadata["repair"]`.
- Grouped crossover can record group/gene decisions into
  `Candidate.metadata["crossover"]` when enabled by the runner.
- Candidate/generation archives are written to JSONL and CSV, with a Markdown
  generation log for quick live inspection.
- Standard monitoring plots are generated from `candidates.jsonl` after each
  generation and can be regenerated with `tools/ea/plot_ea_progress.py`,
  including qualitative warp/IWE proxies, mutation-impact deltas against best
  parents, and elite lineage trajectories.
- Completed archives can be analyzed with `tools/ea/analyze_ea_archive.py`
  for objective/late-metric correlations and model-free hyperparameter
  importance based on absolute Spearman rank correlation.
- Optional W&B tracking logs EA-level candidate and generation summaries.
- Resume support in `runner.py` via `--resume` and `--resume-from`, using
  `state.py` to restore population, history, and RNG state.
- Incremental state persistence after each candidate evaluation, generation
  summary, next-generation construction, and search completion.
- Resumed searches skip already evaluated candidates and continue with pending
  candidates from the serialized population.
- Focused tests for deterministic DSEC-flow selection, config-seeded initial
  populations, resume/skip semantics, state roundtrip, crossover metadata,
  archive writing, offline archive analysis, multi-fidelity rung promotion,
  final-rung validation fitness, evaluator retry/backoff, invalid OOM-like
  subprocess logs, non-finite logs, missing metrics, qualitative plotting,
  mutation-impact plotting, lineage plotting, and late-window convergence
  fitness.

The main remaining pieces are:

- Add SLURM-oriented launch mode for multi-candidate scheduling. This may not
  be the right direction for this cluster, because many short queued jobs can
  block on queue traffic. A better full-search strategy may be to queue one job
  with N GPUs and use those GPUs internally for candidate fitness evaluation.
- Add a small dry-run trainer/fake evaluator mode for CI without DSEC or GPUs.

Until SLURM multi-candidate orchestration is in place, this is best treated as a
trainer-backed EA scaffold with usable FireNet-LIF local-neighborhood search,
multi-fidelity scheduling, validation-quality final rungs, resume, structured
archives, and live monitoring rather than a complete long-running HPC search
system.
