from __future__ import annotations

import json
import math
import random
import sys

import pytest

torch = pytest.importorskip("torch")

from smcfnet.config.config import Config
from smcfnet.evolution import (
    Candidate,
    EvolutionarySearch,
    Fitness,
    FireNetLIFTarget,
    GeneSpec,
    GroupedUniformCrossover,
    ParetoSelection,
    SearchArchive,
    SensitivityAwareMutation,
    CuadradoUNetTarget,
    TrainerBackedEvaluator,
    analyze_ea_archive,
    build_convergence_fitness,
    candidate_from_dict,
    expand_parameter_grid,
    load_search_state,
    parse_mcm_log_metrics,
    parse_mcm_validation_log_metrics,
    save_search_state,
    sample_parameter_space,
    summarize_metric_rows,
)
from smcfnet.evolution.firenet.search_space import FIRENET_LIF_GENE_SPECS, apply_firenet_lif_guardrails, make_firenet_lif_target
from smcfnet.evolution.cuadrado_unet.search_space import make_cuadrado_unet_target
from smcfnet.evolution.plotting import plot_ea_progress
from smcfnet.evolution.runner import build_initial_genomes, build_target
from smcfnet.evolution.ssnn_net.search_space import make_ssnn_net_target, make_ssnn_net_v2_target


def test_expand_parameter_grid_supports_nested_options():
    specs = expand_parameter_grid(
        base={"family": "firenet_lif", "head": {"readout_type": "single_scale"}},
        search_space={
            "head": {"readout_type": ["single_scale", "multi_scale"]},
            "stage_depths": [1, 2],
        },
    )

    assert len(specs) == 4
    assert {spec["head"]["readout_type"] for spec in specs} == {"single_scale", "multi_scale"}
    assert {spec["stage_depths"] for spec in specs} == {1, 2}


def test_sample_parameter_space_supports_nested_options():
    rng = random.Random(2)
    spec = sample_parameter_space(
        base={"family": "firenet_lif", "head": {"readout_type": "single_scale"}},
        search_space={"head": {"readout_type": ["single_scale", "multi_scale"]}, "stage_depths": [1, 2]},
        rng=rng,
    )

    assert spec["family"] == "firenet_lif"
    assert spec["head"]["readout_type"] in {"single_scale", "multi_scale"}
    assert spec["stage_depths"] in {1, 2}


def test_firenet_lif_target_applies_search_config():
    cfg = Config()
    target = FireNetLIFTarget(search_space={"stage_depths": [1, 2]})

    target.apply_to_config(cfg, {"family": "firenet_lif", "stage_depths": 2})

    assert cfg.model.mcm.architecture == "sofe"
    assert cfg.model.mcm.estimator_representation == "streaming_events"
    assert cfg.training.streaming_step_policy == "chunk_backward"
    assert cfg.model.mcm.sofe_architecture["family"] == "firenet_lif"
    assert cfg.model.mcm.sofe_architecture["stage_depths"] == [2, 2, 2]


def test_firenet_lif_target_applies_structured_mcm_genome():
    cfg = Config()
    target = make_firenet_lif_target()

    target.apply_to_config(
        cfg,
        {
            "mcm": {
                "events_per_step_target": 25000,
                "bptt_steps": 10,
                "flow_scaling": 8.0,
                "smooth_weight": 0.3,
                "contrast_weight": 2.0,
                "loss_weight": 0.2,
            },
            "sofe_architecture": {
                "count_scale": 2.0,
                "flow_head_init_std": 0.05,
                "spiking_neuron": {"leak": [-2.0, 0.1], "thresh": [0.8, 0.1]},
            },
        },
    )

    assert cfg.model.mcm.architecture == "sofe"
    assert cfg.model.mcm.events_per_step_target == 25000
    assert cfg.model.mcm.bptt_steps == 10
    assert cfg.model.mcm.flow_scaling == pytest.approx(8.0)
    assert cfg.model.mcm.smooth_weight == pytest.approx(0.3)
    assert cfg.model.mcm.contrast_weight == pytest.approx(2.0)
    assert cfg.model.mcm.loss_weight == pytest.approx(0.2)
    assert cfg.model.mcm.sofe_architecture["count_scale"] == pytest.approx(2.0)
    assert cfg.model.mcm.sofe_architecture["flow_head_init_std"] == pytest.approx(0.05)


def test_firenet_lif_target_extracts_seed_genome_from_config():
    cfg = Config()
    cfg.model.mcm.events_per_step_target = 15000
    cfg.model.mcm.bptt_steps = 5
    cfg.model.mcm.flow_scaling = 4.0
    cfg.model.mcm.smooth_weight = 1.0
    cfg.model.mcm.contrast_weight = 1.0
    cfg.model.mcm.loss_weight = 0.1
    cfg.model.mcm.sofe_architecture = {
        "family": "firenet_lif",
        "count_scale": 1.0,
        "firenet_spike_decay_mode": "decay",
        "flow_head_init_std": 0.01,
        "spiking_neuron": {
            "leak": [-2.0, 0.1],
            "thresh": [0.8, 0.1],
            "learn_leak": True,
            "learn_thresh": True,
            "hard_reset": True,
        },
    }
    target = make_firenet_lif_target()

    genome = target.genome_from_config(cfg)

    assert genome["mcm"]["events_per_step_target"] == 15000
    assert genome["mcm"]["bptt_steps"] == 5
    assert genome["mcm"]["flow_scaling"] == pytest.approx(4.0)
    assert genome["sofe_architecture"]["firenet_spike_decay_mode"] == "decay"
    assert genome["sofe_architecture"]["firenet_leak"] == [-2.0, 0.1]
    assert genome["sofe_architecture"]["firenet_thresh"] == [0.8, 0.1]


def test_config_seeded_initial_population_keeps_seed_and_adds_variants():
    cfg = Config()
    cfg.model.mcm.events_per_step_target = 15000
    cfg.model.mcm.bptt_steps = 5
    cfg.model.mcm.flow_scaling = 4.0
    cfg.model.mcm.sofe_architecture = {
        "family": "firenet_lif",
        "count_scale": 1.0,
        "firenet_spike_decay_mode": "decay",
        "flow_head_init_std": 0.01,
        "spiking_neuron": {
            "leak": [-2.0, 0.1],
            "thresh": [0.8, 0.1],
            "learn_leak": True,
            "learn_thresh": True,
            "hard_reset": True,
        },
    }
    target = make_firenet_lif_target()
    mutation = SensitivityAwareMutation(
        FIRENET_LIF_GENE_SPECS,
        mutation_p={"heavy": 0.0, "medium": 0.0, "light": 0.0},
        min_mutations=1,
        max_mutations=1,
    )

    initial = build_initial_genomes(
        cfg=cfg,
        target=target,
        target_name="firenet_lif",
        source="config",
        population_size=4,
        mutation=mutation,
        rng=random.Random(4),
    )

    assert initial is not None
    assert len(initial) == 4
    assert initial[0]["mcm"]["events_per_step_target"] == 15000
    assert initial[0]["sofe_architecture"]["firenet_spike_decay_mode"] == "decay"
    assert any(genome != initial[0] for genome in initial[1:])


def test_firenet_lif_guardrails_repair_predictable_oom_combo():
    result = apply_firenet_lif_guardrails(
        {
            "mcm": {"events_per_step_target": 50000, "bptt_steps": 15, "flow_scaling": 8.0},
            "sofe_architecture": {
                "count_scale": 4.0,
                "spiking_neuron": {"leak": [-2.0, 0.1], "thresh": [0.5, 0.1]},
            },
        }
    )

    assert result.valid
    assert result.genome["mcm"]["bptt_steps"] == 10
    assert result.genome["sofe_architecture"]["spiking_neuron"]["thresh"][0] == pytest.approx(0.6)
    assert result.repairs


def test_grouped_uniform_crossover_preserves_semantic_blocks():
    crossover = GroupedUniformCrossover(
        {
            "temporal": [("mcm", "events_per_step_target"), ("mcm", "bptt_steps")],
            "readout": [("mcm", "flow_scaling")],
        },
        crossover_probability=1.0,
        group_swap_probability=1.0,
    )
    left = {"mcm": {"events_per_step_target": 5000, "bptt_steps": 5, "flow_scaling": 4.0}}
    right = {"mcm": {"events_per_step_target": 50000, "bptt_steps": 15, "flow_scaling": 16.0}}

    child = crossover(left, right, random.Random(4))

    temporal_pair = (child["mcm"]["events_per_step_target"], child["mcm"]["bptt_steps"])
    assert temporal_pair in {(5000, 5), (50000, 15)}
    assert child["mcm"]["flow_scaling"] in {4.0, 16.0}


def test_grouped_uniform_crossover_blends_numeric_genes():
    crossover = GroupedUniformCrossover(
        {"readout": [("mcm", "flow_scaling")]},
        crossover_probability=1.0,
        group_swap_probability=0.0,
        gene_swap_probability=1.0,
        log_blend_paths=[("mcm", "flow_scaling")],
        blend_alpha=0.0,
    )
    left = {"mcm": {"flow_scaling": 4.0}}
    right = {"mcm": {"flow_scaling": 16.0}}

    child = crossover(left, right, random.Random(7))

    assert 4.0 <= child["mcm"]["flow_scaling"] <= 16.0
    assert child["mcm"]["flow_scaling"] not in {4.0, 16.0}


def test_grouped_uniform_crossover_can_record_metadata():
    crossover = GroupedUniformCrossover(
        {"readout": [("mcm", "flow_scaling")]},
        crossover_probability=1.0,
        group_swap_probability=0.0,
        gene_swap_probability=1.0,
        log_blend_paths=[("mcm", "flow_scaling")],
        record_metadata=True,
    )
    left = {"mcm": {"flow_scaling": 4.0}}
    right = {"mcm": {"flow_scaling": 16.0}}

    child, metadata = crossover(left, right, random.Random(7))

    assert "mcm" in child
    assert metadata["performed"] is True
    assert metadata["crossover_op"] == "grouped_uniform"
    assert "mcm.flow_scaling" in metadata["crossed_paths"]


def test_sensitivity_aware_mutation_uses_adjacent_heavy_moves():
    mutation = SensitivityAwareMutation(
        {
            ("mcm", "bptt_steps"): GeneSpec(
                path=("mcm", "bptt_steps"),
                kind="categorical",
                values=[5, 10, 15],
                sensitivity="heavy",
                ordered=True,
            )
        },
        mutation_p={"heavy": 1.0},
        random_jump_p={"heavy": 0.0},
        min_mutations=1,
        max_mutations=1,
    )

    child, metadata = mutation({"mcm": {"bptt_steps": 10}}, random.Random(5))

    assert child["mcm"]["bptt_steps"] in {5, 15}
    assert metadata["num_mutations"] == 1
    assert metadata["mutations"][0]["path"] == "mcm.bptt_steps"
    assert metadata["mutations"][0]["sensitivity"] == "heavy"


def test_sensitivity_aware_mutation_forces_minimum_mutation_and_records_metadata():
    mutation = SensitivityAwareMutation(
        {"flag": GeneSpec(path="flag", kind="bool", sensitivity="light")},
        mutation_p={"light": 0.0},
        min_mutations=1,
        max_mutations=1,
    )

    child, metadata = mutation({"flag": True}, random.Random(2))

    assert child["flag"] is False
    assert metadata["num_mutations"] == 1
    assert metadata["mutations"][0]["forced"] is True


def test_firenet_lif_mutation_preserves_pair_structures():
    mutation = SensitivityAwareMutation(
        {
            ("sofe_architecture", "spiking_neuron", "thresh"): FIRENET_LIF_GENE_SPECS[
                ("sofe_architecture", "spiking_neuron", "thresh")
            ]
        },
        mutation_p={"heavy": 1.0},
        random_jump_p={"heavy": 0.0},
        min_mutations=1,
        max_mutations=1,
    )
    genome = {"sofe_architecture": {"spiking_neuron": {"thresh": [0.8, 0.1]}}}

    child, metadata = mutation(genome, random.Random(3))

    assert isinstance(child["sofe_architecture"]["spiking_neuron"]["thresh"], list)
    assert len(child["sofe_architecture"]["spiking_neuron"]["thresh"]) == 2
    assert metadata["mutations"][0]["kind"] == "categorical"


def test_cuadrado_target_applies_constructor_surface():
    cfg = Config()
    target = CuadradoUNetTarget(search_space={"kernel_size": [5, 7]})

    target.apply_to_config(
        cfg,
        {
            "num_bins": 13,
            "mode": "ann",
            "image_height": 240,
            "image_width": 320,
            "kernel_size": 5,
            "multiply_factor": 7.5,
        },
    )

    assert cfg.model.mcm.architecture == "cuadrado_unet"
    assert cfg.model.mcm.estimator_representation == "cuadrado_histogram"
    assert cfg.data.event_bins == 13
    assert cfg.data.image_height == 240
    assert cfg.data.image_width == 320
    assert cfg.model.mcm.cuadrado_kernel_size == 5
    assert cfg.model.mcm.cuadrado_multiply_factor == 7.5


def test_make_cuadrado_unet_target_uses_search_space_presets():
    target = make_cuadrado_unet_target()
    assert target.name == "cuadrado_unet"
    assert target.search_space
    assert target.expand_search_space()


def test_ssnn_target_applies_constructor_surface():
    cfg = Config()
    target = make_ssnn_net_target(
        search_space={"sofe_architecture": {"base_channels": [16, 24]}}
    )

    target.apply_to_config(
        cfg,
        {
            "mcm": {
                "events_per_step_target": 10000,
                "bptt_steps": 8,
                "decay_tau_us": 5000.0,
                "flow_scaling": 8.0,
                "smooth_weight": 0.3,
                "contrast_weight": 2.0,
                "loss_weight": 0.2,
            },
            "sofe_architecture": {
                "family": "ssnn_net",
                "base_channels": 24,
                "kernel_size": 5,
                "ssnn_multiply_factor": 7.5,
                "flow_activation": "tanh",
                "ssnn_pred_kernel_size": 5,
            },
        },
    )

    assert cfg.model.mcm.architecture == "sofe"
    assert cfg.model.mcm.estimator_representation == "streaming_events"
    assert cfg.model.mcm.input_representation == "streaming_events"
    assert cfg.model.mcm.prefetcher_type == "streaming"
    assert cfg.model.mcm.events_per_step_target == 10000
    assert cfg.model.mcm.bptt_steps == 8
    assert cfg.model.mcm.decay_tau_us == pytest.approx(5000.0)
    assert cfg.model.mcm.flow_scaling == pytest.approx(8.0)
    assert cfg.model.mcm.sofe_architecture["family"] == "ssnn_net"
    assert cfg.model.mcm.sofe_architecture["base_channels"] == 24
    assert cfg.model.mcm.sofe_architecture["kernel_size"] == 5
    assert cfg.model.mcm.sofe_architecture["ssnn_multiply_factor"] == pytest.approx(7.5)
    assert cfg.model.mcm.sofe_architecture["flow_activation"] == "tanh"
    assert cfg.model.mcm.sofe_architecture["ssnn_pred_kernel_size"] == 5


def test_ssnn_search_space_expansion_uses_defaults():
    search_space = {"sofe_architecture": {"flow_activation": ["softsign", "tanh"]}}
    target = make_ssnn_net_target(search_space=search_space)
    specs = target.expand_search_space()
    assert specs
    assert all(spec["sofe_architecture"]["family"] == "ssnn_net" for spec in specs)
    activations = {spec["sofe_architecture"]["flow_activation"] for spec in specs}
    assert activations == {"softsign", "tanh"}


def test_ssnn_v2_target_and_factory_apply_streaming_surface():
    cfg = Config()
    target = make_ssnn_net_v2_target(
        search_space={
            "sofe_architecture": {
                "ssnn_encoder_signal": ["membrane", "spike"],
                "ssnn_readout": ["single", "pyramid_residual"],
            }
        }
    )
    runner_target = build_target("ssnn_net_v2", {})

    target.apply_to_config(
        cfg,
        {
            "mcm": {
                "events_per_step_target": 25000,
                "bptt_steps": 8,
                "decay_tau_us": 20000.0,
                "flow_scaling": 8.0,
                "smooth_weight": 0.7,
                "contrast_weight": 1.0,
                "loss_weight": 0.1,
            },
            "sofe_architecture": {
                "ssnn_stage_channels": [16, 32, 64, 128],
                "ssnn_bottleneck_dilations": [1, 2],
                "ssnn_encoder_signal": "spike",
                "ssnn_edge_tap": "none",
                "ssnn_refine_channels": 16,
                "ssnn_refine_layers": 1,
                "ssnn_readout": "pyramid_residual",
            },
        },
    )

    assert target.name == "ssnn_net_v2"
    assert runner_target.name == "ssnn_net_v2"
    assert cfg.model.mcm.architecture == "sofe"
    assert cfg.model.mcm.estimator_representation == "streaming_events"
    assert cfg.model.mcm.input_representation == "streaming_events"
    assert cfg.model.mcm.prefetcher_type == "streaming"
    assert cfg.model.mcm.events_per_step_target == 25000
    assert cfg.model.mcm.bptt_steps == 8
    assert cfg.model.mcm.decay_tau_us == pytest.approx(20000.0)
    assert cfg.model.mcm.flow_scaling == pytest.approx(8.0)
    arch = cfg.model.mcm.sofe_architecture
    assert arch["family"] == "ssnn_net"
    assert arch["ssnn_version"] == "v2"
    assert arch["ssnn_stage_channels"] == [16, 32, 64, 128]
    assert arch["ssnn_bottleneck_dilations"] == [1, 2]
    assert arch["ssnn_encoder_signal"] == "spike"
    assert arch["ssnn_edge_tap"] == "none"
    assert arch["ssnn_refine_channels"] == 16
    assert arch["ssnn_refine_layers"] == 1
    assert arch["ssnn_readout"] == "pyramid_residual"


@pytest.mark.parametrize("target_name", ["cuadrado_unet", "ssnn_net"])
def test_build_initial_genomes_supports_config_seed(target_name):
    cfg = Config()
    cfg.evolution.initial_population_source = "config"
    cfg.evolution.population_size = 2

    if target_name == "cuadrado_unet":
        target = make_cuadrado_unet_target()
        cfg.model.mcm.mode = "snn"
        cfg.model.mcm.cuadrado_kernel_size = 7
        cfg.model.mcm.cuadrado_multiply_factor = 10.0
        cfg.data.event_bins = 21
    else:
        target = make_ssnn_net_target()
        cfg.model.mcm.mode = "snn"
        cfg.model.mcm.architecture = "sofe"
        cfg.model.mcm.estimator_representation = "streaming_events"
        cfg.model.mcm.step_time_us = None
        cfg.model.mcm.events_per_step_target = 15000
        cfg.model.mcm.bptt_steps = 5
        cfg.model.mcm.sofe_architecture = target.default_genome()["sofe_architecture"]

    genomes = build_initial_genomes(
        cfg=cfg,
        target=target,
        target_name=target_name,
        source="config",
        population_size=2,
        mutation=None,
        rng=random.Random(0),
    )

    assert genomes is not None
    assert len(genomes) == 2
    assert genomes[0] == target.normalize_genome(genomes[0])


class _DummyTarget:
    name = "dummy"

    def default_genome(self):
        return {"score": 0.0}

    def normalize_genome(self, genome):
        return {"score": float(genome["score"])}

    def apply_to_config(self, cfg, genome):
        cfg.training.learning_rate = float(genome["score"])


class _DummyEvaluator:
    def __call__(self, *, cfg, target, candidate, context):
        del target, candidate, context
        return Fitness(metrics={"score": float(cfg.training.learning_rate)}, objectives=(float(cfg.training.learning_rate),))


def test_evolutionary_search_returns_best_candidate():
    cfg = Config()
    engine = EvolutionarySearch(
        base_config=cfg,
        target=_DummyTarget(),
        evaluator=_DummyEvaluator(),
        population_size=3,
        generations=0,
        rng=random.Random(0),
    )

    result = engine.run(initial_genomes=[{"score": 3.0}, {"score": 1.0}, {"score": 2.0}])

    assert result.best is not None
    assert result.best.fitness is not None
    assert result.best.fitness.metrics["score"] == pytest.approx(1.0)
    assert result.history[0]["population_size"] == 3


def test_evolutionary_search_state_callback_and_resume_skip_evaluated():
    cfg = Config()
    phases = []
    evaluator_calls = []

    class CountingEvaluator:
        def __call__(self, *, cfg, target, candidate, context):
            del target, context
            evaluator_calls.append(candidate.identifier)
            return Fitness(metrics={"score": float(cfg.training.learning_rate)}, objectives=(float(cfg.training.learning_rate),))

    engine = EvolutionarySearch(
        base_config=cfg,
        target=_DummyTarget(),
        evaluator=CountingEvaluator(),
        population_size=2,
        generations=0,
        rng=random.Random(0),
        on_state_update=lambda generation, population, history, phase: phases.append((generation, len(population), phase)),
    )
    evaluated = Candidate(
        genome={"score": 1.0},
        fitness=Fitness(metrics={"score": 1.0}, objectives=(1.0,)),
        generation=0,
        identifier="g0_0",
    )
    pending = Candidate(genome={"score": 2.0}, generation=0, identifier="g0_1")

    result = engine.run(initial_population=[evaluated, pending], start_generation=0)

    assert evaluator_calls == ["g0_1"]
    assert result.best.identifier == "g0_0"
    assert any(phase == "candidate_evaluated" for _, _, phase in phases)
    assert any(phase == "complete" for _, _, phase in phases)


def test_evolutionary_search_evaluates_offspring_in_next_generation():
    cfg = Config()
    evaluator_calls = []

    class CountingEvaluator:
        def __call__(self, *, cfg, target, candidate, context):
            del target, context
            evaluator_calls.append(candidate.identifier)
            return Fitness(metrics={"score": float(cfg.training.learning_rate)}, objectives=(float(cfg.training.learning_rate),))

    class IncrementMutation:
        def __call__(self, genome, rng):
            del rng
            return {"score": float(genome["score"]) + 1.0}

    engine = EvolutionarySearch(
        base_config=cfg,
        target=_DummyTarget(),
        evaluator=CountingEvaluator(),
        population_size=3,
        generations=1,
        elite_size=1,
        offspring_size=2,
        mutation=IncrementMutation(),
        rng=random.Random(0),
    )

    result = engine.run(initial_genomes=[{"score": 3.0}, {"score": 1.0}, {"score": 2.0}])

    assert evaluator_calls[:3] == ["g0_0", "g0_1", "g0_2"]
    assert evaluator_calls[3:] == ["g1_0", "g1_1"]
    assert {candidate.identifier for candidate in result.population} >= {"g0_1"}
    assert any(candidate.identifier and candidate.identifier.startswith("g1_") for candidate in result.population)


def test_evolutionary_search_multi_fidelity_rungs_promote_and_validate_final():
    cfg = Config()
    calls = []

    class RungEvaluator:
        def __call__(self, *, cfg, target, candidate, context):
            del cfg, target, context
            rung = dict(candidate.metadata.get("rung") or {})
            calls.append((candidate.identifier, rung.get("index"), rung.get("budget_iters"), rung.get("use_validation_fitness")))
            score = float(candidate.genome["score"])
            return Fitness(metrics={"score": score}, objectives=(score,))

    engine = EvolutionarySearch(
        base_config=cfg,
        target=_DummyTarget(),
        evaluator=RungEvaluator(),
        population_size=4,
        generations=0,
        elite_size=1,
        rung_budgets=[10, 30],
        rung_promote_fraction=0.5,
        min_rung_survivors=1,
        final_rung_validation_fitness=True,
        rng=random.Random(0),
    )

    result = engine.run(initial_genomes=[{"score": 4.0}, {"score": 1.0}, {"score": 3.0}, {"score": 2.0}])

    assert [call[1] for call in calls].count(0) == 4
    assert [call[1] for call in calls].count(1) == 2
    assert all(call[2] == 10 for call in calls if call[1] == 0)
    assert all(call[2] == 30 for call in calls if call[1] == 1)
    assert all(call[3] is True for call in calls if call[1] == 1)
    assert result.best is not None
    assert result.best.identifier in {"g0_1_r1", "g0_3_r1"}


def test_evolutionary_search_reuses_completed_elite_rung_evaluations():
    cfg = Config()
    calls = []

    class CountingRungEvaluator:
        def __call__(self, *, cfg, target, candidate, context):
            del cfg, target, context
            rung = dict(candidate.metadata.get("rung") or {})
            calls.append((candidate.identifier, rung.get("index"), candidate.metadata.get("eval_id")))
            score = float(candidate.genome["score"])
            return Fitness(metrics={"score": score}, objectives=(score,), notes={"quality_source": "total"})

    class WorsenMutation:
        def __call__(self, genome, rng):
            del rng
            return {"score": float(genome["score"]) + 10.0}

    engine = EvolutionarySearch(
        base_config=cfg,
        target=_DummyTarget(),
        evaluator=CountingRungEvaluator(),
        population_size=3,
        generations=2,
        elite_size=1,
        offspring_size=2,
        mutation=WorsenMutation(),
        rung_budgets=[10, 30],
        rung_promote_fraction=0.5,
        min_rung_survivors=1,
        rng=random.Random(0),
    )

    result = engine.run(initial_genomes=[{"score": 1.0}, {"score": 2.0}, {"score": 3.0}])

    evaluated_ids = [candidate_id for candidate_id, _, _ in calls]
    assert evaluated_ids.count("g0_0_r0") == 1
    assert evaluated_ids.count("g0_0_r1") == 1
    assert all(eval_id is not None for _, _, eval_id in calls)
    assert len(calls) < 3 * 3 * 2
    assert result.best is not None
    assert result.best.identifier == "g0_0_r1"


def test_pareto_selection_rejects_mixed_quality_sources():
    candidates = [
        Candidate(
            genome={"score": 1.0},
            fitness=Fitness(objectives=(1.0,), notes={"quality_source": "total"}),
            identifier="proxy",
        ),
        Candidate(
            genome={"score": 2.0},
            fitness=Fitness(objectives=(2.0,), notes={"quality_source": "validation"}),
            identifier="validation",
        ),
    ]

    with pytest.raises(ValueError, match="mixed quality sources"):
        ParetoSelection()(candidates, 1)


def test_metric_parser_and_convergence_fitness(tmp_path):
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "[INFO] [MCM] iter=0 total=10 mcm_epe=9 mcm_contrast=-0.3 flow_mag_mean=0.2 "
        "flow_centered_mag_mean=0.1 flow_mean_flow_ratio=0.99 flow_dx_mean=0.001 "
        "flow_dy_mean=0.002 mcm_smooth_raw=0.003 gpu_mem_peak_allocated_mb=1000 "
        "G1_spike_rate=0.03 R1a_spike_rate=0.04 R1b_spike_rate=0.05 "
        "G2_spike_rate=0.02 R2a_spike_rate=0.03 R2b_spike_rate=0.04 "
        "num_steps=8 num_readouts=1\n"
        "[INFO] [MCM] iter=5 total=3 mcm_epe=2 mcm_contrast=-0.7 flow_mag_mean=6 "
        "flow_centered_mag_mean=5 flow_mean_flow_ratio=0.5 flow_dx_mean=0.1 "
        "flow_dy_mean=0.09 flow_u_spatial_std_mean=2.0 flow_v_spatial_std_mean=1.8 "
        "mcm_smooth_raw=0.2 gpu_mem_peak_allocated_mb=1200 "
        "G1_spike_rate=0.05 R1a_spike_rate=0.06 R1b_spike_rate=0.07 "
        "G2_spike_rate=0.08 R2a_spike_rate=0.09 R2b_spike_rate=0.10 "
        "num_steps=10 num_readouts=2\n"
        "[INFO] [MCM][VAL] epoch=0 iter=5 total=30 mcm_epe=20 flow_mag_mean=0.1\n"
        "[INFO] [MCM] Epoch 0 done in 12.0s total=4 mcm_epe=3\n"
    )

    rows = parse_mcm_log_metrics(log_path)
    metrics = summarize_metric_rows(rows)
    fitness = build_convergence_fitness(metrics, rows=rows, budget_iters=3000)

    assert len(rows) == 2
    assert metrics["train/mcm_epe_mean"] == pytest.approx(5.5)
    assert metrics["train/mcm_epe_median"] == pytest.approx(5.5)
    assert fitness.valid
    assert len(fitness.objectives) == 4
    assert fitness.metrics["fitness/epe_late"] == pytest.approx(2.0)
    assert fitness.metrics["fitness/epe_improvement"] == pytest.approx(7.0)
    assert fitness.metrics["fitness/quality_late"] == pytest.approx(3.0)
    assert fitness.metrics["fitness/quality_improvement"] == pytest.approx(7.0)
    assert fitness.metrics["fitness/centered_ratio_late"] == pytest.approx(5.0 / 6.0)
    assert fitness.metrics["fitness/iwe_contrast_late"] == pytest.approx(-0.7)
    assert fitness.metrics["fitness/smooth_raw_late"] == pytest.approx(0.2)
    assert fitness.metrics["fitness/warp_structure_proxy_late"] > 0.0
    assert fitness.notes["quality_source"] == "total"
    assert fitness.objectives[0] == pytest.approx(fitness.metrics["fitness/objective_quality"])
    assert fitness.objectives[0] < fitness.metrics["fitness/quality_late"]
    assert fitness.objectives[1] < 0.1


def test_convergence_fitness_uses_total_quality_when_epe_absent(tmp_path):
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "[INFO] [MCM] iter=0 total=129.3 mcm_contrast=1.8 flow_mag_mean=1.1 "
        "flow_centered_mag_mean=1.0 flow_mean_flow_ratio=0.3 "
        "flow_u_spatial_std_mean=0.9 flow_v_spatial_std_mean=0.8 "
        "mcm_smooth_raw=127500 gpu_mem_peak_allocated_mb=6276\n"
        "[INFO] [MCM] iter=90 total=3.75 mcm_contrast=1.9 flow_mag_mean=0.47 "
        "flow_centered_mag_mean=0.19 flow_mean_flow_ratio=0.87 "
        "flow_u_spatial_std_mean=0.02 flow_v_spatial_std_mean=0.23 "
        "mcm_smooth_raw=1811 gpu_mem_peak_allocated_mb=8682\n"
    )

    rows = parse_mcm_log_metrics(log_path)
    metrics = summarize_metric_rows(rows)
    fitness = build_convergence_fitness(metrics, rows=rows, budget_iters=100)

    assert fitness.valid
    assert fitness.notes["quality_source"] == "total"
    assert fitness.metrics["fitness/epe_available"] == pytest.approx(0.0)
    assert fitness.metrics["fitness/epe_tiebreaker"] == pytest.approx(1.0e5)
    assert fitness.metrics["fitness/quality_late"] == pytest.approx(3.75)
    assert fitness.metrics["fitness/quality_improvement"] == pytest.approx(125.55)
    assert fitness.metrics["fitness/objective_quality"] == pytest.approx(fitness.objectives[0])
    assert fitness.objectives[0] < fitness.metrics["fitness/quality_late"]
    assert all(math.isfinite(float(value)) for value in fitness.objectives)


def test_validation_metric_parser_normalizes_val_keys(tmp_path):
    log_path = tmp_path / "train.log"
    log_path.write_text("[INFO] [MCM][VAL] epoch=0 iter=10 val_total=2.5 val_mcm_epe=1.25 val_iwe_contrast=0.5\n")

    rows = parse_mcm_validation_log_metrics(log_path)

    assert rows == [{"epoch": 0.0, "iter": 10.0, "total": 2.5, "mcm_epe": 1.25, "iwe_contrast": 0.5}]


def test_search_state_roundtrip(tmp_path):
    candidate = Candidate(
        genome={"score": 1.0},
        fitness=Fitness(metrics={"score": 1.0}, objectives=(1.0,)),
        generation=0,
        identifier="g0_0",
    )
    path = tmp_path / "state.json"

    save_search_state(path, target="dummy", generation=0, population=[candidate], rng=random.Random(3))
    state = load_search_state(path)

    assert state["version"] == 1
    assert state["target"] == "dummy"
    assert state["population"][0]["identifier"] == "g0_0"
    assert state["population"][0]["fitness"]["metrics"]["score"] == pytest.approx(1.0)
    restored = candidate_from_dict(state["population"][0])
    assert restored.identifier == "g0_0"
    assert restored.fitness.metrics["score"] == pytest.approx(1.0)


def test_trainer_backed_evaluator_parses_subprocess_metrics(tmp_path):
    script = (
        "import pathlib, sys, yaml;"
        "cfg=yaml.safe_load(pathlib.Path(sys.argv[-1]).read_text());"
        "out=pathlib.Path(cfg['log']['output_dir'])/cfg['log']['experiment_name'];"
        "out.mkdir(parents=True, exist_ok=True);"
        "(out/'train.log').write_text('[INFO] [MCM] iter=0 total=2 mcm_epe=1.5 flow_mag_mean=3 "
        "flow_centered_mag_mean=2.5 flow_mean_flow_ratio=0.4 flow_dx_mean=0.1 flow_dy_mean=0.1 "
        "mcm_smooth_raw=0.2 gpu_mem_peak_allocated_mb=123\\n')"
    )
    evaluator = TrainerBackedEvaluator(
        root_dir=tmp_path / "evolution",
        max_steps_per_candidate=1,
        command_prefix=[sys.executable, "-c", script],
        device="cpu",
        cwd=tmp_path,
    )
    candidate = Candidate(genome={"score": 1.0}, generation=0, identifier="g0_0")

    fitness = evaluator(cfg=Config(), target=_DummyTarget(), candidate=candidate, context=type("C", (), {"is_master": lambda self: True})())

    assert fitness.valid
    assert fitness.metrics["train/mcm_epe_mean"] == pytest.approx(1.5)
    assert (tmp_path / "evolution" / "generation_000" / "g0_0" / "metrics.json").exists()


def test_trainer_backed_evaluator_can_use_final_rung_validation_quality(tmp_path):
    script = (
        "import pathlib, sys, yaml;"
        "cfg=yaml.safe_load(pathlib.Path(sys.argv[-1]).read_text());"
        "out=pathlib.Path(cfg['log']['output_dir'])/cfg['log']['experiment_name'];"
        "out.mkdir(parents=True, exist_ok=True);"
        "(out/'train.log').write_text('[INFO] [MCM] iter=0 total=2 mcm_epe=8 flow_mag_mean=3 "
        "flow_centered_mag_mean=2.5 flow_mean_flow_ratio=0.4 flow_u_spatial_std_mean=1 flow_v_spatial_std_mean=1 "
        "mcm_smooth_raw=0.2 gpu_mem_peak_allocated_mb=123 num_steps=8 num_readouts=1\\n"
        "[INFO] [MCM][VAL] epoch=0 iter=1 val_total=1.1 val_mcm_epe=1.25\\n')"
    )
    evaluator = TrainerBackedEvaluator(
        root_dir=tmp_path / "evolution",
        max_steps_per_candidate=1,
        command_prefix=[sys.executable, "-c", script],
        device="cpu",
        cwd=tmp_path,
    )
    candidate = Candidate(
        genome={"score": 1.0},
        generation=0,
        identifier="g0_0_r1",
        metadata={"rung": {"index": 1, "budget_iters": 1, "use_validation_fitness": True}},
    )

    fitness = evaluator(cfg=Config(), target=_DummyTarget(), candidate=candidate, context=type("C", (), {"is_master": lambda self: True})())

    assert fitness.valid
    assert fitness.objectives[0] == pytest.approx(1.25)
    assert fitness.metrics["fitness/validation_quality"] == pytest.approx(1.25)
    assert fitness.notes["quality_source"] == "validation"


def test_trainer_backed_evaluator_marks_oom_subprocess_failure_invalid(tmp_path):
    script = "import sys; print('RuntimeError: CUDA out of memory while allocating tensor'); sys.exit(1)"
    evaluator = TrainerBackedEvaluator(
        root_dir=tmp_path / "evolution",
        max_steps_per_candidate=1,
        command_prefix=[sys.executable, "-c", script],
        device="cpu",
        cwd=tmp_path,
    )
    candidate = Candidate(genome={"score": 1.0}, generation=0, identifier="g0_oom")

    fitness = evaluator(cfg=Config(), target=_DummyTarget(), candidate=candidate, context=type("C", (), {"is_master": lambda self: True})())

    assert not fitness.valid
    assert fitness.notes["reason"] == "oom"
    assert fitness.objectives == (float("inf"), float("inf"), float("inf"), float("inf"))


def test_trainer_backed_evaluator_marks_non_finite_subprocess_failure_invalid(tmp_path):
    script = "import sys; print('Detected non-finite loss: nan at iter 3'); sys.exit(1)"
    evaluator = TrainerBackedEvaluator(
        root_dir=tmp_path / "evolution",
        max_steps_per_candidate=1,
        command_prefix=[sys.executable, "-c", script],
        device="cpu",
        cwd=tmp_path,
    )
    candidate = Candidate(genome={"score": 1.0}, generation=0, identifier="g0_nan")

    fitness = evaluator(cfg=Config(), target=_DummyTarget(), candidate=candidate, context=type("C", (), {"is_master": lambda self: True})())

    assert not fitness.valid
    assert fitness.notes["reason"] == "non_finite"


def test_trainer_backed_evaluator_marks_missing_metrics_invalid(tmp_path):
    script = (
        "import pathlib, sys, yaml;"
        "cfg=yaml.safe_load(pathlib.Path(sys.argv[-1]).read_text());"
        "out=pathlib.Path(cfg['log']['output_dir'])/cfg['log']['experiment_name'];"
        "out.mkdir(parents=True, exist_ok=True);"
        "(out/'train.log').write_text('trainer finished without MCM metrics\\n')"
    )
    evaluator = TrainerBackedEvaluator(
        root_dir=tmp_path / "evolution",
        max_steps_per_candidate=1,
        command_prefix=[sys.executable, "-c", script],
        device="cpu",
        cwd=tmp_path,
    )
    candidate = Candidate(genome={"score": 1.0}, generation=0, identifier="g0_missing")

    fitness = evaluator(cfg=Config(), target=_DummyTarget(), candidate=candidate, context=type("C", (), {"is_master": lambda self: True})())

    assert not fitness.valid
    assert fitness.notes["reason"] == "no_mcm_metrics"


def test_trainer_backed_evaluator_retries_transient_dataloader_failure(tmp_path):
    script = (
        "import pathlib, sys, yaml;"
        "attempt_path=pathlib.Path('attempts.txt');"
        "attempt=int(attempt_path.read_text()) if attempt_path.exists() else 0;"
        "attempt_path.write_text(str(attempt + 1));"
        "cfg=yaml.safe_load(pathlib.Path(sys.argv[-1]).read_text());"
        "out=pathlib.Path(cfg['log']['output_dir'])/cfg['log']['experiment_name'];"
        "out.mkdir(parents=True, exist_ok=True);"
        "import sys as _sys;"
        "(_sys.exit(1) if attempt == 0 else None);"
        "(out/'train.log').write_text('[INFO] [MCM] iter=0 total=2 mcm_epe=1.5 flow_mag_mean=3 "
        "flow_centered_mag_mean=2.5 flow_mean_flow_ratio=0.4 flow_dx_mean=0.1 flow_dy_mean=0.1 "
        "mcm_smooth_raw=0.2 gpu_mem_peak_allocated_mb=123\\n')"
    )
    script = script.replace("(_sys.exit(1) if attempt == 0 else None);", "((print('DataLoader worker exited unexpectedly'), _sys.exit(1)) if attempt == 0 else None);")
    evaluator = TrainerBackedEvaluator(
        root_dir=tmp_path / "evolution",
        max_steps_per_candidate=1,
        max_retries=1,
        retry_backoff_sec=0.0,
        command_prefix=[sys.executable, "-c", script],
        device="cpu",
        cwd=tmp_path,
    )
    candidate = Candidate(genome={"score": 1.0}, generation=0, identifier="g0_retry")

    fitness = evaluator(cfg=Config(), target=_DummyTarget(), candidate=candidate, context=type("C", (), {"is_master": lambda self: True})())

    assert fitness.valid
    assert fitness.metrics["candidate/retry_count"] == pytest.approx(1.0)
    assert fitness.notes["retry_count"] == 1
    assert len(fitness.notes["attempts"]) == 2
    assert (tmp_path / "evolution" / "generation_000" / "g0_retry" / "evaluator_subprocess_retry_01.log").exists()


def test_trainer_backed_evaluator_retries_transient_filesystem_failure_then_invalid(tmp_path):
    script = "import sys; print('OSError: [Errno 5] Input/output error while opening HDF5 file'); sys.exit(1)"
    evaluator = TrainerBackedEvaluator(
        root_dir=tmp_path / "evolution",
        max_steps_per_candidate=1,
        max_retries=2,
        retry_backoff_sec=0.0,
        command_prefix=[sys.executable, "-c", script],
        device="cpu",
        cwd=tmp_path,
    )
    candidate = Candidate(genome={"score": 1.0}, generation=0, identifier="g0_fs")

    fitness = evaluator(cfg=Config(), target=_DummyTarget(), candidate=candidate, context=type("C", (), {"is_master": lambda self: True})())

    assert not fitness.valid
    assert fitness.notes["reason"] == "filesystem_transient"
    assert fitness.notes["retry_count"] == 2
    assert len(fitness.notes["attempts"]) == 3


def test_search_archive_writes_candidate_records(tmp_path):
    candidate = Candidate(
        genome={"mcm": {"flow_scaling": 4.0}},
        fitness=Fitness(
            metrics={
                "fitness/epe_late": 8.0,
                "fitness/flow_mag_late": 3.0,
                "fitness/objective_quality": 8.0,
                "fitness/objective_flow_structure": 0.2,
                "fitness/objective_snn_stability": 0.1,
                "fitness/objective_compute_cost": 0.5,
            },
            objectives=(8.0, 0.2, 0.1, 0.5),
            maximize=(False, False, False, False),
            valid=True,
        ),
        generation=0,
        identifier="g0_0",
        parents=("p0", "p1"),
        metadata={"mutation": {"mutations": [{"path": "mcm.flow_scaling"}]}},
    )
    archive = SearchArchive(tmp_path, budget_iters=3000, seed=42)

    archive.on_candidate_evaluated(candidate)

    text = (tmp_path / "candidates.jsonl").read_text()
    assert '"candidate_id": "g0_0"' in text
    assert '"f_quality": 8.0' in text
    assert (tmp_path / "candidates.csv").exists()


def test_analyze_ea_archive_writes_correlation_and_importance(tmp_path):
    results = tmp_path / "candidates.jsonl"
    rows = []
    for index in range(5):
        rows.append(
            {
                "candidate_id": f"g0_{index}",
                "generation": 0,
                "status": "completed",
                "genome": {
                    "mcm": {
                        "flow_scaling": 2.0 + index,
                        "bptt_steps": 3 + index,
                    },
                    "sofe_architecture": {
                        "count_scale": 1.0 + 0.1 * index,
                    },
                },
                "objectives": {
                    "f_quality": 10.0 - index,
                    "f_flow_structure": 2.0 - 0.2 * index,
                    "f_snn_stability": 0.1 + 0.01 * index,
                    "f_compute_cost": 0.5 + 0.1 * index,
                },
                "metrics_late": {
                    "mcm_epe_median": 10.0 - index,
                    "flow_mag_mean_median": 1.0 + index,
                    "centered_ratio": 0.6 + 0.05 * index,
                    "flow_mean_flow_ratio_median": 0.9 - 0.05 * index,
                    "spatial_std_median": 0.5 + 0.1 * index,
                    "gpu_peak_mb": 1000.0 + 10.0 * index,
                    "wall_ms_per_iter": 100.0 + index,
                },
            }
        )
    results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = analyze_ea_archive(results, tmp_path / "analysis")

    assert summary["completed_records"] == 5
    assert (tmp_path / "analysis" / "objective_correlations.csv").exists()
    assert (tmp_path / "analysis" / "hyperparameter_importance.csv").exists()
    assert "genome.mcm.flow_scaling" in (tmp_path / "analysis" / "hyperparameter_importance.csv").read_text()


def test_plot_ea_progress_writes_qualitative_mutation_and_lineage_plots(tmp_path):
    records = [
        {
            "candidate_id": "g0_0",
            "generation": 0,
            "parent_ids": [],
            "mutation_paths": [],
            "status": "completed",
            "genome": {"mcm": {"flow_scaling": 4.0, "bptt_steps": 5, "events_per_step_target": 75000, "smooth_weight": 0.1}, "sofe_architecture": {"count_scale": 1.0, "firenet_leak": [-4.0], "firenet_thresh": [0.8], "surrogate_alpha": 2.0}},
            "objectives": {"f_quality": 9.0, "f_flow_structure": 1.0, "f_snn_stability": 0.1, "f_compute_cost": 0.5},
            "metrics_late": {"mcm_epe_median": 9.0, "flow_mag_mean_median": 1.0, "flow_centered_mag_mean_median": 0.8, "centered_ratio": 0.8, "flow_mean_flow_ratio_median": 0.5, "spatial_std_median": 0.7, "iwe_contrast_median": 0.5, "smooth_raw_median": 0.1, "warp_structure_proxy": 0.4},
        },
        {
            "candidate_id": "g0_1",
            "generation": 0,
            "parent_ids": [],
            "mutation_paths": [],
            "status": "completed",
            "genome": {"mcm": {"flow_scaling": 5.0, "bptt_steps": 5, "events_per_step_target": 75000, "smooth_weight": 0.2}, "sofe_architecture": {"count_scale": 1.2, "firenet_leak": [-4.2], "firenet_thresh": [0.9], "surrogate_alpha": 2.0}},
            "objectives": {"f_quality": 8.0, "f_flow_structure": 0.9, "f_snn_stability": 0.1, "f_compute_cost": 0.6},
            "metrics_late": {"mcm_epe_median": 8.0, "flow_mag_mean_median": 1.5, "flow_centered_mag_mean_median": 1.2, "centered_ratio": 0.8, "flow_mean_flow_ratio_median": 0.45, "spatial_std_median": 0.9, "iwe_contrast_median": 0.6, "smooth_raw_median": 0.12, "warp_structure_proxy": 0.66},
        },
        {
            "candidate_id": "g1_0",
            "generation": 1,
            "parent_ids": ["g0_0", "g0_1"],
            "mutation_paths": ["mcm.flow_scaling"],
            "status": "completed",
            "genome": {"mcm": {"flow_scaling": 6.0, "bptt_steps": 5, "events_per_step_target": 75000, "smooth_weight": 0.2}, "sofe_architecture": {"count_scale": 1.2, "firenet_leak": [-4.2], "firenet_thresh": [0.9], "surrogate_alpha": 2.0}},
            "objectives": {"f_quality": 7.0, "f_flow_structure": 0.7, "f_snn_stability": 0.1, "f_compute_cost": 0.7},
            "metrics_late": {"mcm_epe_median": 7.0, "flow_mag_mean_median": 2.0, "flow_centered_mag_mean_median": 1.8, "centered_ratio": 0.9, "flow_mean_flow_ratio_median": 0.35, "spatial_std_median": 1.1, "iwe_contrast_median": 0.9, "smooth_raw_median": 0.15, "warp_structure_proxy": 1.17},
        },
    ]
    results = tmp_path / "candidates.jsonl"
    results.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    plot_ea_progress(results, tmp_path / "plots")

    assert (tmp_path / "plots" / "07_qualitative_warp_iwe_metrics.png").exists()
    assert (tmp_path / "plots" / "08_mutation_impact.png").exists()
    assert (tmp_path / "plots" / "09_lineage_quality.png").exists()
