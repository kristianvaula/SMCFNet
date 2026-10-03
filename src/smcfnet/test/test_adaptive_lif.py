"""Tests for AdaptiveLIFNeuron (homeostatic threshold adaptation).

Validates the Zhang 2023 SpikeFPN formulation:
    A_th(t) = u_th + beta * a(t)
    a(t) = tau_a * a(t-1) + y(t-1)
"""

import torch
import pytest

from smcfnet.model.spiking_blocks import (
    AdaptiveLIFNeuron,
    SpikingConvBlock,
    create_neuron,
)


class TestAdaptiveLIFNeuron:
    """Unit tests for the AdaptiveLIFNeuron."""

    def test_forward_produces_spikes(self):
        """Strong input should produce some spikes (not all zeros)."""
        neuron = AdaptiveLIFNeuron(num_channels=1)
        neuron.reset_membrane()
        spikes = []
        for _ in range(10):
            s = neuron(torch.ones(1, 1, 4, 4) * 2.0)
            spikes.append(s)
        total = torch.stack(spikes).sum()
        assert total > 0, "No spikes produced with strong input"

    def test_forward_not_all_ones(self):
        """Weak input should not produce spikes every timestep."""
        neuron = AdaptiveLIFNeuron(num_channels=1)
        neuron.reset_membrane()
        spikes = []
        for _ in range(10):
            s = neuron(torch.ones(1, 1, 4, 4) * 0.1)
            spikes.append(s)
        total = torch.stack(spikes).sum()
        max_possible = 10 * 1 * 1 * 4 * 4
        assert total < max_possible, "All spikes fired with weak input"

    def test_activity_trace_accumulates(self):
        """Activity trace a(t) should increase when neuron fires."""
        neuron = AdaptiveLIFNeuron(num_channels=1)
        neuron.reset_membrane()
        # Drive with strong input to ensure firing
        for _ in range(5):
            neuron(torch.ones(1, 1, 4, 4) * 2.0)
        assert neuron.a is not None
        assert neuron.a.sum() > 0, "Activity trace did not accumulate"

    def test_threshold_increases_with_firing(self):
        """Sustained firing should raise the effective threshold."""
        neuron = AdaptiveLIFNeuron(num_channels=1, beta=0.07)
        neuron.reset_membrane()
        initial_theta = neuron.u_th
        # Drive with strong input
        for _ in range(20):
            neuron(torch.ones(1, 1, 4, 4) * 2.0)
        effective_theta = neuron.u_th + neuron.beta * neuron.a
        assert effective_theta.max().item() > initial_theta, (
            "Threshold did not increase after sustained firing"
        )

    def test_threshold_returns_toward_baseline(self):
        """After firing stops, threshold should decay back toward u_th."""
        neuron = AdaptiveLIFNeuron(num_channels=1, beta=0.07)
        neuron.reset_membrane()
        # Drive with strong input
        for _ in range(10):
            neuron(torch.ones(1, 1, 4, 4) * 2.0)
        peak_a = neuron.a.max().item()
        # Now feed zero input (no new spikes)
        for _ in range(50):
            neuron(torch.zeros(1, 1, 4, 4))
        decayed_a = neuron.a.max().item()
        assert decayed_a < peak_a, "Activity trace did not decay during silence"

    def test_tau_a_constrained(self):
        """tau_a should stay within [tau_a_min, tau_a_max] regardless of logit value."""
        neuron = AdaptiveLIFNeuron(
            num_channels=4, tau_a_min=0.2, tau_a_max=0.4
        )
        tau = neuron.tau_a
        assert tau.min().item() >= 0.2 - 1e-6
        assert tau.max().item() <= 0.4 + 1e-6

    def test_tau_a_constrained_after_gradient(self):
        """tau_a should remain in bounds after simulated gradient updates."""
        neuron = AdaptiveLIFNeuron(
            num_channels=4, tau_a_min=0.2, tau_a_max=0.4
        )
        # Simulate large gradient updates
        with torch.no_grad():
            neuron._tau_a_logit.add_(torch.randn_like(neuron._tau_a_logit) * 10.0)
        tau = neuron.tau_a
        assert tau.min().item() >= 0.2 - 1e-6
        assert tau.max().item() <= 0.4 + 1e-6

    def test_tau_a_init_value(self):
        """tau_a should initialize close to the requested value."""
        neuron = AdaptiveLIFNeuron(
            num_channels=1, tau_a_init=0.3, tau_a_min=0.2, tau_a_max=0.4
        )
        tau = neuron.tau_a.item()
        assert abs(tau - 0.3) < 0.01, f"tau_a initialized to {tau}, expected ~0.3"

    def test_reset_clears_state(self):
        """reset_membrane should clear v, a, and s."""
        neuron = AdaptiveLIFNeuron(num_channels=1)
        neuron(torch.ones(1, 1, 4, 4) * 2.0)
        assert neuron.v is not None
        assert neuron.a is not None
        assert neuron.s is not None
        neuron.reset_membrane()
        assert neuron.v is None
        assert neuron.a is None
        assert neuron.s is None

    def test_backward_gradients_flow(self):
        """Gradients should flow through tau_a via the activity trace."""
        neuron = AdaptiveLIFNeuron(num_channels=4)
        neuron.reset_membrane()
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        total = torch.zeros(1)
        for _ in range(5):
            s = neuron(x)
            total = total + s.sum()
        total.backward()
        assert neuron._tau_a_logit.grad is not None, "No gradient on tau_a"
        assert neuron._tau_a_logit.grad.abs().sum() > 0, "tau_a gradient is zero"

    def test_output_is_binary(self):
        """Spikes should be 0 or 1."""
        neuron = AdaptiveLIFNeuron(num_channels=1)
        neuron.reset_membrane()
        for _ in range(5):
            s = neuron(torch.randn(1, 1, 4, 4))
            unique = s.unique()
            for val in unique:
                assert val.item() in (0.0, 1.0), f"Non-binary spike value: {val}"

    def test_default_hyperparameters(self):
        """Check Zhang 2023 SpikeFPN defaults."""
        neuron = AdaptiveLIFNeuron()
        assert neuron.alpha == 0.9
        assert neuron.u_th == 0.3
        assert neuron.beta == 0.07
        assert abs(neuron.tau_a.item() - 0.3) < 0.01


class TestNeuronFactory:
    """Tests for the neuron factory with adaptive_lif."""

    def test_factory_creates_adaptive_lif(self):
        neuron = create_neuron("adaptive_lif", num_channels=8)
        assert isinstance(neuron, AdaptiveLIFNeuron)

    def test_factory_passes_kwargs(self):
        neuron = create_neuron(
            "adaptive_lif", beta=0.1, tau_a_init=0.35, num_channels=4
        )
        assert neuron.beta == 0.1
        assert (neuron.tau_a - 0.35).abs().max().item() < 0.01

    def test_factory_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown neuron type"):
            create_neuron("does_not_exist")


class TestSpikingConvBlockAdaptiveLIF:
    """Integration tests: SpikingConvBlock with adaptive_lif neuron."""

    def test_snn_mode_forward(self):
        block = SpikingConvBlock(
            3, 16, kernel_size=3, neuron_type="adaptive_lif",
            neuron_kwargs={"beta": 0.07, "tau_a_init": 0.3},
        )
        block.mode = "snn"
        block.reset_membrane()
        x = torch.randn(2, 3, 8, 8)
        out = block(x)
        assert out.shape == (2, 16, 8, 8)

    def test_ann_mode_forward(self):
        block = SpikingConvBlock(
            3, 16, kernel_size=3, neuron_type="adaptive_lif",
            neuron_kwargs={"beta": 0.07, "tau_a_init": 0.3},
        )
        block.mode = "ann"
        x = torch.randn(2, 3, 8, 8)
        out = block(x)
        assert out.shape == (2, 16, 8, 8)

    def test_num_channels_auto_set(self):
        """SpikingConvBlock should auto-set num_channels for adaptive_lif."""
        block = SpikingConvBlock(3, 32, neuron_type="adaptive_lif")
        neuron = block.snn_act
        assert isinstance(neuron, AdaptiveLIFNeuron)
        assert neuron._tau_a_logit.shape[0] == 32

    def test_temporal_sequence(self):
        """Multiple timesteps through same block should work."""
        block = SpikingConvBlock(3, 16, neuron_type="adaptive_lif")
        block.mode = "snn"
        block.reset_membrane()
        for _ in range(5):
            out = block(torch.randn(1, 3, 8, 8))
        assert out.shape == (1, 16, 8, 8)

    def test_backward_through_block(self):
        """Gradients should flow through the full block."""
        block = SpikingConvBlock(3, 16, neuron_type="adaptive_lif")
        block.mode = "snn"
        block.reset_membrane()
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        total = torch.zeros(1)
        for _ in range(3):
            total = total + block(x).sum()
        total.backward()
        assert x.grad is not None
