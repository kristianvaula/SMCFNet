import torch
import pytest


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_cuadrado_unet_skips_decoder_output_size_mutation_for_static_shape(monkeypatch):
    pytest.importorskip("spikingjelly")

    from smcfnet.model.ofe.cuadrado_unet import CuadradoUNet

    model = CuadradoUNet(
        num_bins=21,
        mode="snn",
        image_height=32,
        image_width=32,
        kernel_size=7,
    )
    model.eval()

    calls = {}
    for name in ("deconv4", "deconv3", "deconv2", "deconv1", "pred1"):
        module = getattr(model, name)
        calls[name] = 0
        original = module.set_output_size

        def wrapped(size, *, _name=name, _original=original):
            calls[_name] += 1
            return _original(size)

        monkeypatch.setattr(module, "set_output_size", wrapped)

    x = torch.randn(1, 2, 21, 32, 32)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)

    assert tuple(y1.shape) == (1, 2, 32, 32)
    assert tuple(y2.shape) == (1, 2, 32, 32)
    assert calls == {
        "deconv4": 0,
        "deconv3": 0,
        "deconv2": 0,
        "deconv1": 0,
        "pred1": 0,
    }


def test_cuadrado_unet_flow_core_does_not_cache_last_flow():
    pytest.importorskip("spikingjelly")

    from smcfnet.model.ofe.cuadrado_unet import CuadradoUNet

    model = CuadradoUNet(
        num_bins=21,
        mode="snn",
        image_height=32,
        image_width=32,
        kernel_size=7,
    )
    model.eval()

    x = torch.randn(1, 2, 21, 32, 32)
    with torch.no_grad():
        flow = model._forward_flow(x)
    assert tuple(flow.shape) == (1, 2, 32, 32)
    assert not hasattr(model, "_last_flow")

    model.reset_state()
    with torch.no_grad():
        flow = model(x)
    assert tuple(flow.shape) == (1, 2, 32, 32)
    assert model._last_flow is not None


def test_cuadrado_unet_compile_for_inference_compiles_flow_core(monkeypatch):
    pytest.importorskip("spikingjelly")

    from smcfnet.model.ofe.cuadrado_unet import CuadradoUNet

    compile_calls = []

    def fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)

        def wrapped(x):
            return fn(x)

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)

    model = CuadradoUNet(
        num_bins=21,
        mode="snn",
        image_height=32,
        image_width=32,
        kernel_size=7,
    )
    model.compile_for_inference(mode="reduce-overhead", fullgraph=False, dynamic=None)

    x = torch.randn(1, 2, 21, 32, 32)
    with torch.no_grad():
        flow = model(x)

    assert tuple(flow.shape) == (1, 2, 32, 32)
    assert compile_calls == [{"mode": "reduce-overhead", "fullgraph": False}]
    assert model._last_flow is not None
    assert all(node.compile_friendly_snn for node in model._if_nodes)


def test_compile_friendly_snn_path_matches_spikingjelly_inference():
    pytest.importorskip("spikingjelly")

    from smcfnet.model.ofe.cuadrado_unet import CuadradoUNet

    reference = CuadradoUNet(
        num_bins=21,
        mode="snn",
        image_height=32,
        image_width=32,
        kernel_size=7,
    )
    optimized = CuadradoUNet(
        num_bins=21,
        mode="snn",
        image_height=32,
        image_width=32,
        kernel_size=7,
    )
    optimized.load_state_dict(reference.state_dict())
    optimized.set_compile_friendly_snn(True)

    reference.eval()
    optimized.eval()
    x = torch.randn(1, 2, 21, 32, 32)

    with torch.no_grad():
        y_reference = reference._forward_flow(x)
        y_optimized = optimized._forward_flow(x)

    assert torch.equal(y_optimized, y_reference)


def test_compile_friendly_snn_state_does_not_alias_input_tensor():
    pytest.importorskip("spikingjelly")

    from smcfnet.model.ofe.cuadrado_unet import SwitchableIFNode

    node = SwitchableIFNode(mode="snn", v_threshold=float("inf"), v_reset=0.0)
    node.compile_friendly_snn = True

    x = torch.randn(2, 3)
    with torch.no_grad():
        y = node(x)

    assert torch.equal(y, torch.zeros_like(x))
    assert node.v is not x
    assert node.v.data_ptr() != x.data_ptr()

    x.add_(1.0)
    assert not torch.equal(node.v, x)


def test_cuadrado_unet_marks_cudagraph_step_begin_for_compiled_forward(monkeypatch):
    pytest.importorskip("spikingjelly")

    from smcfnet.model.ofe.cuadrado_unet import CuadradoUNet

    compile_calls = []
    mark_calls = []

    def fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)

        def wrapped(x):
            return fn(x)

        return wrapped

    class FakeCompiler:
        @staticmethod
        def cudagraph_mark_step_begin():
            mark_calls.append(True)

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch, "compiler", FakeCompiler(), raising=False)

    model = CuadradoUNet(
        num_bins=21,
        mode="snn",
        image_height=32,
        image_width=32,
        kernel_size=7,
    )
    model.compile_for_inference(mode="reduce-overhead", fullgraph=False, dynamic=None)

    x = torch.randn(1, 2, 21, 32, 32)
    with torch.no_grad():
        _ = model(x)

    assert compile_calls == [{"mode": "reduce-overhead", "fullgraph": False}]
    assert mark_calls == [True]


@pytest.mark.parametrize(
    "class_name",
    ["BottomSeparableCuadradoUNet", "FactorizedCuadradoUNet"],
)
def test_cuadrado_unet_experimental_variants_preserve_flow_shape(class_name):
    pytest.importorskip("spikingjelly")

    import smcfnet.model.ofe.cuadrado_unet as cuadrado_unet

    cls = getattr(cuadrado_unet, class_name)
    model = cls(
        num_bins=21,
        mode="snn",
        image_height=32,
        image_width=32,
        kernel_size=7,
    )
    model.eval()

    x = torch.randn(1, 2, 21, 32, 32)
    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (1, 2, 32, 32)
