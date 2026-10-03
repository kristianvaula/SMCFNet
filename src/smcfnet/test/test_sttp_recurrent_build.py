from smcfnet.config.config import Config
from smcfnet.model.build import build_model
from smcfnet.model.sttp.sttp_head import SpikingRecurrentSTTPHead, ConvGRUSTTPHead


def _cfg(head_design):
    c = Config()
    c.model.sttp.enabled = True
    c.model.sttp.head_design = head_design
    c.model.sttp.context_window = 5
    c.model.sttp.recurrent_hidden_dim = 16
    return c


def test_build_spiking_recurrent_head():
    model = build_model(_cfg("spiking_recurrent"))
    assert isinstance(model.sttp_head, SpikingRecurrentSTTPHead)
    assert model.sttp_head.num_scales == 3


def test_build_convgru_head():
    model = build_model(_cfg("convgru"))
    assert isinstance(model.sttp_head, ConvGRUSTTPHead)
