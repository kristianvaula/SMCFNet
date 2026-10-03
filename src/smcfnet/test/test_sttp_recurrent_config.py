from smcfnet.config.config import Config


def test_recurrent_sttp_config_defaults():
    c = Config()
    assert c.model.sttp.context_window == 1
    assert c.model.sttp.recurrent_hidden_dim == 64
    assert c.model.sttp.neuron_model == "plif"
    assert c.data.stem_feature_dir is None


def test_recurrent_sttp_config_overrides():
    c = Config()
    c.model.sttp.head_design = "spiking_recurrent"
    c.model.sttp.context_window = 5
    c.data.stem_feature_dir = "/tmp/stemcache"
    assert c.model.sttp.context_window == 5
    assert c.data.stem_feature_dir == "/tmp/stemcache"
