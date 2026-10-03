
from .backbone import CSPDarknetDualStream
from .losses import IOUloss
from .pretrained import load_yolox_pretrained_dualstream
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import DualStreamYOLOX

# build.py depends on the dataset layer (h5py) — lazy-import via PEP 562
# so that model classes can be imported without h5py installed.
_BUILD_NAMES = frozenset({
	'build_model', 'build_optimizer', 'build_train_loader_cfg',
	'build_test_loader_cfg', 'build_eval_loader_cfg', 'build_lr_scheduler', 'build_dsec_prefetcher',
})

def __getattr__(name):
	if name in _BUILD_NAMES:
		from .build import (
			build_model,
			build_optimizer,
			build_train_loader_cfg,
			build_test_loader_cfg,
			build_eval_loader_cfg,
			build_lr_scheduler,
			build_dsec_prefetcher,
		)
		g = globals()
		g['build_model'] = build_model
		g['build_optimizer'] = build_optimizer
		g['build_train_loader_cfg'] = build_train_loader_cfg
		g['build_test_loader_cfg'] = build_test_loader_cfg
		g['build_eval_loader_cfg'] = build_eval_loader_cfg
		g['build_lr_scheduler'] = build_lr_scheduler
		g['build_dsec_prefetcher'] = build_dsec_prefetcher
		return g[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
	'DualStreamYOLOX',
	'IOUloss',
	'YOLOXHead',
	'YOLOPAFPN',
	'load_yolox_pretrained_dualstream',
	'build_model',
	'build_optimizer',
	'build_train_loader_cfg',
	'build_test_loader_cfg',
	'build_eval_loader_cfg',
	'build_lr_scheduler',
	'build_dsec_prefetcher',
]
