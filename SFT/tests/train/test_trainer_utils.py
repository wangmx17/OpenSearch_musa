# Copyright 2026 OpenSearch contributors.

from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers.training_args import OptimizerNames

from llamafactory.train import trainer_utils


def _training_args(optim):
    return SimpleNamespace(
        optim=optim,
        weight_decay=0.01,
        learning_rate=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    )


def _finetuning_args(**overrides):
    values = {
        "use_galore": False,
        "use_apollo": False,
        "loraplus_lr_ratio": None,
        "use_badam": False,
        "use_adam_mini": False,
        "use_muon": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _install_fake_fused_adamw(monkeypatch: pytest.MonkeyPatch):
    class FakeFusedAdamW:
        def __init__(self, parameter_groups, **kwargs):
            self.parameter_groups = parameter_groups
            self.kwargs = kwargs

    optim_module = ModuleType("torch_musa.optim")
    optim_module.FusedAdamW = FakeFusedAdamW
    musa_module = ModuleType("torch_musa")
    musa_module.__path__ = []
    musa_module.optim = optim_module
    monkeypatch.setitem(__import__("sys").modules, "torch_musa", musa_module)
    monkeypatch.setitem(__import__("sys").modules, "torch_musa.optim", optim_module)
    return FakeFusedAdamW


@pytest.mark.parametrize("optim", [OptimizerNames.ADAMW_TORCH, "adamw_torch"])
def test_fused_adamw_accepts_optimizer_enum_and_string(monkeypatch: pytest.MonkeyPatch, optim) -> None:
    fused_adamw = _install_fake_fused_adamw(monkeypatch)
    monkeypatch.setenv("OPENSEARCH_USE_MUSA_FUSED_ADAMW", "1")

    optimizer = trainer_utils.create_custom_optimizer(
        torch.nn.Linear(2, 2), _training_args(optim), _finetuning_args()
    )

    assert isinstance(optimizer, fused_adamw)


@pytest.mark.parametrize(
    ("flag", "factory"),
    [
        ("use_galore", "_create_galore_optimizer"),
        ("use_apollo", "_create_apollo_optimizer"),
        ("loraplus_lr_ratio", "_create_loraplus_optimizer"),
        ("use_badam", "_create_badam_optimizer"),
        ("use_adam_mini", "_create_adam_mini_optimizer"),
        ("use_muon", "_create_muon_optimizer"),
    ],
)
def test_fused_adamw_does_not_override_alternative_optimizer(
    monkeypatch: pytest.MonkeyPatch, flag: str, factory: str
) -> None:
    monkeypatch.setenv("OPENSEARCH_USE_MUSA_FUSED_ADAMW", "1")
    sentinel = object()
    value = 1.0 if flag == "loraplus_lr_ratio" else True
    monkeypatch.setattr(trainer_utils, factory, lambda *args, **kwargs: sentinel)

    finetuning_args = _finetuning_args(**{flag: value})
    optimizer = trainer_utils.create_custom_optimizer(object(), _training_args("adamw_torch"), finetuning_args)

    assert optimizer is sentinel


def test_fused_adamw_import_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fail_torch_musa_import(name, *args, **kwargs):
        if name == "torch_musa.optim":
            raise ImportError("injected FusedAdamW import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_torch_musa_import)
    monkeypatch.setenv("OPENSEARCH_USE_MUSA_FUSED_ADAMW", "1")

    assert trainer_utils.create_custom_optimizer(object(), _training_args(OptimizerNames.ADAMW_TORCH), _finetuning_args()) is None
