# Copyright 2026 OpenSearch contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Sparse supervised-token lm_head path for Qwen3-VL-MoE training."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel


IGNORE_INDEX = -100
logger = logging.get_logger(__name__)
_SPARSE_LM_HEAD_LOSS_ONLY = ContextVar("opensearch_sparse_lm_head_loss_only", default=False)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no", "off"}


@contextmanager
def sparse_lm_head_loss_only(enabled: bool = True) -> Generator[None, None, None]:
    """Allow sparse logits only while the caller promises not to consume model outputs."""
    token = _SPARSE_LM_HEAD_LOSS_ONLY.set(enabled)
    try:
        yield
    finally:
        _SPARSE_LM_HEAD_LOSS_ONLY.reset(token)


def sparse_causal_lm_loss(
    logits: torch.Tensor,
    shift_labels: torch.Tensor,
    *,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute causal CE on already-selected valid target positions."""
    if logits.ndim != 2:
        raise ValueError(f"Sparse lm_head expects 2-D logits, got {tuple(logits.shape)}.")

    target = shift_labels.reshape(-1).to(device=logits.device, dtype=torch.long)
    if target.numel() != logits.size(0):
        raise ValueError(
            f"Sparse lm_head target/logits mismatch: {target.numel()} targets for {logits.size(0)} rows."
        )
    if target.numel() == 0:
        raise ValueError("Sparse lm_head received empty targets.")

    # Match Transformers' ForCausalLMLoss: upcast only the selected logits.
    loss = F.cross_entropy(
        logits.float(),
        target,
        ignore_index=ignore_index,
        reduction="sum" if num_items_in_batch is not None else "mean",
    )
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(device=loss.device, dtype=loss.dtype)
        loss = loss / num_items_in_batch
    return loss


def _select_sparse_lm_inputs(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Select hidden states and shifted labels used by causal cross entropy."""
    if labels.ndim != 2 or hidden_states.ndim != 3:
        return None
    if hidden_states.size(0) != labels.size(0) or hidden_states.size(1) != labels.size(1):
        return None

    # Logit at position t predicts label t+1. The final hidden state is never
    # used by the causal loss, so it is omitted before applying the mask.
    shift_labels = labels[..., 1:]
    valid_mask = shift_labels.ne(ignore_index)
    selected_hidden = hidden_states[..., :-1, :][valid_mask]
    selected_labels = shift_labels[valid_mask]
    # numel() is shape metadata and does not force a device scalar back to the
    # host, unlike bool(valid_mask.any()) on MUSA.
    if selected_labels.numel() == 0:
        return None

    return selected_hidden, selected_labels


def patch_qwen3_vl_moe_sparse_lm_head(model: "PreTrainedModel") -> bool:
    """Patch Qwen3-VL-MoE training lm_head to skip ignored SFT tokens."""
    if not _env_flag("OPENSEARCH_MUSA_SPARSE_LM_HEAD", "0"):
        return False
    if getattr(model.config, "model_type", None) != "qwen3_vl_moe":
        return False
    if getattr(model, "_opensearch_sparse_lm_head_patched", False):
        return True

    output_layer = getattr(model, "lm_head", None)
    if output_layer is None or not callable(getattr(output_layer, "forward", None)):
        logger.warning_rank0_once("Sparse Qwen3-VL-MoE lm_head skipped: no callable lm_head.")
        return False

    original_model_forward = model.forward
    original_lm_head_forward = output_layer.forward
    original_loss_function = model.loss_function
    if getattr(original_loss_function, "__name__", None) != "ForCausalLMLoss":
        logger.warning_rank0_once(
            "Sparse Qwen3-VL-MoE lm_head skipped: the configured loss is not ForCausalLMLoss."
        )
        return False

    def sparse_lm_head_forward(head: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        labels = getattr(model, "_opensearch_sparse_lm_labels", None)
        if labels is None:
            return original_lm_head_forward(hidden_states)

        selected = _select_sparse_lm_inputs(hidden_states, labels)
        if selected is None:
            return original_lm_head_forward(hidden_states)

        selected_hidden, selected_labels = selected
        model._opensearch_sparse_lm_shift_labels = selected_labels
        return original_lm_head_forward(selected_hidden)

    output_layer.forward = MethodType(sparse_lm_head_forward, output_layer)

    def sparse_loss_function(
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        num_items_in_batch: torch.Tensor | int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        shift_labels = getattr(model, "_opensearch_sparse_lm_shift_labels", None)
        if shift_labels is None or logits.ndim != 2 or logits.size(0) != shift_labels.numel():
            return original_loss_function(
                logits=logits,
                labels=labels,
                vocab_size=vocab_size,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )

        return sparse_causal_lm_loss(
            logits,
            shift_labels,
            num_items_in_batch=num_items_in_batch,
        )

    model._loss_function = sparse_loss_function

    def sparse_model_forward(self, *args: Any, **kwargs: Any) -> Any:
        labels = kwargs.get("labels")
        logits_to_keep = kwargs.get("logits_to_keep", 0)
        active = (
            _SPARSE_LM_HEAD_LOSS_ONLY.get()
            and self.training
            and isinstance(labels, torch.Tensor)
            and isinstance(logits_to_keep, int)
            and logits_to_keep == 0
        )
        if not active:
            return original_model_forward(*args, **kwargs)

        previous_labels = getattr(self, "_opensearch_sparse_lm_labels", None)
        previous_shift_labels = getattr(self, "_opensearch_sparse_lm_shift_labels", None)
        self._opensearch_sparse_lm_labels = labels
        self._opensearch_sparse_lm_shift_labels = None
        try:
            return original_model_forward(*args, **kwargs)
        finally:
            self._opensearch_sparse_lm_labels = previous_labels
            self._opensearch_sparse_lm_shift_labels = previous_shift_labels

    model.forward = MethodType(sparse_model_forward, model)
    model._opensearch_sparse_lm_head_patched = True
    logger.info_rank0("Enabled sparse supervised-token lm_head/loss for Qwen3-VL-MoE.")
    return True


__all__ = [
    "patch_qwen3_vl_moe_sparse_lm_head",
    "sparse_causal_lm_loss",
    "sparse_lm_head_loss_only",
]
