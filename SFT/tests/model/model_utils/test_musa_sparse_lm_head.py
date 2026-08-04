# Copyright 2026 OpenSearch contributors.

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from llamafactory.model.model_utils.musa_sparse_lm_head import (
    _select_sparse_lm_inputs,
    patch_qwen3_vl_moe_sparse_lm_head,
    sparse_causal_lm_loss,
    sparse_lm_head_loss_only,
)


def ForCausalLMLoss(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100, **kwargs):
    logits = logits.float().view(-1, vocab_size)
    shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous().view(-1)
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.cross_entropy(logits, shift_labels, ignore_index=ignore_index, reduction=reduction)
    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch
    return loss


class FakeQwen3VLMoe(torch.nn.Module):
    def __init__(self, hidden_size: int = 5, vocab_size: int = 11) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_vl_moe", text_config=SimpleNamespace(vocab_size=vocab_size))
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self._loss_function = ForCausalLMLoss

    @property
    def loss_function(self):
        return self._loss_function

    def forward(self, hidden_states, labels=None, logits_to_keep=0):
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size)
        return {"loss": loss, "logits": logits}


def test_sparse_causal_lm_loss_matches_dense_loss_and_gradient():
    torch.manual_seed(0)
    hidden_dense = torch.randn(2, 7, 5, requires_grad=True)
    weight_dense = torch.randn(11, 5, requires_grad=True)
    labels = torch.tensor(
        [
            [-100, -100, 3, 4, -100, 6, 7],
            [-100, 2, -100, -100, 5, 8, -100],
        ],
        dtype=torch.long,
    )

    dense_logits = F.linear(hidden_dense, weight_dense)
    dense_loss = F.cross_entropy(
        dense_logits[:, :-1].reshape(-1, dense_logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    selected_hidden, selected_labels = _select_sparse_lm_inputs(hidden_dense, labels)
    sparse_logits = F.linear(selected_hidden, weight_dense)
    sparse_loss = sparse_causal_lm_loss(sparse_logits, selected_labels)

    dense_loss.backward()
    dense_hidden_grad = hidden_dense.grad.detach().clone()
    dense_weight_grad = weight_dense.grad.detach().clone()

    hidden_sparse = hidden_dense.detach().clone().requires_grad_(True)
    weight_sparse = weight_dense.detach().clone().requires_grad_(True)
    selected_hidden, selected_labels = _select_sparse_lm_inputs(hidden_sparse, labels)
    sparse_loss = sparse_causal_lm_loss(F.linear(selected_hidden, weight_sparse), selected_labels)
    sparse_loss.backward()

    torch.testing.assert_close(sparse_loss, dense_loss.detach(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(hidden_sparse.grad, dense_hidden_grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(weight_sparse.grad, dense_weight_grad, atol=1e-6, rtol=1e-5)


def test_sparse_selection_returns_none_for_all_ignored_labels():
    hidden_states = torch.randn(1, 4, 3)
    labels = torch.full((1, 4), -100, dtype=torch.long)
    assert _select_sparse_lm_inputs(hidden_states, labels) is None


def test_sparse_causal_lm_loss_accepts_integer_num_items_in_batch():
    logits = torch.randn(3, 7)
    labels = torch.tensor([1, 2, 3])
    actual = sparse_causal_lm_loss(logits, labels, num_items_in_batch=3)
    expected = F.cross_entropy(logits.float(), labels, reduction="sum") / 3
    torch.testing.assert_close(actual, expected)


def test_sparse_patch_preserves_dense_outputs_outside_loss_only_context(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_MUSA_SPARSE_LM_HEAD", "1")
    torch.manual_seed(1)
    model = FakeQwen3VLMoe().train()
    hidden_states = torch.randn(2, 6, 5)
    labels = torch.tensor(
        [
            [-100, -100, 1, 2, -100, 3],
            [-100, 4, -100, 5, 6, -100],
        ]
    )

    assert patch_qwen3_vl_moe_sparse_lm_head(model)
    dense_outputs = model(hidden_states, labels=labels)
    assert dense_outputs["logits"].shape == (2, 6, 11)

    with sparse_lm_head_loss_only():
        sparse_outputs = model(hidden_states, labels=labels)

    supervised_tokens = labels[..., 1:].ne(-100).sum().item()
    assert sparse_outputs["logits"].shape == (supervised_tokens, 11)
    torch.testing.assert_close(sparse_outputs["loss"], dense_outputs["loss"])

    model.eval()
    with sparse_lm_head_loss_only():
        eval_outputs = model(hidden_states, labels=labels)
    assert eval_outputs["logits"].shape == (2, 6, 11)


def test_sparse_patch_skips_custom_loss(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_MUSA_SPARSE_LM_HEAD", "1")
    model = FakeQwen3VLMoe()

    def custom_loss(logits, labels, vocab_size, **kwargs):
        return logits.sum() * 0

    model._loss_function = custom_loss
    assert not patch_qwen3_vl_moe_sparse_lm_head(model)
