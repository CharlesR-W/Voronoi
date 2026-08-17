from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from voronoi_lab.exp1.torch_mechanics import (  # noqa: E402
    edit_activation_sites,
    summarize_resnet_mechanical_evidence,
    validate_jvp,
    zero_intervention_parity,
)


class TinyResidual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Conv2d(2, 2, 1, bias=False), torch.nn.Conv2d(2, 2, 1, bias=False)]
        )
        self.head = torch.nn.Linear(2, 3, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = x + torch.tanh(block(x))
        return self.head(x.mean(dim=(2, 3)))


def _encode(model, images, cut):
    value = images
    for block in model.blocks[: cut + 1]:
        value = value + torch.tanh(block(value))
    return value


def _suffix(model, representation, cut):
    value = representation
    for block in model.blocks[cut + 1 :]:
        value = value + torch.tanh(block(value))
    return model.head(value.mean(dim=(2, 3)))


def test_zero_intervention_parity_is_exact() -> None:
    torch.manual_seed(1)
    model = TinyResidual().double()
    images = torch.randn(4, 2, 3, 3, dtype=torch.float64)
    result = zero_intervention_parity(model, images, [0, 1], encode=_encode, suffix=_suffix)
    assert result.exact
    assert len(result.full_logits) == len(result.split_logits) == 2
    assert result.full_logits == result.split_logits


def test_site_edits_accumulate_complete_channel_vectors() -> None:
    activation = torch.zeros(1, 3, 2, 2)
    edited = edit_activation_sites(
        activation,
        batch_indices=[0, 0],
        rows=[1, 1],
        columns=[0, 0],
        native_displacements=[[1, 2, 3], [-1, 1, 2]],
    )
    assert torch.equal(activation, torch.zeros_like(activation))
    assert torch.equal(edited[0, :, 1, 0], torch.tensor([0.0, 3.0, 5.0]))


def test_jvp_agrees_with_centered_finite_difference() -> None:
    torch.manual_seed(2)
    matrix = torch.randn(5, 4, dtype=torch.float64)
    point = torch.randn(7, 4, dtype=torch.float64)
    direction = torch.randn_like(point)
    result = validate_jvp(lambda x: torch.tanh(x @ matrix.T), point, direction, epsilon=1e-5)
    assert result.median_relative_error < 1e-9
    assert result.p95_relative_error < 1e-8
    assert len(result.automatic_output) == len(result.finite_difference_output) == 35


def test_numpy_summary_reconstructs_identity_and_jvp_metrics_from_raw_arrays() -> None:
    summary = summarize_resnet_mechanical_evidence(
        {
            "first": {"full_logits": [1.0, 2.0], "split_logits": [1.0, 2.0]},
            "second": {"full_logits": [1.0, 2.0], "split_logits": [1.0, 2.5]},
        },
        {
            "second": {
                "automatic_jvp": [3.0, 4.0],
                "finite_difference_jvp": [0.0, 0.0],
            }
        },
        identity_cuts=("first", "second"),
        jvp_cuts=("first", "second"),
        denominator_floor=1e-12,
    )
    assert summary.identity_per_cut == {"first": 0.0, "second": 0.5}
    assert summary.identity_exact is False
    assert summary.identity_max_absolute_error == 0.5
    assert summary.jvp_relative_error_by_cut == {"second": 1.0}
    assert summary.jvp_cuts_completed == 1
    assert summary.jvp_median_relative_error is None
    assert summary.jvp_p95_relative_error is None


@pytest.mark.parametrize(
    "raw",
    ([True], [float("nan")], [[1.0]], []),
)
def test_numpy_summary_rejects_non_numeric_nonfinite_or_unflattened_evidence(raw) -> None:
    with pytest.raises(ValueError):
        summarize_resnet_mechanical_evidence(
            {"cut": {"full_logits": [0.0], "split_logits": raw}},
            {},
            identity_cuts=("cut",),
            jvp_cuts=("cut",),
            denominator_floor=1e-12,
        )
