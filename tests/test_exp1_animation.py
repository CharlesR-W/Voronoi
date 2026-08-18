from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageSequence

from voronoi_lab.exp1 import animation
from voronoi_lab.exp1.animation import (
    ArchitectureCellsFrame,
    RealFakeFrame,
    render_architecture_cells_animation,
    render_real_fake_animation,
)

_RESNET_RGB_MAXIMA = np.array([2.0, 3.0, 4.0])
_VGG_RGB_MAXIMA = np.array([5.0, 6.0, 7.0])


def _gif_durations(image: Image.Image) -> list[int]:
    return [int(frame.info["duration"]) for frame in ImageSequence.Iterator(image)]


def _real_fake_frames() -> list[RealFakeFrame]:
    coordinates = np.linspace(-1.0, 1.0, 11)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates, indexing="xy")
    curve_x = np.linspace(0.0, 1.0, 17)
    frames = []
    for index, checkpoint in enumerate(("epoch-000", "epoch-010", "epoch-020")):
        shift = 0.2 * index
        real = (x_grid - shift) ** 2 + 0.5 * y_grid**2
        fake = 0.4 + 0.7 * (x_grid + shift) ** 2 + y_grid**2
        frames.append(
            RealFakeFrame(
                checkpoint=checkpoint,
                real_heatmap=real,
                fake_heatmap=fake,
                heatmap_x=coordinates,
                heatmap_y=coordinates,
                curve_x=curve_x,
                real_curve=(1.0 + index) * curve_x**2,
                fake_curve=0.25 + (0.5 + index) * curve_x,
            )
        )
    return frames


def _architecture_frames() -> list[ArchitectureCellsFrame]:
    coordinates = np.linspace(-0.25, 1.25, 13)
    alpha, beta = np.meshgrid(coordinates, coordinates, indexing="xy")
    resnet_anchors = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    vgg_anchors = np.array([[0.0, 0.0], [1.0, 0.0], [0.625, 1.0]])
    frames = []
    for index, checkpoint in enumerate(("step-000", "step-100")):
        red = np.exp(-3.0 * ((alpha - 0.0) ** 2 + (beta - 0.0) ** 2))
        green = np.exp(-3.0 * ((alpha - 1.0) ** 2 + (beta - 0.0) ** 2))
        resnet_blue = np.exp(-3.0 * ((alpha - 0.5) ** 2 + (beta - 1.0) ** 2))
        vgg_blue = np.exp(-3.0 * ((alpha - 0.625) ** 2 + (beta - 1.0) ** 2))
        resnet_raw = np.stack(
            (2.0 * red, 3.0 * green, 4.0 * resnet_blue),
            axis=-1,
        )
        vgg_raw = np.stack(
            (5.0 * red, 6.0 * green, 7.0 * vgg_blue),
            axis=-1,
        )
        resnet_rgb = resnet_raw / _RESNET_RGB_MAXIMA
        vgg_rgb = vgg_raw / _VGG_RGB_MAXIMA
        jacobian = 1.0 + index + 5.0 * np.exp(-20.0 * (alpha - 0.5) ** 2)
        frames.append(
            ArchitectureCellsFrame(
                checkpoint=checkpoint,
                resnet_rgb=resnet_rgb,
                resnet_jacobian=jacobian,
                vgg_rgb=vgg_rgb,
                vgg_jacobian=0.6 * jacobian + 0.2,
                alpha_coordinates=coordinates,
                beta_coordinates=coordinates,
                resnet_anchors=resnet_anchors,
                vgg_anchors=vgg_anchors,
            )
        )
    return frames


def test_real_fake_renderer_writes_verified_gif_png_and_metadata(tmp_path) -> None:
    prefix = tmp_path / "real_fake"
    artifacts = render_real_fake_animation(
        _real_fake_frames(),
        prefix,
        canvas_size=(1200, 760),
    )

    assert artifacts.frame_count == 4
    assert artifacts.durations_ms == (3000, 1000, 1000, 6000)
    with Image.open(artifacts.gif_path) as gif:
        assert gif.n_frames == 4
        assert gif.size == (1200, 760)
        assert _gif_durations(gif) == [3000, 1000, 1000, 6000]
        assert {frame.size for frame in ImageSequence.Iterator(gif)} == {(1200, 760)}
        assert int(gif.info["loop"]) == 0
    with Image.open(artifacts.final_png_path) as final_png:
        assert final_png.size == (1200, 760)
        assert final_png.format == "PNG"

    metadata = json.loads(artifacts.metadata_path.read_text())
    assert metadata["schema_version"] == 2
    assert metadata["animation_kind"] == "real_fake_scalar_fields"
    assert metadata["input_checkpoints"] == ["epoch-000", "epoch-010", "epoch-020"]
    assert metadata["input_checkpoint_count"] == 3
    assert metadata["presentation_hold_frame_count"] == 1
    assert [frame["role"] for frame in metadata["rendered_frames"]] == [
        "orientation",
        "checkpoint",
        "checkpoint",
        "conclusion",
    ]
    assert metadata["rendered_frames"][-1]["checkpoint"] == "epoch-020"
    assert metadata["scales"]["heatmap"]["source"] == "computed_global"
    assert metadata["layout_fixed_across_frames"] is True
    assert metadata["checkpoint_panels_synchronized"] is True
    assert metadata["estimands"] == {
        "heatmap": "caller-supplied scalar field",
        "curve": "caller-supplied response profile",
    }


def test_architecture_renderer_keeps_models_synchronized_and_uses_one_scale(tmp_path) -> None:
    prefix = tmp_path / "architectures"
    input_frames = _architecture_frames()
    artifacts = render_architecture_cells_animation(
        input_frames,
        prefix,
        resnet_rgb_channel_maxima=_RESNET_RGB_MAXIMA,
        vgg_rgb_channel_maxima=_VGG_RGB_MAXIMA,
        canvas_size=(1200, 760),
    )

    assert artifacts.frame_count == 3
    assert artifacts.durations_ms == (3000, 1000, 6000)
    with Image.open(artifacts.gif_path) as gif:
        assert gif.n_frames == 3
        assert gif.size == (1200, 760)
        assert _gif_durations(gif) == [3000, 1000, 6000]
        assert {frame.size for frame in ImageSequence.Iterator(gif)} == {(1200, 760)}
    with Image.open(artifacts.final_png_path) as final_png:
        assert final_png.size == (1200, 760)

    metadata = json.loads(artifacts.metadata_path.read_text())
    assert metadata["schema_version"] == 2
    expected_minimum = min(
        float(field.min())
        for frame in input_frames
        for field in (frame.resnet_jacobian, frame.vgg_jacobian)
    )
    expected_maximum = max(
        float(field.max())
        for frame in input_frames
        for field in (frame.resnet_jacobian, frame.vgg_jacobian)
    )
    assert metadata["scales"]["jacobian"]["data_min"] == pytest.approx(expected_minimum)
    assert metadata["scales"]["jacobian"]["data_max"] == pytest.approx(expected_maximum)
    assert metadata["scales"]["rgb_cells"]["resnet_channel_maxima"] == [2.0, 3.0, 4.0]
    assert metadata["scales"]["rgb_cells"]["vgg_channel_maxima"] == [5.0, 6.0, 7.0]
    assert metadata["scales"]["rgb_cells"]["comparability_contract"] == (
        "per_architecture_channel_normalization_not_absolute_cross_architecture_scale"
    )
    assert metadata["rendering"]["engine"] == "Pillow"
    assert metadata["rendering"]["font_regular"]["sha256"]
    assert metadata["labels"]["jacobian"] == "2D ‖·‖F"
    assert metadata["labels"]["rgb"] == "SOURCE ANALOGUE · THREE-ANCHOR RGB"
    assert metadata["labels"]["resnet_jacobian"] == "NEW HYBRID · D(T-I)"
    assert metadata["labels"]["vgg_jacobian"] == "NEW HYBRID · DT"
    assert metadata["jacobian_estimands"] == {
        "resnet": "2D-plane-restricted ||D(T-I)||_F",
        "vgg": "2D-plane-restricted ||DT||_F",
    }
    assert "not a causal architecture ablation" in metadata["comparison_note"]
    assert metadata["rendered_frames"] == [
        {
            "checkpoint": "step-000",
            "duration_ms": 3000,
            "index": 0,
            "role": "orientation",
        },
        {
            "checkpoint": "step-100",
            "duration_ms": 1000,
            "index": 1,
            "role": "checkpoint",
        },
        {
            "checkpoint": "step-100",
            "duration_ms": 6000,
            "index": 2,
            "role": "conclusion",
        },
    ]


def test_renderer_rejects_unsynchronized_checkpoint_geometry(tmp_path) -> None:
    frames = _real_fake_frames()
    mismatched = RealFakeFrame(
        checkpoint="epoch-030",
        real_heatmap=np.zeros((9, 9)),
        fake_heatmap=np.zeros((9, 9)),
        heatmap_x=np.linspace(-1.0, 1.0, 9),
        heatmap_y=np.linspace(-1.0, 1.0, 9),
        curve_x=frames[0].curve_x,
        real_curve=frames[0].real_curve,
        fake_curve=frames[0].fake_curve,
    )
    with pytest.raises(ValueError, match="one shape"):
        render_real_fake_animation([frames[0], mismatched], tmp_path / "invalid")

    duplicate = RealFakeFrame(
        checkpoint=frames[0].checkpoint,
        real_heatmap=frames[1].real_heatmap,
        fake_heatmap=frames[1].fake_heatmap,
        heatmap_x=frames[1].heatmap_x,
        heatmap_y=frames[1].heatmap_y,
        curve_x=frames[1].curve_x,
        real_curve=frames[1].real_curve,
        fake_curve=frames[1].fake_curve,
    )
    with pytest.raises(ValueError, match="unique"):
        render_real_fake_animation([frames[0], duplicate], tmp_path / "duplicate")


def test_renderer_rejects_ambiguous_coordinates_and_invalid_jacobians(tmp_path) -> None:
    frames = _real_fake_frames()
    shifted = RealFakeFrame(
        checkpoint="epoch-030",
        real_heatmap=frames[1].real_heatmap,
        fake_heatmap=frames[1].fake_heatmap,
        heatmap_x=frames[1].heatmap_x + 0.1,
        heatmap_y=frames[1].heatmap_y,
        curve_x=frames[1].curve_x,
        real_curve=frames[1].real_curve,
        fake_curve=frames[1].fake_curve,
    )
    with pytest.raises(ValueError, match="identical heatmap coordinates"):
        render_real_fake_animation([frames[0], shifted], tmp_path / "shifted")

    architecture = _architecture_frames()[0]
    with pytest.raises(ValueError, match="nonnegative"):
        ArchitectureCellsFrame(
            checkpoint="bad-jacobian",
            resnet_rgb=architecture.resnet_rgb,
            resnet_jacobian=-np.ones_like(architecture.resnet_jacobian),
            vgg_rgb=architecture.vgg_rgb,
            vgg_jacobian=architecture.vgg_jacobian,
            alpha_coordinates=architecture.alpha_coordinates,
            beta_coordinates=architecture.beta_coordinates,
        )


def test_grid_validation_accepts_float32_linspace_quantization() -> None:
    coordinates = np.linspace(-0.25, 1.25, 21, dtype=np.float32)
    rgb = np.zeros((21, 21, 3), dtype=np.float32)
    jacobian = np.ones((21, 21), dtype=np.float32)
    frame = ArchitectureCellsFrame(
        checkpoint="epoch-100",
        resnet_rgb=rgb,
        resnet_jacobian=jacobian,
        vgg_rgb=rgb,
        vgg_jacobian=jacobian,
        alpha_coordinates=coordinates,
        beta_coordinates=coordinates,
    )
    np.testing.assert_array_equal(frame.alpha_coordinates, coordinates.astype(np.float64))

    nonuniform = coordinates.copy()
    nonuniform[10] += np.float32(1.0e-3)
    with pytest.raises(ValueError, match="uniformly spaced"):
        ArchitectureCellsFrame(
            checkpoint="bad-grid",
            resnet_rgb=rgb,
            resnet_jacobian=jacobian,
            vgg_rgb=rgb,
            vgg_jacobian=jacobian,
            alpha_coordinates=nonuniform,
            beta_coordinates=coordinates,
        )


def test_renderer_rejects_layout_that_cannot_keep_labels_readable(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 1200 by 760"):
        render_real_fake_animation(
            _real_fake_frames(),
            tmp_path / "too-small",
            canvas_size=(1199, 760),
        )
    with pytest.raises(ValueError, match="at most 72 characters"):
        render_real_fake_animation(
            _real_fake_frames(),
            tmp_path / "long-title",
            title="x" * 73,
            canvas_size=(1200, 760),
        )
    with pytest.raises(ValueError, match="scalar field title does not fit its panel"):
        render_architecture_cells_animation(
            _architecture_frames(),
            tmp_path / "wide-scalar-label",
            resnet_rgb_channel_maxima=_RESNET_RGB_MAXIMA,
            vgg_rgb_channel_maxima=_VGG_RGB_MAXIMA,
            resnet_jacobian_label="W" * 30,
            canvas_size=(1200, 760),
        )
    canvas = Image.new("RGB", (1200, 760))
    draw = animation.ImageDraw.Draw(canvas)
    with pytest.raises(ValueError, match="tick labels overlap the lower title"):
        animation._validate_stacked_field_vertical_clearance(
            draw,
            animation._fonts((1200, 760)),
            upper_rect=(0, 0, 100, 100),
            lower_rect=(0, 110, 100, 210),
            lower_title="LOWER TITLE",
        )


def test_grid_samples_align_with_declared_marker_coordinates() -> None:
    values = np.zeros((5, 7))
    values[1, 4] = 1.0
    rect = (10, 20, 70, 60)
    extent = (-3.0, 3.0, -2.0, 2.0)
    sampled = animation._bilinear_sample_grid(
        values,
        (rect[2] - rect[0] + 1, rect[3] - rect[1] + 1),
    )
    raster_row, raster_column = np.unravel_index(np.argmax(sampled), sampled.shape)
    rendered_peak = (rect[0] + int(raster_column), rect[1] + int(raster_row))
    declared_marker = animation._data_to_pixel(1.0, -1.0, rect, extent)

    assert rendered_peak == declared_marker


def test_renderer_is_byte_deterministic_for_identical_inputs(tmp_path) -> None:
    prefix = tmp_path / "repeat"
    first = render_real_fake_animation(
        _real_fake_frames(),
        prefix,
        canvas_size=(1200, 760),
    )
    first_gif = first.gif_path.read_bytes()
    first_png = first.final_png_path.read_bytes()
    first_metadata = first.metadata_path.read_bytes()

    second = render_real_fake_animation(
        _real_fake_frames(),
        prefix,
        canvas_size=(1200, 760),
    )
    assert second.gif_path.read_bytes() == first_gif
    assert second.final_png_path.read_bytes() == first_png
    assert second.metadata_path.read_bytes() == first_metadata
