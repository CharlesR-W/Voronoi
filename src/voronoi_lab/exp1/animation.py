"""Deterministic Pillow renderers for Experiment 1 checkpoint animations.

The renderers consume already-prepared NumPy fields.  They deliberately do not
import model or training code: a caller supplies synchronized checkpoint frames,
and this module turns those immutable data products into a GIF, a final PNG, and
JSON metadata describing the exact timing and global display scales.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from PIL import __version__ as PILLOW_VERSION

FloatArray = NDArray[np.float64]

_BACKGROUND = "#F4F1EA"
_PANEL_BACKGROUND = "#FFFFFF"
_INK = "#172033"
_MUTED = "#5D6678"
_GRID = "#D8DDE7"
_REAL = "#176B87"
_FAKE = "#D97706"
_RESNET = "#176B87"
_VGG = "#8B5CF6"
_BLUE = "#2563EB"
_RED = "#DC2626"
_GREEN = "#16A34A"


def _finite_array(values: ArrayLike, *, name: str, ndim: int | None = None) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    array.setflags(write=False)
    return array


def _scalar_field(values: ArrayLike, *, name: str) -> FloatArray:
    field = _finite_array(values, name=name, ndim=2)
    if min(field.shape) < 2:
        raise ValueError(f"{name} must be at least 2 by 2")
    return field


def _nonnegative_scalar_field(values: ArrayLike, *, name: str) -> FloatArray:
    field = _scalar_field(values, name=name)
    if float(field.min()) < -1e-12:
        raise ValueError(f"{name} must be nonnegative")
    if float(field.min()) < 0.0:
        field = np.maximum(field, 0.0)
        field.setflags(write=False)
    return field


def _grid_coordinates(values: ArrayLike, *, name: str) -> FloatArray:
    source = np.asarray(values)
    coordinates = _curve(values, name=name)
    differences = np.diff(coordinates)
    if np.any(differences <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    ideal = np.linspace(coordinates[0], coordinates[-1], len(coordinates), dtype=np.float64)
    precision = (
        float(np.finfo(source.dtype).eps)
        if np.issubdtype(source.dtype, np.floating)
        else float(np.finfo(np.float64).eps)
    )
    quantization_tolerance = (
        4.0
        * precision
        * max(
            1.0,
            float(np.max(np.abs(coordinates))),
        )
    )
    if not np.allclose(
        coordinates,
        ideal,
        rtol=0.0,
        atol=quantization_tolerance,
    ):
        raise ValueError(f"{name} must be uniformly spaced for raster rendering")
    return coordinates


def _curve(values: ArrayLike, *, name: str) -> FloatArray:
    curve = _finite_array(values, name=name, ndim=1)
    if len(curve) < 2:
        raise ValueError(f"{name} must contain at least two values")
    return curve


def _rgb_field(values: ArrayLike, *, name: str) -> FloatArray:
    source = np.asarray(values)
    if source.ndim != 3 or source.shape[-1] != 3 or min(source.shape[:2]) < 2:
        raise ValueError(f"{name} must have shape (height, width, 3)")
    if source.size == 0 or not np.all(np.isfinite(source)):
        raise ValueError(f"{name} must be nonempty and finite")
    if np.issubdtype(source.dtype, np.integer):
        if int(source.min()) < 0 or int(source.max()) > 255:
            raise ValueError(f"integer {name} values must lie in [0, 255]")
        array = np.array(source, dtype=np.float64, copy=True) / 255.0
    else:
        array = np.array(source, dtype=np.float64, copy=True)
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError(f"floating-point {name} values must lie in [0, 1]")
    array.setflags(write=False)
    return array


def _checkpoint_label(value: object) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError("checkpoint labels must be nonempty")
    return label


def _label(value: str, *, name: str, max_characters: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    label = value.strip()
    if not label:
        raise ValueError(f"{name} must be nonempty")
    if len(label) > max_characters:
        raise ValueError(
            f"{name} must contain at most {max_characters} characters for this fixed layout"
        )
    return label


@dataclass(frozen=True, slots=True)
class RealFakeFrame:
    """Prepared scalar fields and profiles for one synchronized checkpoint."""

    checkpoint: str
    real_heatmap: FloatArray
    fake_heatmap: FloatArray
    heatmap_x: FloatArray
    heatmap_y: FloatArray
    curve_x: FloatArray
    real_curve: FloatArray
    fake_curve: FloatArray

    def __post_init__(self) -> None:
        checkpoint = _checkpoint_label(self.checkpoint)
        real_heatmap = _scalar_field(self.real_heatmap, name="real_heatmap")
        fake_heatmap = _scalar_field(self.fake_heatmap, name="fake_heatmap")
        if real_heatmap.shape != fake_heatmap.shape:
            raise ValueError("real and fake heatmaps must have the same shape")
        heatmap_x = _grid_coordinates(self.heatmap_x, name="heatmap_x")
        heatmap_y = _grid_coordinates(self.heatmap_y, name="heatmap_y")
        if real_heatmap.shape != (len(heatmap_y), len(heatmap_x)):
            raise ValueError("heatmaps must have shape (len(heatmap_y), len(heatmap_x))")
        curve_x = _curve(self.curve_x, name="curve_x")
        real_curve = _curve(self.real_curve, name="real_curve")
        fake_curve = _curve(self.fake_curve, name="fake_curve")
        if curve_x.shape != real_curve.shape or curve_x.shape != fake_curve.shape:
            raise ValueError("curve coordinates and real/fake curves must have matching shapes")
        if np.any(np.diff(curve_x) <= 0):
            raise ValueError("curve_x must be strictly increasing")
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(self, "real_heatmap", real_heatmap)
        object.__setattr__(self, "fake_heatmap", fake_heatmap)
        object.__setattr__(self, "heatmap_x", heatmap_x)
        object.__setattr__(self, "heatmap_y", heatmap_y)
        object.__setattr__(self, "curve_x", curve_x)
        object.__setattr__(self, "real_curve", real_curve)
        object.__setattr__(self, "fake_curve", fake_curve)


@dataclass(frozen=True, slots=True)
class ArchitectureCellsFrame:
    """Prepared ResNet/VGG cell colors and Jacobian fields at one checkpoint."""

    checkpoint: str
    resnet_rgb: FloatArray
    resnet_jacobian: FloatArray
    vgg_rgb: FloatArray
    vgg_jacobian: FloatArray
    alpha_coordinates: FloatArray
    beta_coordinates: FloatArray
    resnet_anchors: FloatArray | None = None
    vgg_anchors: FloatArray | None = None

    def __post_init__(self) -> None:
        checkpoint = _checkpoint_label(self.checkpoint)
        resnet_rgb = _rgb_field(self.resnet_rgb, name="resnet_rgb")
        vgg_rgb = _rgb_field(self.vgg_rgb, name="vgg_rgb")
        resnet_jacobian = _nonnegative_scalar_field(self.resnet_jacobian, name="resnet_jacobian")
        vgg_jacobian = _nonnegative_scalar_field(self.vgg_jacobian, name="vgg_jacobian")
        alpha_coordinates = _grid_coordinates(self.alpha_coordinates, name="alpha_coordinates")
        beta_coordinates = _grid_coordinates(self.beta_coordinates, name="beta_coordinates")
        shapes = {
            resnet_rgb.shape[:2],
            vgg_rgb.shape[:2],
            resnet_jacobian.shape,
            vgg_jacobian.shape,
        }
        if len(shapes) != 1:
            raise ValueError("all architecture cell and Jacobian grids must have one shape")
        if resnet_jacobian.shape != (len(beta_coordinates), len(alpha_coordinates)):
            raise ValueError(
                "architecture grids must have shape (len(beta_coordinates), len(alpha_coordinates))"
            )
        anchors: list[FloatArray | None] = []
        for name, values in (
            ("resnet_anchors", self.resnet_anchors),
            ("vgg_anchors", self.vgg_anchors),
        ):
            if values is None:
                anchors.append(None)
                continue
            coordinates = _finite_array(values, name=name, ndim=2)
            if coordinates.shape != (3, 2):
                raise ValueError(f"{name} must have shape (3, 2) in alpha/beta coordinates")
            anchors.append(coordinates)
        if (anchors[0] is None) != (anchors[1] is None):
            raise ValueError("ResNet and VGG anchors must either both be present or both be absent")
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(self, "resnet_rgb", resnet_rgb)
        object.__setattr__(self, "resnet_jacobian", resnet_jacobian)
        object.__setattr__(self, "vgg_rgb", vgg_rgb)
        object.__setattr__(self, "vgg_jacobian", vgg_jacobian)
        object.__setattr__(self, "alpha_coordinates", alpha_coordinates)
        object.__setattr__(self, "beta_coordinates", beta_coordinates)
        object.__setattr__(self, "resnet_anchors", anchors[0])
        object.__setattr__(self, "vgg_anchors", anchors[1])


@dataclass(frozen=True, slots=True)
class AnimationTiming:
    """Explicit GIF timing in milliseconds.

    Values are constrained to GIF's ten-millisecond clock so that metadata and
    decoded frame durations agree exactly.
    """

    orientation_ms: int = 3000
    checkpoint_ms: int = 1000
    conclusion_ms: int = 6000

    def __post_init__(self) -> None:
        for name, value in (
            ("orientation_ms", self.orientation_ms),
            ("checkpoint_ms", self.checkpoint_ms),
            ("conclusion_ms", self.conclusion_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value % 10:
                raise ValueError(f"{name} must be divisible by 10 for exact GIF timing")


@dataclass(frozen=True, slots=True)
class AnimationArtifacts:
    """Paths and verified timing for one rendered animation bundle."""

    gif_path: Path
    final_png_path: Path
    metadata_path: Path
    frame_count: int
    durations_ms: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Scale:
    display_min: float
    display_max: float
    data_min: float
    data_max: float
    source: str

    def as_metadata(self) -> dict[str, float | str | bool]:
        return {
            "display_min": self.display_min,
            "display_max": self.display_max,
            "data_min": self.data_min,
            "data_max": self.data_max,
            "source": self.source,
            "clips_data": self.display_min > self.data_min or self.display_max < self.data_max,
        }


@dataclass(frozen=True, slots=True)
class _Fonts:
    title: ImageFont.ImageFont
    heading: ImageFont.ImageFont
    body: ImageFont.ImageFont
    small: ImageFont.ImageFont
    tiny: ImageFont.ImageFont
    badge: ImageFont.ImageFont


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def _fonts(canvas_size: tuple[int, int]) -> _Fonts:
    scale = min(canvas_size[0] / 1440.0, canvas_size[1] / 900.0)

    def sized(value: int) -> int:
        return max(11, round(value * scale))

    return _Fonts(
        title=_font(sized(31), bold=True),
        heading=_font(sized(21), bold=True),
        body=_font(sized(17)),
        small=_font(sized(14)),
        tiny=_font(sized(12)),
        badge=_font(sized(14), bold=True),
    )


def _font_provenance(font: ImageFont.ImageFont) -> dict[str, str | None]:
    try:
        family, style = font.getname()
    except (AttributeError, OSError):
        family, style = type(font).__name__, None
    source = getattr(font, "path", None)
    source_path = Path(source) if isinstance(source, (str, Path)) else None
    digest: str | None = None
    filename: str | None = None
    if source_path is not None and source_path.is_file():
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        filename = source_path.name
    return {
        "family": str(family),
        "style": None if style is None else str(style),
        "file": filename,
        "sha256": digest,
    }


def _rendering_provenance(canvas_size: tuple[int, int]) -> dict[str, Any]:
    fonts = _fonts(canvas_size)
    return {
        "engine": "Pillow",
        "pillow_version": PILLOW_VERSION,
        "font_regular": _font_provenance(fonts.body),
        "font_bold": _font_provenance(fonts.heading),
        "gif_quantization": {
            "colors": 256,
            "dither": "none",
            "method": "median_cut",
        },
    }


def _validate_canvas(canvas_size: tuple[int, int]) -> tuple[int, int]:
    if len(canvas_size) != 2:
        raise ValueError("canvas_size must be a width/height pair")
    width, height = canvas_size
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("canvas dimensions must be integers")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("canvas dimensions must be integers")
    if width < 1200 or height < 760:
        raise ValueError("canvas_size must be at least 1200 by 760 for readable labels")
    return width, height


def _global_scale(
    arrays: Sequence[FloatArray],
    provided: tuple[float, float] | None,
    *,
    name: str,
) -> _Scale:
    data_min = min(float(array.min()) for array in arrays)
    data_max = max(float(array.max()) for array in arrays)
    if provided is not None:
        if len(provided) != 2 or not np.all(np.isfinite(provided)):
            raise ValueError(f"{name} must contain two finite values")
        display_min, display_max = (float(value) for value in provided)
        if display_min >= display_max:
            raise ValueError(f"{name} must be strictly increasing")
        source = "provided_global"
    elif data_min < data_max:
        display_min, display_max = data_min, data_max
        source = "computed_global"
    else:
        padding = max(abs(data_min) * 0.05, 1.0)
        display_min, display_max = data_min - padding, data_max + padding
        source = "computed_global_constant_padding"
    return _Scale(display_min, display_max, data_min, data_max, source)


def _padded_global_scale(
    arrays: Sequence[FloatArray],
    provided: tuple[float, float] | None,
    *,
    name: str,
    include_zero: bool = False,
) -> _Scale:
    raw = _global_scale(arrays, provided, name=name)
    if provided is not None:
        return raw
    data_min = min(raw.data_min, 0.0) if include_zero else raw.data_min
    data_max = max(raw.data_max, 0.0) if include_zero else raw.data_max
    span = data_max - data_min
    if span <= 0:
        span = max(abs(data_min), 1.0)
    padding = 0.05 * span
    return _Scale(
        data_min - padding,
        data_max + padding,
        raw.data_min,
        raw.data_max,
        "computed_global_with_padding",
    )


def _validate_checkpoints(frames: Sequence[RealFakeFrame | ArchitectureCellsFrame]) -> None:
    if not frames:
        raise ValueError("at least one checkpoint frame is required")
    labels = [frame.checkpoint for frame in frames]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint labels must be unique and already synchronized")


def _same_array(reference: FloatArray, candidate: FloatArray) -> bool:
    return reference.shape == candidate.shape and np.array_equal(reference, candidate)


def _validate_real_fake_sequence(frames: Sequence[RealFakeFrame]) -> None:
    _validate_checkpoints(frames)
    heatmap_shape = frames[0].real_heatmap.shape
    curve_x = frames[0].curve_x
    for frame in frames:
        if frame.real_heatmap.shape != heatmap_shape or frame.fake_heatmap.shape != heatmap_shape:
            raise ValueError("all real/fake heatmaps must retain one shape across checkpoints")
        if not _same_array(frames[0].heatmap_x, frame.heatmap_x) or not _same_array(
            frames[0].heatmap_y, frame.heatmap_y
        ):
            raise ValueError("all real/fake frames must use identical heatmap coordinates")
        if not _same_array(curve_x, frame.curve_x):
            raise ValueError("all real/fake frames must use identical curve_x coordinates")


def _validate_architecture_sequence(frames: Sequence[ArchitectureCellsFrame]) -> None:
    _validate_checkpoints(frames)
    grid_shape = frames[0].resnet_jacobian.shape
    for frame in frames:
        if frame.resnet_jacobian.shape != grid_shape:
            raise ValueError("all architecture frames must retain one grid shape")
        if frame.vgg_jacobian.shape != grid_shape:
            raise ValueError("all architecture frames must retain one grid shape")
        if not _same_array(frames[0].alpha_coordinates, frame.alpha_coordinates) or not _same_array(
            frames[0].beta_coordinates, frame.beta_coordinates
        ):
            raise ValueError("all architecture frames must use identical plane coordinates")
        if (frames[0].resnet_anchors is None) != (frame.resnet_anchors is None):
            raise ValueError("anchor-marker presence must remain fixed across checkpoints")


def _format_tick(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000 or magnitude < 0.01:
        return f"{value:.1e}"
    return f"{value:.3g}"


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            proposed = f"{current} {word}"
            if _text_width(draw, proposed, font) <= max_width:
                current = proposed
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return "\n".join(lines)


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fonts: _Fonts,
    *,
    fill: str,
    foreground: str = "#FFFFFF",
) -> None:
    draw.rounded_rectangle(box, radius=9, fill=fill)
    draw.text(
        ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
        text,
        font=fonts.badge,
        fill=foreground,
        anchor="mm",
    )


def _draw_header(
    draw: ImageDraw.ImageDraw,
    canvas_size: tuple[int, int],
    fonts: _Fonts,
    *,
    title: str,
    checkpoint: str,
) -> None:
    width, _ = canvas_size
    badge_width = round(width * 0.20)
    title_box = draw.textbbox(
        (width // 2, 42),
        title,
        font=fonts.title,
        anchor="mm",
    )
    if title_box[0] < 102 or title_box[2] > width - badge_width - 44:
        raise ValueError("title does not fit between the fixed GIF and checkpoint badges")
    _draw_badge(draw, (24, 24, 82, 56), "GIF", fonts, fill=_INK)
    draw.text((width // 2, 42), title, font=fonts.title, fill=_INK, anchor="mm")
    display_checkpoint = checkpoint
    while (
        len(display_checkpoint) > 5
        and _text_width(draw, f"CHECKPOINT {display_checkpoint}", fonts.badge) > badge_width - 26
    ):
        midpoint = len(display_checkpoint) // 2
        display_checkpoint = (
            f"{display_checkpoint[: midpoint - 2]}…{display_checkpoint[midpoint + 2 :]}"
        )
    if _text_width(draw, f"CHECKPOINT {display_checkpoint}", fonts.badge) > badge_width - 26:
        raise ValueError("checkpoint label does not fit the fixed checkpoint badge")
    _draw_badge(
        draw,
        (width - badge_width - 24, 20, width - 24, 60),
        f"CHECKPOINT {display_checkpoint}",
        fonts,
        fill=_BLUE,
    )
    draw.line((24, 82, width - 24, 82), fill="#C7CDD8", width=2)


_COLOR_STOPS = np.asarray(
    (
        (10, 18, 46),
        (61, 18, 101),
        (135, 38, 105),
        (218, 72, 94),
        (254, 204, 107),
    ),
    dtype=np.float64,
)


def _scalar_to_rgb(values: FloatArray, scale: _Scale) -> NDArray[np.uint8]:
    normalized = np.clip(
        (values - scale.display_min) / (scale.display_max - scale.display_min),
        0.0,
        1.0,
    )
    position = normalized * (_COLOR_STOPS.shape[0] - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, _COLOR_STOPS.shape[0] - 1)
    fraction = (position - lower)[..., None]
    colors = _COLOR_STOPS[lower] * (1.0 - fraction) + _COLOR_STOPS[upper] * fraction
    return np.rint(colors).astype(np.uint8)


def _bilinear_sample_grid(
    values: FloatArray,
    size: tuple[int, int],
) -> FloatArray:
    """Resample point values with endpoint samples on the endpoint pixels."""

    width, height = size
    rows, columns = values.shape
    x_positions = np.linspace(0.0, columns - 1, width)
    x_lower = np.floor(x_positions).astype(np.int64)
    x_upper = np.minimum(x_lower + 1, columns - 1)
    x_fraction = x_positions - x_lower
    horizontal = (
        values[:, x_lower] * (1.0 - x_fraction)[None, :] + values[:, x_upper] * x_fraction[None, :]
    )

    # Input row zero is the minimum y coordinate, while raster row zero is the top.
    y_positions = np.linspace(rows - 1, 0.0, height)
    y_lower = np.floor(y_positions).astype(np.int64)
    y_upper = np.minimum(y_lower + 1, rows - 1)
    y_fraction = y_positions - y_lower
    sampled = (
        horizontal[y_lower, :] * (1.0 - y_fraction)[:, None]
        + horizontal[y_upper, :] * y_fraction[:, None]
    )
    sampled.setflags(write=False)
    return sampled


def _nearest_sample_grid(
    values: FloatArray,
    size: tuple[int, int],
) -> FloatArray:
    """Nearest-neighbor point sampling with coordinate-aligned endpoints."""

    width, height = size
    rows, columns = values.shape[:2]
    x_indices = np.rint(np.linspace(0.0, columns - 1, width)).astype(np.int64)
    y_indices = np.rint(np.linspace(rows - 1, 0.0, height)).astype(np.int64)
    sampled = values[y_indices[:, None], x_indices[None, :], :]
    sampled.setflags(write=False)
    return sampled


def _draw_rotated_text(
    canvas: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    center: tuple[int, int],
    *,
    fill: str = _INK,
    max_length: int | None = None,
) -> None:
    measuring_draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    text_box = measuring_draw.textbbox((0, 0), text, font=font)
    if max_length is not None and text_box[2] - text_box[0] > max_length:
        raise ValueError("rotated axis label does not fit the fixed animation layout")
    padding = 10
    scratch = Image.new(
        "RGBA",
        (
            max(1, text_box[2] - text_box[0] + 2 * padding),
            max(1, text_box[3] - text_box[1] + 2 * padding),
        ),
        (0, 0, 0, 0),
    )
    scratch_draw = ImageDraw.Draw(scratch)
    scratch_draw.text(
        (padding - text_box[0], padding - text_box[1]),
        text,
        font=font,
        fill=fill,
    )
    rotated = scratch.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.alpha_composite(
        rotated, (center[0] - rotated.width // 2, center[1] - rotated.height // 2)
    )


def _data_to_pixel(
    x: float,
    y: float,
    rect: tuple[int, int, int, int],
    extent: tuple[float, float, float, float],
) -> tuple[int, int]:
    left, top, right, bottom = rect
    xmin, xmax, ymin, ymax = extent
    px = left + (x - xmin) / (xmax - xmin) * (right - left)
    py = bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
    return round(px), round(py)


def _draw_field_axes(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    extent: tuple[float, float, float, float],
    fonts: _Fonts,
    *,
    x_label: str,
    y_label: str,
) -> None:
    left, top, right, bottom = rect
    xmin, xmax, ymin, ymax = extent
    draw.rectangle(rect, outline=_INK, width=2)
    for value in np.linspace(xmin, xmax, 3):
        x, _ = _data_to_pixel(float(value), ymin, rect, extent)
        draw.line((x, bottom, x, bottom + 7), fill=_INK, width=2)
        draw.text(
            (x, bottom + 10),
            _format_tick(float(value)),
            font=fonts.tiny,
            fill=_MUTED,
            anchor="ma",
        )
    for value in np.linspace(ymin, ymax, 3):
        _, y = _data_to_pixel(xmin, float(value), rect, extent)
        draw.line((left - 7, y, left, y), fill=_INK, width=2)
        draw.text(
            (left - 11, y),
            _format_tick(float(value)),
            font=fonts.tiny,
            fill=_MUTED,
            anchor="rm",
        )
    draw.text(
        ((left + right) // 2, bottom + 38),
        x_label,
        font=fonts.small,
        fill=_INK,
        anchor="mm",
    )
    _draw_rotated_text(
        canvas,
        y_label,
        fonts.small,
        (left - 53, (top + bottom) // 2),
        max_length=bottom - top - 16,
    )


def _draw_scalar_field(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    values: FloatArray,
    rect: tuple[int, int, int, int],
    extent: tuple[float, float, float, float],
    scale: _Scale,
    fonts: _Fonts,
    *,
    title: str,
    x_label: str,
    y_label: str,
    center_marker: tuple[float, float] | None = None,
) -> None:
    panel_width = rect[2] - rect[0]
    if _text_width(draw, title, fonts.heading) > panel_width:
        raise ValueError("scalar field title does not fit its panel")
    raster_size = (rect[2] - rect[0] + 1, rect[3] - rect[1] + 1)
    colors = _scalar_to_rgb(
        _bilinear_sample_grid(values, raster_size),
        scale,
    )
    image = Image.fromarray(colors)
    canvas.paste(image, (rect[0], rect[1]))
    _draw_field_axes(
        canvas,
        draw,
        rect,
        extent,
        fonts,
        x_label=x_label,
        y_label=y_label,
    )
    draw.text(
        ((rect[0] + rect[2]) // 2, rect[1] - 29),
        title,
        font=fonts.heading,
        fill=_INK,
        anchor="mm",
    )
    if center_marker is not None:
        x, y = _data_to_pixel(*center_marker, rect, extent)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#FFFFFF", outline=_INK, width=3)


def _validate_stacked_field_vertical_clearance(
    draw: ImageDraw.ImageDraw,
    fonts: _Fonts,
    *,
    upper_rect: tuple[int, int, int, int],
    lower_rect: tuple[int, int, int, int],
    lower_title: str,
) -> None:
    """Ensure upper tick labels clear the lower scalar-field title."""

    upper_tick_box = draw.textbbox(
        (upper_rect[2], upper_rect[3] + 10),
        "0",
        font=fonts.tiny,
        anchor="ma",
    )
    lower_title_box = draw.textbbox(
        ((lower_rect[0] + lower_rect[2]) // 2, lower_rect[1] - 29),
        lower_title,
        font=fonts.heading,
        anchor="mm",
    )
    if upper_tick_box[3] >= lower_title_box[1]:
        raise ValueError("stacked scalar field tick labels overlap the lower title")


def _draw_rgb_field(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    values: FloatArray,
    rect: tuple[int, int, int, int],
    extent: tuple[float, float, float, float],
    fonts: _Fonts,
    *,
    x_label: str,
    y_label: str,
    anchors: FloatArray | None,
) -> None:
    raster_size = (rect[2] - rect[0] + 1, rect[3] - rect[1] + 1)
    pixels = np.rint(_nearest_sample_grid(values, raster_size) * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels)
    canvas.paste(image, (rect[0], rect[1]))
    _draw_field_axes(
        canvas,
        draw,
        rect,
        extent,
        fonts,
        x_label=x_label,
        y_label=y_label,
    )
    if anchors is None:
        return
    colors = (_RED, _GREEN, _BLUE)
    for label, color, (alpha, beta) in zip("ABC", colors, anchors, strict=True):
        x, y = _data_to_pixel(float(alpha), float(beta), rect, extent)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color, outline="#FFFFFF", width=3)
        draw.text((x + 13, y - 10), label, font=fonts.badge, fill=_INK)


def _draw_colorbar(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    scale: _Scale,
    fonts: _Fonts,
    *,
    label: str,
) -> None:
    height = rect[3] - rect[1]
    gradient = np.linspace(scale.display_max, scale.display_min, height)[:, None]
    colors = _scalar_to_rgb(np.repeat(gradient, rect[2] - rect[0], axis=1), scale)
    canvas.paste(Image.fromarray(colors), (rect[0], rect[1]))
    draw.rectangle(rect, outline=_INK, width=2)
    draw.text(
        ((rect[0] + rect[2]) // 2, rect[1] - 25),
        label,
        font=fonts.small,
        fill=_INK,
        anchor="mm",
    )
    for value in np.linspace(scale.display_min, scale.display_max, 3):
        fraction = (value - scale.display_min) / (scale.display_max - scale.display_min)
        y = round(rect[3] - fraction * (rect[3] - rect[1]))
        draw.line((rect[2], y, rect[2] + 7, y), fill=_INK, width=2)
        draw.text(
            (rect[2] + 11, y),
            _format_tick(float(value)),
            font=fonts.tiny,
            fill=_MUTED,
            anchor="lm",
        )


def _line_points(
    x: FloatArray,
    y: FloatArray,
    rect: tuple[int, int, int, int],
    x_scale: _Scale,
    y_scale: _Scale,
) -> list[tuple[int, int]]:
    left, top, right, bottom = rect
    px = left + (x - x_scale.display_min) / (x_scale.display_max - x_scale.display_min) * (
        right - left
    )
    py = bottom - (y - y_scale.display_min) / (y_scale.display_max - y_scale.display_min) * (
        bottom - top
    )
    px = np.clip(px, left, right)
    py = np.clip(py, top, bottom)
    return [
        (round(float(x_value)), round(float(y_value)))
        for x_value, y_value in zip(px, py, strict=True)
    ]


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: str,
    width: int,
    dash: int = 12,
    gap: int = 8,
) -> None:
    for start, end in pairwise(points):
        x0, y0 = start
        x1, y1 = end
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length == 0:
            continue
        cursor = 0.0
        while cursor < length:
            segment_end = min(cursor + dash, length)
            sx = x0 + (x1 - x0) * cursor / length
            sy = y0 + (y1 - y0) * cursor / length
            ex = x0 + (x1 - x0) * segment_end / length
            ey = y0 + (y1 - y0) * segment_end / length
            draw.line((round(sx), round(sy), round(ex), round(ey)), fill=fill, width=width)
            cursor += dash + gap


def _draw_curve_plot(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    frame: RealFakeFrame,
    rect: tuple[int, int, int, int],
    x_scale: _Scale,
    y_scale: _Scale,
    fonts: _Fonts,
    *,
    curve_label: str,
) -> dict[str, tuple[int, int]]:
    draw.rounded_rectangle(
        (rect[0] - 13, rect[1] - 38, rect[2] + 13, rect[3] + 45),
        radius=12,
        fill=_PANEL_BACKGROUND,
        outline="#D2D7E0",
        width=2,
    )
    draw.text(
        ((rect[0] + rect[2]) // 2, rect[1] - 22),
        "SOURCE ANALOGUE · PERTURBATION PROFILE",
        font=fonts.heading,
        fill=_INK,
        anchor="mm",
    )
    for value in np.linspace(x_scale.display_min, x_scale.display_max, 5):
        fraction = (value - x_scale.display_min) / (x_scale.display_max - x_scale.display_min)
        x = round(rect[0] + fraction * (rect[2] - rect[0]))
        draw.line((x, rect[1], x, rect[3]), fill=_GRID, width=1)
        draw.text(
            (x, rect[3] + 9),
            _format_tick(float(value)),
            font=fonts.tiny,
            fill=_MUTED,
            anchor="ma",
        )
    for value in np.linspace(y_scale.display_min, y_scale.display_max, 4):
        fraction = (value - y_scale.display_min) / (y_scale.display_max - y_scale.display_min)
        y = round(rect[3] - fraction * (rect[3] - rect[1]))
        draw.line((rect[0], y, rect[2], y), fill=_GRID, width=1)
        draw.text(
            (rect[0] - 10, y),
            _format_tick(float(value)),
            font=fonts.tiny,
            fill=_MUTED,
            anchor="rm",
        )
    real_points = _line_points(frame.curve_x, frame.real_curve, rect, x_scale, y_scale)
    fake_points = _line_points(frame.curve_x, frame.fake_curve, rect, x_scale, y_scale)
    draw.line(real_points, fill=_REAL, width=4, joint="curve")
    _draw_dashed_line(draw, fake_points, fill=_FAKE, width=4)
    for point, color in ((real_points[-1], _REAL), (fake_points[-1], _FAKE)):
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="#FFFFFF", width=2)
    draw.rectangle(rect, outline=_INK, width=2)
    draw.text(
        ((rect[0] + rect[2]) // 2, rect[3] + 33),
        "Perturbation coordinate",
        font=fonts.small,
        fill=_INK,
        anchor="mm",
    )
    _draw_rotated_text(
        canvas,
        curve_label,
        fonts.small,
        (rect[0] - 52, (rect[1] + rect[3]) // 2),
        max_length=rect[3] - rect[1] - 16,
    )
    legend_left = rect[0] + 18
    legend_top = rect[1] + 13
    draw.rounded_rectangle(
        (legend_left, legend_top, legend_left + 268, legend_top + 56),
        radius=8,
        fill="#FFFFFFE8",
        outline="#CCD2DC",
        width=1,
    )
    draw.line(
        (legend_left + 13, legend_top + 18, legend_left + 53, legend_top + 18),
        fill=_REAL,
        width=4,
    )
    draw.text(
        (legend_left + 62, legend_top + 18),
        "Real activation",
        font=fonts.small,
        fill=_INK,
        anchor="lm",
    )
    _draw_dashed_line(
        draw,
        [(legend_left + 13, legend_top + 40), (legend_left + 53, legend_top + 40)],
        fill=_FAKE,
        width=4,
        dash=8,
        gap=5,
    )
    draw.text(
        (legend_left + 62, legend_top + 40),
        "Matched fake activation",
        font=fonts.small,
        fill=_INK,
        anchor="lm",
    )
    return {"real_endpoint": real_points[-1], "fake_endpoint": fake_points[-1]}


def _draw_callout(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    target: tuple[int, int],
    fonts: _Fonts,
    *,
    color: str,
) -> None:
    wrapped = _wrap_text(draw, text, fonts.body, box[2] - box[0] - 28)
    origin = (box[0] + 14, box[1] + 14)
    text_box = draw.multiline_textbbox(origin, wrapped, font=fonts.body, spacing=6)
    if (
        text_box[0] < box[0] + 14
        or text_box[1] < box[1] + 14
        or text_box[2] > box[2] - 14
        or text_box[3] > box[3] - 14
    ):
        raise ValueError("callout text does not fit the fixed animation layout")
    draw.rounded_rectangle(box, radius=12, fill="#FFFFF2", outline=color, width=3)
    draw.multiline_text(
        origin,
        wrapped,
        font=fonts.body,
        fill=_INK,
        spacing=6,
    )
    start = (box[0], (box[1] + box[3]) // 2)
    draw.line((start, target), fill=color, width=4)
    x, y = target
    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="#FFFFFF", width=2)


def _new_canvas(canvas_size: tuple[int, int]) -> Image.Image:
    return Image.new("RGBA", canvas_size, _BACKGROUND)


def _render_real_fake_frame(
    frame: RealFakeFrame,
    *,
    canvas_size: tuple[int, int],
    extent: tuple[float, float, float, float],
    heatmap_scale: _Scale,
    curve_x_scale: _Scale,
    curve_y_scale: _Scale,
    title: str,
    scalar_label: str,
    heatmap_classification_label: str,
    curve_label: str,
    role: str,
    final_callout: str,
) -> Image.Image:
    width, height = canvas_size
    fonts = _fonts(canvas_size)
    canvas = _new_canvas(canvas_size)
    draw = ImageDraw.Draw(canvas, "RGBA")
    _draw_header(draw, canvas_size, fonts, title=title, checkpoint=frame.checkpoint)

    top = round(height * 0.17)
    heat_size = min(round(height * 0.42), round(width * 0.275))
    first_rect = (round(width * 0.065), top, round(width * 0.065) + heat_size, top + heat_size)
    second_left = first_rect[2] + round(width * 0.055)
    second_rect = (second_left, top, second_left + heat_size, top + heat_size)
    _draw_scalar_field(
        canvas,
        draw,
        frame.real_heatmap,
        first_rect,
        extent,
        heatmap_scale,
        fonts,
        title=f"{heatmap_classification_label} · REAL",
        x_label="Direction 1",
        y_label="Direction 2",
        center_marker=(0.0, 0.0),
    )
    _draw_scalar_field(
        canvas,
        draw,
        frame.fake_heatmap,
        second_rect,
        extent,
        heatmap_scale,
        fonts,
        title=f"{heatmap_classification_label} · MATCHED FAKE",
        x_label="Direction 1",
        y_label="Direction 2",
        center_marker=(0.0, 0.0),
    )

    bar_left = second_rect[2] + round(width * 0.025)
    bar_rect = (bar_left, top, bar_left + 28, top + heat_size)
    _draw_colorbar(
        canvas,
        draw,
        bar_rect,
        heatmap_scale,
        fonts,
        label=scalar_label,
    )

    curve_rect = (
        round(width * 0.075),
        round(height * 0.735),
        width - round(width * 0.055),
        height - round(height * 0.075),
    )
    endpoints = _draw_curve_plot(
        canvas,
        draw,
        frame,
        curve_rect,
        curve_x_scale,
        curve_y_scale,
        fonts,
        curve_label=curve_label,
    )

    info_left = bar_rect[2] + 55
    info_box = (info_left, top, width - 28, top + heat_size)
    if role == "orientation":
        text = (
            "Orientation → real and covariance-matched fake centers advance at the same "
            "checkpoint. Heatmap colors and curve axes stay fixed for the whole GIF."
        )
        _draw_callout(draw, info_box, text, (bar_rect[0], top + 45), fonts, color=_BLUE)
    elif role == "conclusion":
        _draw_callout(
            draw,
            info_box,
            final_callout,
            endpoints["real_endpoint"],
            fonts,
            color=_FAKE,
        )
    else:
        draw.rounded_rectangle(
            info_box,
            radius=12,
            fill="#FFFFFFD9",
            outline="#D2D7E0",
            width=2,
        )
        draw.text(
            (info_left + 18, top + 20),
            "DIAGNOSTIC",
            font=fonts.heading,
            fill=_INK,
        )
        diagnostic = (
            "Shared scalar scale\n"
            "Shared perturbation axes\n"
            "White circle = center\n"
            "Solid = real profile\n"
            "Dashed = matched fake"
        )
        draw.multiline_text(
            (info_left + 18, top + 62),
            diagnostic,
            font=fonts.body,
            fill=_MUTED,
            spacing=12,
        )
    return canvas.convert("RGB")


def _draw_architecture_row_badge(
    draw: ImageDraw.ImageDraw,
    fonts: _Fonts,
    *,
    y: int,
    text: str,
    color: str,
) -> None:
    box = (18, y - 25, 116, y + 25)
    _draw_badge(draw, box, text, fonts, fill=color)


def _jacobian_max_target(
    frame: ArchitectureCellsFrame,
    resnet_rect: tuple[int, int, int, int],
    vgg_rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    if float(frame.resnet_jacobian.max()) >= float(frame.vgg_jacobian.max()):
        field = frame.resnet_jacobian
        rect = resnet_rect
    else:
        field = frame.vgg_jacobian
        rect = vgg_rect
    maxima = np.argwhere(field == field.max())
    row, column = np.rint(np.median(maxima, axis=0)).astype(int)
    x = rect[0] + column / (field.shape[1] - 1) * (rect[2] - rect[0])
    y = rect[3] - row / (field.shape[0] - 1) * (rect[3] - rect[1])
    return round(x), round(y)


def _draw_anchor_legend(
    draw: ImageDraw.ImageDraw,
    fonts: _Fonts,
    *,
    left: int,
    top: int,
) -> None:
    draw.text((left, top), "ANCHORS", font=fonts.heading, fill=_INK)
    for index, (label, color) in enumerate(zip("ABC", (_RED, _GREEN, _BLUE), strict=True)):
        y = top + 48 + index * 38
        draw.ellipse((left, y - 9, left + 18, y + 9), fill=color, outline="#FFFFFF", width=2)
        draw.text((left + 30, y), f"Anchor {label}", font=fonts.body, fill=_INK, anchor="lm")


def _render_architecture_frame(
    frame: ArchitectureCellsFrame,
    *,
    canvas_size: tuple[int, int],
    extent: tuple[float, float, float, float],
    jacobian_scale: _Scale,
    title: str,
    jacobian_label: str,
    resnet_jacobian_label: str,
    vgg_jacobian_label: str,
    role: str,
    final_callout: str,
) -> Image.Image:
    width, height = canvas_size
    fonts = _fonts(canvas_size)
    canvas = _new_canvas(canvas_size)
    draw = ImageDraw.Draw(canvas, "RGBA")
    _draw_header(draw, canvas_size, fonts, title=title, checkpoint=frame.checkpoint)

    top = round(height * 0.16)
    gap = round(height * 0.085)
    panel_size = min(
        (height - top - gap - round(height * 0.055)) // 2,
        round(width * 0.235),
    )
    cells_left = round(width * 0.16)
    jacobian_left = round(width * 0.46)
    row_two_top = top + panel_size + gap
    resnet_cells = (cells_left, top, cells_left + panel_size, top + panel_size)
    resnet_jacobian = (
        jacobian_left,
        top,
        jacobian_left + panel_size,
        top + panel_size,
    )
    vgg_cells = (
        cells_left,
        row_two_top,
        cells_left + panel_size,
        row_two_top + panel_size,
    )
    vgg_jacobian = (
        jacobian_left,
        row_two_top,
        jacobian_left + panel_size,
        row_two_top + panel_size,
    )
    _validate_stacked_field_vertical_clearance(
        draw,
        fonts,
        upper_rect=resnet_jacobian,
        lower_rect=vgg_jacobian,
        lower_title=vgg_jacobian_label,
    )

    source_heading = "SOURCE ANALOGUE · THREE-ANCHOR RGB"
    if _text_width(draw, source_heading, fonts.small) > panel_size:
        raise ValueError("source analogue title does not fit its panel")
    draw.text(
        ((resnet_cells[0] + resnet_cells[2]) // 2, top - 35),
        source_heading,
        font=fonts.small,
        fill=_INK,
        anchor="mm",
    )
    _draw_architecture_row_badge(
        draw,
        fonts,
        y=(resnet_cells[1] + resnet_cells[3]) // 2,
        text="RESNET",
        color=_RESNET,
    )
    _draw_architecture_row_badge(
        draw,
        fonts,
        y=(vgg_cells[1] + vgg_cells[3]) // 2,
        text="VGG",
        color=_VGG,
    )
    for values, rect, anchors, x_label in (
        (frame.resnet_rgb, resnet_cells, frame.resnet_anchors, ""),
        (frame.vgg_rgb, vgg_cells, frame.vgg_anchors, "\u03b1"),
    ):
        _draw_rgb_field(
            canvas,
            draw,
            values,
            rect,
            extent,
            fonts,
            x_label=x_label,
            y_label="\u03b2",
            anchors=anchors,
        )
    for values, rect, field_label, x_label in (
        (frame.resnet_jacobian, resnet_jacobian, resnet_jacobian_label, ""),
        (frame.vgg_jacobian, vgg_jacobian, vgg_jacobian_label, "\u03b1"),
    ):
        _draw_scalar_field(
            canvas,
            draw,
            values,
            rect,
            extent,
            jacobian_scale,
            fonts,
            title=field_label,
            x_label=x_label,
            y_label="\u03b2",
        )

    bar_left = resnet_jacobian[2] + 23
    bar_rect = (bar_left, top, bar_left + 28, vgg_jacobian[3])
    _draw_colorbar(
        canvas,
        draw,
        bar_rect,
        jacobian_scale,
        fonts,
        label=jacobian_label,
    )
    info_left = bar_rect[2] + 64
    info_box = (info_left, top, width - 26, vgg_jacobian[3])
    if role == "orientation":
        text = (
            "Orientation → rows share a coefficient grid and Jacobian scale; RGB is "
            "normalized per row/channel. ResNet shows 2D D(T-I), VGG 2D DT; not causal."
        )
        _draw_callout(draw, info_box, text, (bar_rect[0], top + 55), fonts, color=_BLUE)
    elif role == "conclusion":
        target = _jacobian_max_target(frame, resnet_jacobian, vgg_jacobian)
        _draw_callout(draw, info_box, final_callout, target, fonts, color=_FAKE)
    else:
        draw.rounded_rectangle(
            info_box,
            radius=12,
            fill="#FFFFFFD9",
            outline="#D2D7E0",
            width=2,
        )
        _draw_anchor_legend(draw, fonts, left=info_left + 18, top=top + 20)
        diagnostic = (
            "Matched image IDs\n"
            "Same coefficient grid\n"
            "Jacobian scale shared\n"
            "RGB normalized per row/channel\n"
            "ResNet: 2D D(T-I)\n"
            "VGG: 2D DT\n"
            "Descriptive; not causal"
        )
        draw.multiline_text(
            (info_left + 18, top + 200),
            diagnostic,
            font=fonts.body,
            fill=_MUTED,
            spacing=12,
        )
    return canvas.convert("RGB")


def _render_schedule(
    checkpoints: Sequence[str],
    timing: AnimationTiming,
) -> tuple[list[str], list[str], list[int]]:
    rendered_checkpoints = [*checkpoints, checkpoints[-1]]
    roles = ["orientation"] + ["checkpoint"] * (len(checkpoints) - 1) + ["conclusion"]
    durations = (
        [timing.orientation_ms]
        + [timing.checkpoint_ms] * (len(checkpoints) - 1)
        + [timing.conclusion_ms]
    )
    return rendered_checkpoints, roles, durations


def _write_animation_bundle(
    frames: Sequence[Image.Image],
    durations: Sequence[int],
    output_prefix: str | Path,
    metadata: dict[str, Any],
) -> AnimationArtifacts:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    gif_path = Path(f"{prefix}.gif")
    final_png_path = Path(f"{prefix}_final.png")
    metadata_path = Path(f"{prefix}_metadata.json")

    palette_frames = [
        frame.quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        for frame in frames
    ]
    palette_frames[0].save(
        gif_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=list(durations),
        loop=0,
        disposal=2,
        optimize=False,
    )
    frames[-1].save(final_png_path, format="PNG", optimize=False)
    _verify_gif(gif_path, len(frames), tuple(durations), frames[0].size)

    metadata["files"] = {
        "gif": gif_path.name,
        "final_png": final_png_path.name,
        "metadata": metadata_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return AnimationArtifacts(
        gif_path=gif_path,
        final_png_path=final_png_path,
        metadata_path=metadata_path,
        frame_count=len(frames),
        durations_ms=tuple(durations),
    )


def _verify_gif(
    path: Path,
    expected_count: int,
    expected_durations: tuple[int, ...],
    expected_size: tuple[int, int],
) -> None:
    with Image.open(path) as image:
        if image.n_frames != expected_count:
            raise RuntimeError(
                f"GIF frame count changed during encoding: {image.n_frames} != {expected_count}"
            )
        if image.size != expected_size:
            raise RuntimeError(
                f"GIF canvas changed during encoding: {image.size} != {expected_size}"
            )
        observed_durations = tuple(
            int(frame.info.get("duration", 0)) for frame in ImageSequence.Iterator(image)
        )
        if observed_durations != expected_durations:
            raise RuntimeError(
                f"GIF timing changed during encoding: {observed_durations} != {expected_durations}"
            )
        if int(image.info.get("loop", -1)) != 0:
            raise RuntimeError("GIF loop metadata must be zero (infinite looping)")


def _frame_metadata(
    rendered_checkpoints: Sequence[str],
    roles: Sequence[str],
    durations: Sequence[int],
) -> list[dict[str, str | int]]:
    return [
        {
            "index": index,
            "checkpoint": checkpoint,
            "role": role,
            "duration_ms": duration,
        }
        for index, (checkpoint, role, duration) in enumerate(
            zip(rendered_checkpoints, roles, durations, strict=True)
        )
    ]


def render_real_fake_animation(
    frames: Sequence[RealFakeFrame],
    output_prefix: str | Path,
    *,
    heatmap_scale: tuple[float, float] | None = None,
    curve_y_range: tuple[float, float] | None = None,
    title: str = "Real vs matched-fake activation geometry",
    scalar_label: str = "Jacobian norm",
    heatmap_classification_label: str = "SCALAR FIELD",
    scalar_estimand: str = "caller-supplied scalar field",
    curve_label: str = "Response",
    curve_estimand: str = "caller-supplied response profile",
    final_callout: str = (
        "Final checkpoint → compare the highlighted real trajectory with the matched-fake "
        "trajectory; the shared axes and color scale rule out a rescaling artifact."
    ),
    canvas_size: tuple[int, int] = (1440, 900),
    timing: AnimationTiming | None = None,
) -> AnimationArtifacts:
    """Render synchronized real/fake heatmaps and curves.

    One extra conclusion frame duplicates the last checkpoint before introducing
    the final callout, so the visual claim is not presented on the same frame as
    a data change.
    """

    timing = AnimationTiming() if timing is None else timing
    prepared = tuple(frames)
    _validate_real_fake_sequence(prepared)
    canvas_size = _validate_canvas(canvas_size)
    title = _label(title, name="title", max_characters=72)
    scalar_label = _label(scalar_label, name="scalar_label", max_characters=36)
    heatmap_classification_label = _label(
        heatmap_classification_label,
        name="heatmap_classification_label",
        max_characters=18,
    )
    scalar_estimand = _label(
        scalar_estimand,
        name="scalar_estimand",
        max_characters=160,
    )
    curve_label = _label(curve_label, name="curve_label", max_characters=36)
    curve_estimand = _label(curve_estimand, name="curve_estimand", max_characters=160)
    final_callout = _label(final_callout, name="final_callout", max_characters=320)
    extent = (
        float(prepared[0].heatmap_x[0]),
        float(prepared[0].heatmap_x[-1]),
        float(prepared[0].heatmap_y[0]),
        float(prepared[0].heatmap_y[-1]),
    )
    if not (extent[0] <= 0.0 <= extent[1] and extent[2] <= 0.0 <= extent[3]):
        raise ValueError("heatmap coordinates must contain the real/fake center at (0, 0)")
    scalar_scale = _global_scale(
        [field for frame in prepared for field in (frame.real_heatmap, frame.fake_heatmap)],
        heatmap_scale,
        name="heatmap_scale",
    )
    curve_x_scale = _padded_global_scale(
        [frame.curve_x for frame in prepared],
        None,
        name="curve_x_range",
    )
    curve_scale = _padded_global_scale(
        [field for frame in prepared for field in (frame.real_curve, frame.fake_curve)],
        curve_y_range,
        name="curve_y_range",
        include_zero=True,
    )
    rendered_checkpoints, roles, durations = _render_schedule(
        [frame.checkpoint for frame in prepared], timing
    )
    rendered_images: list[Image.Image] = []
    for index, frame in enumerate(prepared):
        rendered_images.append(
            _render_real_fake_frame(
                frame,
                canvas_size=canvas_size,
                extent=extent,
                heatmap_scale=scalar_scale,
                curve_x_scale=curve_x_scale,
                curve_y_scale=curve_scale,
                title=title,
                scalar_label=scalar_label,
                heatmap_classification_label=heatmap_classification_label,
                curve_label=curve_label,
                role=roles[index],
                final_callout=final_callout,
            )
        )
    rendered_images.append(
        _render_real_fake_frame(
            prepared[-1],
            canvas_size=canvas_size,
            extent=extent,
            heatmap_scale=scalar_scale,
            curve_x_scale=curve_x_scale,
            curve_y_scale=curve_scale,
            title=title,
            scalar_label=scalar_label,
            heatmap_classification_label=heatmap_classification_label,
            curve_label=curve_label,
            role="conclusion",
            final_callout=final_callout,
        )
    )
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "animation_kind": "real_fake_scalar_fields",
        "canvas": {"width": canvas_size[0], "height": canvas_size[1]},
        "input_checkpoints": [frame.checkpoint for frame in prepared],
        "input_checkpoint_count": len(prepared),
        "presentation_hold_frame_count": 1,
        "rendered_frame_count": len(rendered_images),
        "rendered_frames": _frame_metadata(rendered_checkpoints, roles, durations),
        "timing": {
            "orientation_ms": timing.orientation_ms,
            "checkpoint_ms": timing.checkpoint_ms,
            "conclusion_ms": timing.conclusion_ms,
            "loop": 0,
        },
        "scales": {
            "heatmap": scalar_scale.as_metadata(),
            "curve_x": curve_x_scale.as_metadata(),
            "curve_y": curve_scale.as_metadata(),
            "heatmap_extent": list(extent),
            "heatmap_x": prepared[0].heatmap_x.tolist(),
            "heatmap_y": prepared[0].heatmap_y.tolist(),
            "array_coordinate_contract": "row_zero_is_minimum_y",
        },
        "labels": {
            "title": title,
            "scalar": scalar_label,
            "heatmap_classification": heatmap_classification_label,
            "curve": curve_label,
            "final_callout": final_callout,
        },
        "estimands": {
            "heatmap": scalar_estimand,
            "curve": curve_estimand,
        },
        "gif_badge": True,
        "layout_fixed_across_frames": True,
        "checkpoint_panels_synchronized": True,
        "rendering": _rendering_provenance(canvas_size),
    }
    return _write_animation_bundle(rendered_images, durations, output_prefix, metadata)


def render_architecture_cells_animation(
    frames: Sequence[ArchitectureCellsFrame],
    output_prefix: str | Path,
    *,
    resnet_rgb_channel_maxima: ArrayLike,
    vgg_rgb_channel_maxima: ArrayLike,
    jacobian_scale: tuple[float, float] | None = None,
    title: str = "Residual vs non-residual geometry",
    jacobian_label: str = "2D ‖·‖F",
    resnet_jacobian_label: str = "NEW HYBRID · D(T-I)",
    vgg_jacobian_label: str = "NEW HYBRID · DT",
    resnet_jacobian_estimand: str = "2D-plane-restricted ||D(T-I)||_F",
    vgg_jacobian_estimand: str = "2D-plane-restricted ||DT||_F",
    comparison_note: str = (
        "Row-specific operators are shown on one numerical display scale; this is a "
        "descriptive side-by-side view, not a causal architecture ablation."
    ),
    final_callout: str = (
        "Final checkpoint → inspect each row on the shared coefficient grid. ResNet shows "
        "2D D(T-I), VGG shows 2D DT; the side-by-side view is descriptive, not causal."
    ),
    canvas_size: tuple[int, int] = (1440, 900),
    timing: AnimationTiming | None = None,
) -> AnimationArtifacts:
    """Render synchronized ResNet/VGG RGB cells and Jacobian heatmaps.

    The RGB inputs must already have been normalized with fixed, per-model
    channel maxima computed over every rendered checkpoint.  The required
    maxima arguments preserve that preparation contract in the output metadata;
    the renderer cannot recover or independently verify them from normalized
    RGB arrays.
    """

    timing = AnimationTiming() if timing is None else timing
    prepared = tuple(frames)
    _validate_architecture_sequence(prepared)
    canvas_size = _validate_canvas(canvas_size)
    title = _label(title, name="title", max_characters=72)
    jacobian_label = _label(jacobian_label, name="jacobian_label", max_characters=36)
    resnet_jacobian_label = _label(
        resnet_jacobian_label,
        name="resnet_jacobian_label",
        max_characters=36,
    )
    vgg_jacobian_label = _label(
        vgg_jacobian_label,
        name="vgg_jacobian_label",
        max_characters=36,
    )
    resnet_jacobian_estimand = _label(
        resnet_jacobian_estimand,
        name="resnet_jacobian_estimand",
        max_characters=160,
    )
    vgg_jacobian_estimand = _label(
        vgg_jacobian_estimand,
        name="vgg_jacobian_estimand",
        max_characters=160,
    )
    comparison_note = _label(
        comparison_note,
        name="comparison_note",
        max_characters=320,
    )
    final_callout = _label(final_callout, name="final_callout", max_characters=320)
    extent = (
        float(prepared[0].alpha_coordinates[0]),
        float(prepared[0].alpha_coordinates[-1]),
        float(prepared[0].beta_coordinates[0]),
        float(prepared[0].beta_coordinates[-1]),
    )
    resnet_normalizers = _finite_array(
        resnet_rgb_channel_maxima,
        name="resnet_rgb_channel_maxima",
        ndim=1,
    )
    vgg_normalizers = _finite_array(
        vgg_rgb_channel_maxima,
        name="vgg_rgb_channel_maxima",
        ndim=1,
    )
    if resnet_normalizers.shape != (3,) or vgg_normalizers.shape != (3,):
        raise ValueError("RGB channel maxima must each contain the three anchor channels")
    if np.any(resnet_normalizers <= 0) or np.any(vgg_normalizers <= 0):
        raise ValueError("RGB channel maxima must be positive")
    for frame in prepared:
        for anchors in (frame.resnet_anchors, frame.vgg_anchors):
            if anchors is None:
                continue
            alpha = anchors[:, 0]
            beta = anchors[:, 1]
            if (
                np.any(alpha < extent[0])
                or np.any(alpha > extent[1])
                or np.any(beta < extent[2])
                or np.any(beta > extent[3])
            ):
                raise ValueError("all anchor coordinates must lie inside the plane coordinates")
    scalar_scale = _global_scale(
        [field for frame in prepared for field in (frame.resnet_jacobian, frame.vgg_jacobian)],
        jacobian_scale,
        name="jacobian_scale",
    )
    rendered_checkpoints, roles, durations = _render_schedule(
        [frame.checkpoint for frame in prepared], timing
    )
    rendered_images: list[Image.Image] = []
    for index, frame in enumerate(prepared):
        rendered_images.append(
            _render_architecture_frame(
                frame,
                canvas_size=canvas_size,
                extent=extent,
                jacobian_scale=scalar_scale,
                title=title,
                jacobian_label=jacobian_label,
                resnet_jacobian_label=resnet_jacobian_label,
                vgg_jacobian_label=vgg_jacobian_label,
                role=roles[index],
                final_callout=final_callout,
            )
        )
    rendered_images.append(
        _render_architecture_frame(
            prepared[-1],
            canvas_size=canvas_size,
            extent=extent,
            jacobian_scale=scalar_scale,
            title=title,
            jacobian_label=jacobian_label,
            resnet_jacobian_label=resnet_jacobian_label,
            vgg_jacobian_label=vgg_jacobian_label,
            role="conclusion",
            final_callout=final_callout,
        )
    )
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "animation_kind": "architecture_cells_and_jacobians",
        "canvas": {"width": canvas_size[0], "height": canvas_size[1]},
        "input_checkpoints": [frame.checkpoint for frame in prepared],
        "input_checkpoint_count": len(prepared),
        "presentation_hold_frame_count": 1,
        "rendered_frame_count": len(rendered_images),
        "rendered_frames": _frame_metadata(rendered_checkpoints, roles, durations),
        "timing": {
            "orientation_ms": timing.orientation_ms,
            "checkpoint_ms": timing.checkpoint_ms,
            "conclusion_ms": timing.conclusion_ms,
            "loop": 0,
        },
        "scales": {
            "rgb_cells": {
                "display_min": 0.0,
                "display_max": 1.0,
                "source": "caller_prepared_with_declared_global_channel_maxima",
                "normalization_scope": "fixed_across_all_checkpoints_within_architecture",
                "comparability_contract": (
                    "per_architecture_channel_normalization_not_absolute_cross_architecture_scale"
                ),
                "resnet_channel_maxima": resnet_normalizers.tolist(),
                "vgg_channel_maxima": vgg_normalizers.tolist(),
            },
            "jacobian": scalar_scale.as_metadata(),
            "plane_extent": list(extent),
            "alpha_coordinates": prepared[0].alpha_coordinates.tolist(),
            "beta_coordinates": prepared[0].beta_coordinates.tolist(),
            "array_coordinate_contract": "row_zero_is_minimum_y",
        },
        "labels": {
            "title": title,
            "jacobian": jacobian_label,
            "rgb": "SOURCE ANALOGUE · THREE-ANCHOR RGB",
            "resnet_jacobian": resnet_jacobian_label,
            "vgg_jacobian": vgg_jacobian_label,
            "final_callout": final_callout,
        },
        "rgb_estimand": (
            "three frozen-context downstream-logit L2 distances encoded as "
            "per-architecture/channel-normalized RGB; not discovered or stable cells"
        ),
        "jacobian_estimands": {
            "resnet": resnet_jacobian_estimand,
            "vgg": vgg_jacobian_estimand,
        },
        "comparison_note": comparison_note,
        "gif_badge": True,
        "layout_fixed_across_frames": True,
        "checkpoint_panels_synchronized": True,
        "rendering": _rendering_provenance(canvas_size),
    }
    return _write_animation_bundle(rendered_images, durations, output_prefix, metadata)


__all__ = [
    "AnimationArtifacts",
    "AnimationTiming",
    "ArchitectureCellsFrame",
    "RealFakeFrame",
    "render_architecture_cells_animation",
    "render_real_fake_animation",
]
