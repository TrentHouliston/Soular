"""Render Soular's brand icon and logo: a soul of sunlight striking a solar panel.

A white-hot sun trailing a tail of flame comes down onto a tilted panel, and
its light splashes across the cells in a crown of sparks. Procedural, so it can
be regenerated or tweaked rather than hand-edited:

    uv run python -m tools.brand_icon

writes the icon -- ``icon.png`` (for light themes: deeper amber, more opaque,
so it holds up on white) and ``dark_icon.png`` (the luminous version) -- and the
wordmark logo, its o the soul, as ``logo.png`` and ``dark_logo.png``, each at 1x
and 2x, into ``custom_components/soular/brand/``. Home Assistant 2026.8+ serves
those for custom integrations directly.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "custom_components" / "soular" / "brand"
SUPERSAMPLE = 4
SEED = 7

# The panel, as a perspective quad in [-1, 1] icon space (y down): its far
# edge is shorter than its near edge, so it reads as tilted up to the sky.
PANEL = ((-0.64, 0.26), (0.50, 0.17), (0.82, 0.76), (-0.80, 0.90))  # far-left, far-right, near-right, near-left
PANEL_COLUMNS, PANEL_ROWS = 5, 3
# Where on the panel (u across, v down, both 0..1) the soul strikes.
IMPACT_UV = (0.46, 0.36)
# Direction the soul travels, in icon space: down and to the left.
HEADING = np.array([-0.45, 1.0])
# How much larger than its unit design the soul is drawn.
SOUL_SCALE = 1.45
# In the logo the tail leans back to the left, over the s, by this many degrees.
LOGO_TAIL_LEAN_DEG = 50.0
TEXT_LIGHT_THEME = (22, 42, 82)
TEXT_DARK_THEME = (238, 242, 248)


@dataclass(frozen=True)
class Palette:
    """Colours for one theme variant."""

    light: tuple[tuple[float, tuple[int, int, int]], ...]  # glow, faintest to core
    alpha_gain: float
    cell: tuple[tuple[int, int, int], tuple[int, int, int]]  # panel cell, far to near
    gap: tuple[int, int, int]
    frame: tuple[int, int, int]


DARK = Palette(
    light=(
        (0.00, (180, 92, 20)),
        (0.30, (232, 160, 52)),
        (0.55, (255, 210, 110)),
        (0.80, (255, 240, 190)),
        (1.00, (255, 255, 248)),
    ),
    alpha_gain=1.15,
    cell=((34, 58, 104), (18, 32, 66)),
    gap=(86, 116, 160),
    frame=(170, 182, 198),
)

LIGHT = Palette(
    light=(
        (0.00, (170, 50, 0)),
        (0.30, (222, 110, 10)),
        (0.55, (247, 166, 30)),
        (0.80, (255, 214, 110)),
        (1.00, (255, 248, 220)),
    ),
    alpha_gain=1.6,
    cell=((30, 52, 96), (14, 26, 56)),
    gap=(70, 98, 140),
    frame=(120, 132, 150),
)


def value_noise(shape: tuple[int, int], cells: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth noise in [0, 1]: a random lattice, bicubically upsampled."""
    lattice = rng.random((cells + 1, cells + 1))
    image = Image.fromarray((lattice * 255).astype(np.uint8)).resize((shape[1], shape[0]), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float64) / 255.0


def fbm(shape: tuple[int, int], rng: np.random.Generator, octaves: int = 5, base: int = 3) -> np.ndarray:
    """Fractal noise centred on zero, roughly in [-1, 1]."""
    total = np.zeros(shape)
    amplitude, norm = 1.0, 0.0
    for octave in range(octaves):
        total += amplitude * (value_noise(shape, base * 2**octave, rng) - 0.5) * 2
        norm += amplitude
        amplitude *= 0.5
    return total / norm


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    """Hermite step from 0 at ``edge0`` to 1 at ``edge1``."""
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def panel_point(u: float, v: float) -> np.ndarray:
    """Map panel coordinates (0..1 across and down) into icon space."""
    far_left, far_right, near_right, near_left = (np.array(c) for c in PANEL)
    far = far_left + (far_right - far_left) * u
    near = near_left + (near_right - near_left) * u
    return far + (near - far) * v


def draw_panel(n: int, palette: Palette) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the panel; return its RGB, its coverage, and each pixel's (u, v) on it."""
    scale = (n - 1) / 2.0

    def px(point: np.ndarray) -> tuple[float, float]:
        return (float((point[0] + 1) * scale), float((point[1] + 1) * scale))

    image = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.polygon([px(np.array(c)) for c in PANEL], fill=(*palette.frame, 255))
    inset = 0.035
    for row in range(PANEL_ROWS):
        for col in range(PANEL_COLUMNS):
            u0 = inset + (1 - 2 * inset) * col / PANEL_COLUMNS
            u1 = inset + (1 - 2 * inset) * (col + 1) / PANEL_COLUMNS
            v0 = inset + (1 - 2 * inset) * row / PANEL_ROWS
            v1 = inset + (1 - 2 * inset) * (row + 1) / PANEL_ROWS
            gap = 0.012
            corners = [(u0 + gap, v0 + gap), (u1 - gap, v0 + gap), (u1 - gap, v1 - gap), (u0 + gap, v1 - gap)]
            shade = (row + 0.5) / PANEL_ROWS
            colour = tuple(int(a + (b - a) * shade) for a, b in zip(*palette.cell, strict=True))
            draw.polygon([px(panel_point(u, v)) for u, v in corners], fill=(*colour, 255))
    # Busbars: thin lines across each cell, the detail that says "solar cell".
    for row in range(PANEL_ROWS):
        for k in (1, 2):
            v = inset + (1 - 2 * inset) * (row + k / 3) / PANEL_ROWS
            draw.line(
                [px(panel_point(inset, v)), px(panel_point(1 - inset, v))],
                fill=(*palette.gap, 255),
                width=max(1, n // 400),
            )

    rgba = np.asarray(image, dtype=np.float64) / 255.0
    rgb, coverage = rgba[..., :3], rgba[..., 3]

    # Each pixel's panel coordinates, by inverting the quad's bilinear map
    # approximately: good enough to place light on the panel's surface.
    y, x = np.mgrid[0:n, 0:n] / (n - 1) * 2.0 - 1.0
    u = np.zeros_like(x) + 0.5
    v = np.zeros_like(x) + 0.5
    for _ in range(8):  # Newton's method on the bilinear map
        p = panel_point_grid(u, v)
        du = (panel_point_grid(u + 1e-3, v) - p) / 1e-3
        dv = (panel_point_grid(u, v + 1e-3) - p) / 1e-3
        rx, ry = x - p[0], y - p[1]
        det = du[0] * dv[1] - du[1] * dv[0]
        u += (rx * dv[1] - ry * dv[0]) / det
        v += (ry * du[0] - rx * du[1]) / det
    return rgb, coverage, np.stack([u, v])


def panel_point_grid(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised :func:`panel_point` over arrays of (u, v)."""
    far_left, far_right, near_right, near_left = (np.array(c)[:, None, None] for c in PANEL)
    far = far_left + (far_right - far_left) * u
    near = near_left + (near_right - near_left) * u
    return far + (near - far) * v


def soul(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, tail_length: float = 1.0, glow_weight: float = 0.55
) -> np.ndarray:
    """Draw the soul in its own frame: head at the origin, tail streaming up (negative y).

    ``tail_length`` scales the tail without changing the head; ``glow_weight``
    sets how far its soft glow spills.
    """
    shape = (x.shape[0], x.shape[1])
    sway_field = fbm(shape, rng, base=2)
    lick = fbm(shape, rng, base=5, octaves=5)
    flicker = fbm(shape, rng, base=9, octaves=3)
    head_y = y
    y = y / tail_length

    def tongue(offset: float, top: float, bottom: float, girth: float, lean: float) -> np.ndarray:
        t = np.clip((bottom - y) / (bottom - top), 0.0, 1.0)
        centre = offset + lean * t**1.5 + sway_field * 0.08 * t + 0.025 * np.sin(t * 6.0) * t
        # Widest at the head and tapering back along the path, like a comet,
        # so it reads as falling onto the panel rather than burning on it.
        width = girth * (1 - t) ** 1.1 * smoothstep(0.0, 0.06, t) + 1e-3
        d = np.abs(x - centre) / (width * (1 + 0.45 * lick * t))
        envelope = smoothstep(0.0, 0.05, t) * (1 - smoothstep(0.55, 1.0, t))
        return np.exp(-(d**2) * 2.0) * envelope

    tail = tongue(0.0, -1.00, 0.10, 0.20, 0.0)
    tail = np.maximum(tail, 0.7 * tongue(-0.05, -0.70, 0.06, 0.10, -0.10))
    tail = np.maximum(tail, 0.65 * tongue(0.05, -0.78, 0.06, 0.09, 0.09))
    heart = tongue(0.0, -0.72, 0.06, 0.055, 0.0)
    tail *= np.clip(1 + 0.7 * flicker * (1 - tail), 0, None)

    # Motion streaks: fine lines trailing back along the path.
    streaks = np.zeros_like(x)
    for _ in range(9):
        offset = rng.normal(0.0, 0.07)
        length = rng.uniform(0.35, 0.85)
        start = rng.uniform(0.02, 0.15)
        t = np.clip(-(y + start) / length, 0, 1)
        streaks += (
            np.exp(-(((x - offset * (1 + t)) / 0.008) ** 2))
            * smoothstep(0.0, 0.05, t)
            * (1 - t) ** 1.5
            * rng.uniform(0.3, 0.7)
        )

    r = np.sqrt(x**2 + head_y**2)
    head = np.exp(-((r / 0.11) ** 4)) * 1.0 + np.exp(-((r / 0.06) ** 2)) * 0.6
    glow = np.exp(-((r / 0.22) ** 2))
    return tail * 0.80 + heart * 0.85 + streaks + head * 1.2 + glow * glow_weight


def colourise(light: np.ndarray, palette: Palette) -> tuple[np.ndarray, np.ndarray]:
    """Turn light intensity into colour and coverage, hotter towards white."""
    level = np.clip(light, 0, 1.5) / 1.5
    alpha = np.clip(light * palette.alpha_gain, 0, 1) ** 0.85
    positions = np.array([stop[0] for stop in palette.light])
    colours = np.array([stop[1] for stop in palette.light], dtype=np.float64) / 255.0
    rgb = np.stack([np.interp(level**0.7, positions, colours[:, c]) for c in range(3)], axis=-1)
    return rgb, alpha


def render(size: int, palette: Palette) -> Image.Image:
    """Render one RGBA icon at ``size`` pixels square."""
    rng = np.random.default_rng(SEED)
    n = size * SUPERSAMPLE
    y, x = np.mgrid[0:n, 0:n] / (n - 1) * 2.0 - 1.0  # [-1, 1], y down

    panel_rgb, panel_alpha, (pu, pv) = draw_panel(n, palette)
    impact = panel_point(*IMPACT_UV)

    # The soul, rotated so its tail streams back along its path. Its head sits
    # just above the panel, flattened slightly where it meets the surface.
    heading = HEADING / np.linalg.norm(HEADING)
    along = (x - impact[0]) * heading[0] + (y - impact[1]) * heading[1]
    across = (x - impact[0]) * heading[1] - (y - impact[1]) * heading[0]
    light = soul(across / SOUL_SCALE, (along + 0.07) / SOUL_SCALE, rng)

    # The splash: light pooling on the panel around the impact, foreshortened
    # by the panel's tilt, with a bright ripple ring.
    du = (pu - IMPACT_UV[0]) * 1.0
    dv = (pv - IMPACT_UV[1]) * 1.6
    ground = np.sqrt(du**2 + dv**2)
    pool = np.exp(-((ground / 0.16) ** 2)) * 1.1 + np.exp(-((ground / 0.42) ** 2)) * 0.45
    ripple = np.exp(-(((ground - 0.23) / 0.022) ** 2)) * 0.55
    on_panel = panel_alpha * ((pu > 0) & (pu < 1) & (pv > 0) & (pv < 1))
    light += (pool + ripple) * on_panel

    # A crown of sparks thrown up and out of the impact.
    angle = np.arctan2(y - impact[1], x - impact[0])
    dist = np.sqrt((x - impact[0]) ** 2 + (y - impact[1]) ** 2)
    spikes = np.zeros((n, n))
    for _ in range(13):
        theta = rng.uniform(-np.pi * 0.92, -np.pi * 0.08)  # upward half
        length = rng.uniform(0.22, 0.46)
        width = rng.uniform(0.010, 0.020)
        offset = np.abs(np.sin(angle - theta)) * dist
        along_spike = np.cos(angle - theta) * dist
        spikes += (
            np.exp(-((offset / width) ** 2))
            * smoothstep(0.05, 0.10, along_spike)
            * (1 - smoothstep(length * 0.5, length, along_spike))
            * rng.uniform(0.7, 1.1)
        )
    light += spikes

    # Embers flung further out.
    for _ in range(12):
        theta = rng.uniform(-np.pi * 0.90, -np.pi * 0.10)
        reach = rng.uniform(0.36, 0.70)
        ex, ey = impact[0] + np.cos(theta) * reach, impact[1] + np.sin(theta) * reach
        radius = rng.uniform(0.010, 0.019)
        light += rng.uniform(0.5, 0.95) * np.exp(-(((x - ex) ** 2 + (y - ey) ** 2) / radius**2))

    # Fade the light to nothing before the edge of the square.
    light *= 1 - smoothstep(0.86, 0.99, np.maximum(np.abs(x), np.abs(y)))
    light = np.clip(light, 0, None)

    light_rgb, light_alpha = colourise(light, palette)

    # Light over the panel: screen-blend where there is panel, plain light elsewhere.
    screen = 1 - (1 - panel_rgb) * (1 - light_rgb * light_alpha[..., None])
    rgb = np.where(panel_alpha[..., None] > 0, screen, light_rgb)
    alpha = np.maximum(panel_alpha, light_alpha)
    rgba = np.dstack([rgb * 255, alpha * 255]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA").resize((size, size), Image.Resampling.LANCZOS)


def render_logo(height: int, palette: Palette, text_colour: tuple[int, int, int]) -> Image.Image:
    """Render the wordmark "soular", its o the glowing soul, ``height`` pixels tall.

    The soul's head takes the o's place and size; its tail streams up and back
    to the left, over the short s and clear of the tall l.
    """
    from matplotlib import font_manager  # noqa: PLC0415
    from PIL import ImageFont  # noqa: PLC0415

    rng = np.random.default_rng(SEED)
    font_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    font_size = 160 * SUPERSAMPLE
    font = ImageFont.truetype(font_path, font_size)

    # Layout, relative to the baseline at the s's origin.
    o_left = font.getlength("s")
    ular_x = font.getlength("so")
    o_box = font.getbbox("o", anchor="ls")
    ular_box = font.getbbox("ular", anchor="ls")
    radius = (o_box[3] - o_box[1]) / 2
    centre = (o_left + (o_box[0] + o_box[2]) / 2, (o_box[1] + o_box[3]) / 2)
    unit = radius / 0.17  # pixels per soul unit: the head fills the o's counter
    tail_length = 1.05
    lean = np.radians(LOGO_TAIL_LEAN_DEG)
    reach = (tail_length * 1.0 + 0.1) * unit
    tip = (centre[0] - np.sin(lean) * reach, centre[1] - np.cos(lean) * reach)

    pad = 0.30 * unit
    left = min(0.0, tip[0] - pad)
    top = min(float(ular_box[1]), tip[1] - pad)
    right = ular_x + ular_box[2] + pad * 0.4
    bottom = max(float(ular_box[3]), centre[1] + radius) + pad * 0.6
    width_px, height_px = int(np.ceil(right - left)), int(np.ceil(bottom - top))
    origin = (-left, -top)  # where the baseline origin lands on the canvas

    text = Image.new("L", (width_px, height_px), 0)
    draw = ImageDraw.Draw(text)
    draw.text(origin, "s", font=font, fill=255, anchor="ls")
    draw.text((origin[0] + ular_x, origin[1]), "ular", font=font, fill=255, anchor="ls")
    text_alpha = np.asarray(text, dtype=np.float64) / 255.0
    # The o itself, drawn in light, so it still reads as a letter around the soul.
    o_glyph = Image.new("L", (width_px, height_px), 0)
    ImageDraw.Draw(o_glyph).text((origin[0] + o_left, origin[1]), "o", font=font, fill=255, anchor="ls")
    o_alpha = np.asarray(o_glyph, dtype=np.float64) / 255.0

    y, x = np.mgrid[0:height_px, 0:width_px].astype(np.float64)
    dx, dy = x - (origin[0] + centre[0]), y - (origin[1] + centre[1])
    # Into the soul's frame: its tail runs up the negative local y axis.
    local_y = (dx * np.sin(lean) + dy * np.cos(lean)) / unit
    local_x = (dx * np.cos(lean) - dy * np.sin(lean)) / unit
    light = soul(local_x, local_y, rng, tail_length=tail_length, glow_weight=0.25)
    light += o_alpha * 0.75

    # A few embers shed from the tail.
    for _ in range(6):
        along = rng.uniform(0.25, 0.85) * tail_length
        aside = rng.normal(0.0, 0.16)
        r = rng.uniform(0.012, 0.022)
        light += rng.uniform(0.5, 0.9) * np.exp(-(((local_x - aside) ** 2 + (local_y + along) ** 2) / r**2))
    # Fade before the canvas edge, so nothing is cut off.
    edge = np.minimum.reduce([x, y, width_px - 1 - x, height_px - 1 - y]) / (0.12 * unit)
    light = np.clip(light * smoothstep(0.0, 1.0, edge), 0, None)

    light_rgb, light_alpha = colourise(light, palette)
    text_rgb = np.array(text_colour, dtype=np.float64) / 255.0
    alpha = light_alpha + text_alpha * (1 - light_alpha)
    rgb = (light_rgb * light_alpha[..., None] + text_rgb * (text_alpha * (1 - light_alpha))[..., None]) / np.maximum(
        alpha, 1e-6
    )[..., None]
    rgba = np.dstack([rgb * 255, alpha * 255]).clip(0, 255).astype(np.uint8)
    width = round(height * width_px / height_px)
    return Image.fromarray(rgba, "RGBA").resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    """Write the icon and logo variants."""
    parser = argparse.ArgumentParser(description="Render Soular's brand icons.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for prefix, palette in (("", LIGHT), ("dark_", DARK)):
        for suffix, size in (("", 256), ("@2x", 512)):
            path = args.out / f"{prefix}icon{suffix}.png"
            render(size, palette).save(path, optimize=True)
            print(f"wrote {path}")
    # Home Assistant wants a logo's shortest side between 128 and 256 pixels.
    for prefix, palette, text in (("", LIGHT, TEXT_LIGHT_THEME), ("dark_", DARK, TEXT_DARK_THEME)):
        for suffix, height in (("", 128), ("@2x", 256)):
            path = args.out / f"{prefix}logo{suffix}.png"
            render_logo(height, palette, text).save(path, optimize=True)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
