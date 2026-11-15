"""Write Soular's flat, vector brand icon and logo.

The iconographic counterpart to :mod:`tools.brand_icon`'s glowing render: the
same scene in bold flat shapes and a small palette, so it stays legible at 32
pixels and scales to anything as an SVG.

    uv run python -m tools.brand_vector

writes ``soular-icon.svg`` and the wordmark -- ``soular-logo.svg``, its o the
soul, and ``soular-logo-dark.svg`` for dark themes -- into ``docs/brand/``. With
``--png DIR`` it also renders Home Assistant brand PNGs into DIR, using
``rsvg-convert`` (librsvg). The glowing set from :mod:`tools.brand_icon` is the
one that ships; this flat set is the alternative, rendered into
``docs/brand/flat/``.
"""

import argparse
import math
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = ROOT / "docs" / "brand"
SIZE = 256

NAVY_FAR = "#24427a"
NAVY_NEAR = "#162a52"
FRAME = "#c4cedb"
FRAME_EDGE = "#8794a8"
ORANGE = "#f07a1a"
GOLD = "#ffb627"
PALE = "#ffe9a8"
WHITE = "#fffaf0"
TEXT_LIGHT_THEME = "#162a52"
TEXT_DARK_THEME = "#eef2f8"

# Panel corners in the 256-unit viewBox, y down: far-left, far-right,
# near-right, near-left. The far edge is shorter, so it reads as tilted.
PANEL = ((40.0, 158.0), (190.0, 146.0), (238.0, 228.0), (16.0, 244.0))
COLUMNS, ROWS = 4, 3
IMPACT_UV = (0.47, 0.30)
# The soul's tail leans this many degrees right of straight up.
TAIL_LEAN_DEG = 26.0
TAIL_SCALE = 0.8


def lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    """Interpolate between two points."""
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def panel_point(u: float, v: float) -> tuple[float, float]:
    """Map panel coordinates (0..1 across, 0..1 far to near) into the viewBox."""
    far_left, far_right, near_right, near_left = PANEL
    return lerp(lerp(far_left, far_right, u), lerp(near_left, near_right, u), v)


def points(coords: list[tuple[float, float]]) -> str:
    """Format points for an SVG polygon."""
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)


def panel() -> str:
    """Draw the panel: a frame and flat cells shaded far to near."""
    frame = (
        f'<polygon points="{points(list(PANEL))}" fill="{FRAME}" stroke="{FRAME_EDGE}" '
        'stroke-width="3" stroke-linejoin="round"/>'
    )
    parts = [frame]
    inset, gap = 0.045, 0.018
    for row in range(ROWS):
        colour = NAVY_FAR if row == 0 else NAVY_NEAR if row == ROWS - 1 else "#1d3666"
        for col in range(COLUMNS):
            u0 = inset + (1 - 2 * inset) * col / COLUMNS + gap / 2
            u1 = inset + (1 - 2 * inset) * (col + 1) / COLUMNS - gap / 2
            v0 = inset + (1 - 2 * inset) * row / ROWS + gap / 2
            v1 = inset + (1 - 2 * inset) * (row + 1) / ROWS - gap / 2
            cell = [panel_point(u0, v0), panel_point(u1, v0), panel_point(u1, v1), panel_point(u0, v1)]
            parts.append(f'<polygon points="{points(cell)}" fill="{colour}"/>')
    return "\n    ".join(parts)


def splash(impact: tuple[float, float]) -> str:
    """Draw light pooling on the panel, foreshortened, and a crown of sparks."""
    ix, iy = impact
    far_left, far_right = PANEL[0], PANEL[1]
    slope = math.degrees(math.atan2(far_right[1] - far_left[1], far_right[0] - far_left[0]))
    parts = [
        f'<g transform="translate({ix:.1f},{iy:.1f}) rotate({slope:.1f})">',
        f'  <ellipse rx="54" ry="15" fill="{ORANGE}"/>',
        f'  <ellipse rx="42" ry="11" fill="{GOLD}"/>',
        f'  <ellipse rx="26" ry="6.5" fill="{PALE}"/>',
        "</g>",
    ]
    # Spikes thrown up and out, fanned either side of the incoming soul.
    for angle, length, width in (
        (-158, 62, 11),
        (-136, 66, 11),
        (-112, 52, 9),
        (-30, 54, 9),
        (-12, 60, 10),
    ):
        a = math.radians(angle)
        # Each spike springs from the splash's rim and flies up and out.
        bx, by = ix + math.cos(a) * 36, iy + math.sin(a) * 9
        tx, ty = ix + math.cos(a) * length * 1.15, iy + math.sin(a) * length
        nx, ny = -math.sin(a) * width / 2, math.cos(a) * width / 2
        spike = [(bx + nx, by + ny), (tx, ty), (bx - nx, by - ny)]
        parts.append(
            f'<polygon points="{points(spike)}" fill="{GOLD}" stroke="{ORANGE}" '
            'stroke-width="1.5" stroke-linejoin="round"/>'
        )
    for ex, ey, r in ((-88, -50, 5.0), (-60, -82, 4.0), (-96, -16, 3.5), (78, -66, 4.5), (98, -30, 3.5)):
        parts.append(f'<circle cx="{ix + ex:.1f}" cy="{iy + ey:.1f}" r="{r}" fill="{GOLD}"/>')
    return "\n    ".join(parts)


def soul_shape(lean_deg: float, tail_scale: float = 1.0) -> str:
    """Draw the soul in its own frame: head at the origin, tail up the negative y axis, then leaned."""
    outer = (
        "M -27 2 C -33 -34, -22 -66, -12 -96 C -16 -84, -22 -70, -30 -66 "
        "C -20 -98, -8 -140, 2 -186 C 6 -150, 14 -124, 22 -104 "
        "C 22 -116, 20 -128, 16 -140 C 30 -110, 34 -50, 27 2 Z"
    )
    middle = "M -18 0 C -22 -36, -12 -76, -2 -140 C 4 -104, 12 -80, 16 -62 C 20 -40, 22 -18, 18 0 Z"
    core = "M -9 -4 C -10 -34, -4 -66, 0 -98 C 4 -66, 10 -34, 9 -4 Z"
    mirror = -1 if lean_deg < 0 else 1
    return f"""<g transform="rotate({lean_deg:.1f}) scale({mirror},1)">
        <g transform="scale({TAIL_SCALE * tail_scale:.3f})">
          <path d="{outer}" fill="{ORANGE}"/>
          <path d="{middle}" fill="{GOLD}"/>
          <path d="{core}" fill="{PALE}"/>
        </g>
        <circle r="25" fill="{GOLD}"/>
        <circle r="19" fill="{PALE}"/>
        <circle r="11" fill="{WHITE}"/>
      </g>"""


def soul(impact: tuple[float, float]) -> str:
    """Draw the soul heading down onto the panel, its tail streaming back up its path."""
    ix, iy = impact
    return f"""<g transform="translate({ix:.1f},{iy - 16:.1f})">
      {soul_shape(TAIL_LEAN_DEG)}
    </g>"""


def build() -> str:
    """Return the whole icon as an SVG document."""
    impact = panel_point(*IMPACT_UV)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">
  <title>Soular</title>
  <g>
    {panel()}
  </g>
  <g>
    {splash(impact)}
  </g>
  {soul(impact)}
</svg>
"""


def glyphs(text: str, size: float) -> tuple[str, tuple[float, float, float, float]]:
    """Outline ``text`` as SVG path data (y down, baseline at 0) and return it with its bounds.

    Outlined rather than set as text, so the logo does not depend on which
    fonts the viewer has. DejaVu Sans ships with matplotlib under a free licence.
    """
    from matplotlib.font_manager import FontProperties  # noqa: PLC0415
    from matplotlib.path import Path as MplPath  # noqa: PLC0415
    from matplotlib.textpath import TextPath  # noqa: PLC0415

    path = TextPath((0, 0), text, size=size, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    commands: list[str] = []
    for vertices, code in path.iter_segments(curves=True, simplify=False):
        pts = [(vertices[i], -vertices[i + 1]) for i in range(0, len(vertices), 2)]
        coords = " ".join(f"{x:.2f} {y:.2f}" for x, y in pts)
        if code == MplPath.MOVETO:
            commands.append(f"M {coords}")
        elif code == MplPath.LINETO:
            commands.append(f"L {coords}")
        elif code == MplPath.CURVE3:
            commands.append(f"Q {coords}")
        elif code == MplPath.CURVE4:
            commands.append(f"C {coords}")
        elif code == MplPath.CLOSEPOLY:
            commands.append("Z")
    extents = path.get_extents()
    return " ".join(commands), (extents.x0, -extents.y1, extents.x1, -extents.y0)


def advance(text: str, size: float) -> float:
    """Return the horizontal advance of ``text``: where the next glyph would start."""
    from matplotlib.font_manager import FontProperties  # noqa: PLC0415
    from matplotlib.textpath import TextToPath  # noqa: PLC0415

    width, _, _ = TextToPath().get_text_width_height_descent(
        text, FontProperties(family="DejaVu Sans", weight="bold", size=size), ismath=False
    )
    return float(width)


def build_logo(text_colour: str) -> tuple[str, float, float]:
    """Return the wordmark -- "soular", its o the soul -- and its width and height."""
    size = 200.0
    s_path, _ = glyphs("s", size)
    ular_path, ular_box = glyphs("ular", size)
    _, o_box = glyphs("o", size)
    ular_x = advance("so", size)
    o_left = advance("s", size)
    # The soul's head takes the o's place and size; its tail streams up and
    # back to the left, over the short s, clear of the tall l.
    o_cx = o_left + (o_box[0] + o_box[2]) / 2
    o_cy = (o_box[1] + o_box[3]) / 2
    scale = (o_box[3] - o_box[1]) / 2 / 25.0 * 0.96
    lean = -(TAIL_LEAN_DEG + 22)
    tail = 0.78

    tail_reach = 186 * TAIL_SCALE * tail * scale
    top = min(ular_box[1], o_cy - tail_reach * math.cos(math.radians(lean)) - 6 * scale)
    left = min(0.0, o_cx + tail_reach * math.sin(math.radians(lean)) - 30 * scale)
    right = ular_x + ular_box[2]
    bottom = max(ular_box[3], o_box[3])
    pad = 0.04 * size
    width, height = right - left + 2 * pad, bottom - top + 2 * pad
    dx, dy = pad - left, pad - top

    root = f'viewBox="0 0 {width:.1f} {height:.1f}" width="{width:.0f}" height="{height:.0f}"'
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" {root}>
  <title>Soular</title>
  <g transform="translate({dx:.1f},{dy:.1f})" fill="{text_colour}">
    <path d="{s_path}"/>
    <path transform="translate({ular_x:.1f},0)" d="{ular_path}"/>
    <g transform="translate({o_cx:.1f},{o_cy:.1f}) scale({scale:.3f})">
      {soul_shape(lean, tail)}
    </g>
  </g>
</svg>
"""
    return svg, width, height


def rsvg(svg: Path, png: Path, width: int, height: int) -> None:
    """Render one SVG to a PNG of the given size with rsvg-convert."""
    tool = shutil.which("rsvg-convert")
    if tool is None:
        msg = "rsvg-convert not found; install librsvg (brew install librsvg)"
        raise SystemExit(msg)
    subprocess.run([tool, "-w", str(width), "-h", str(height), "-o", str(png), str(svg)], check=True)  # noqa: S603
    print(f"wrote {png}")


def main() -> None:
    """Write the SVGs, and the Home Assistant brand PNGs when asked."""
    parser = argparse.ArgumentParser(description="Write Soular's vector brand icon and logo.")
    parser.add_argument("--svg-dir", type=Path, default=SVG_DIR)
    parser.add_argument("--png", type=Path, help="also render the brand PNGs into this directory")
    args = parser.parse_args()
    args.svg_dir.mkdir(parents=True, exist_ok=True)

    icon = args.svg_dir / "soular-icon.svg"
    icon.write_text(build())
    print(f"wrote {icon}")
    logos = {}
    for name, colour in (("soular-logo.svg", TEXT_LIGHT_THEME), ("soular-logo-dark.svg", TEXT_DARK_THEME)):
        svg, width, height = build_logo(colour)
        (args.svg_dir / name).write_text(svg)
        logos[name] = width / height
        print(f"wrote {args.svg_dir / name}")

    if args.png:
        args.png.mkdir(parents=True, exist_ok=True)
        for suffix, scale in (("", 1), ("@2x", 2)):
            rsvg(icon, args.png / f"icon{suffix}.png", 256 * scale, 256 * scale)
            # Home Assistant wants a logo's shortest side between 128 and 256 pixels.
            for svg_name, png_name in (("soular-logo.svg", "logo"), ("soular-logo-dark.svg", "dark_logo")):
                height = 128 * scale
                width = round(height * logos[svg_name])
                rsvg(args.svg_dir / svg_name, args.png / f"{png_name}{suffix}.png", width, height)


if __name__ == "__main__":
    main()
