# Brand assets

The icon is a soul of sunlight striking a solar panel. The logo is the wordmark
with that soul as its "o".

| File                   |                                                                  |
| ---------------------- | ---------------------------------------------------------------- |
| `soular-icon.svg`      | The flat icon. This is the one Home Assistant shows.             |
| `soular-logo.svg`      | The wordmark, for light backgrounds.                             |
| `soular-logo-dark.svg` | The wordmark, for dark backgrounds.                              |
| `glow/`                | A glowing, painterly render of the icon, kept as an alternative. |

Home Assistant 2026.8 and later read brand images straight from
`custom_components/soular/brand/`, so they ship with the integration.

## Regenerating

Both are generated, so change the generator rather than the files:

```bash
uv run python -m tools.brand_vector --png custom_components/soular/brand
```

writes the SVGs here and renders the icon and logo PNGs Home Assistant uses. It
needs `rsvg-convert` (`brew install librsvg`).

```bash
uv run python -m tools.brand_icon
```

renders the glow version into `glow/`. To ship it instead of the flat icon, add
`--out custom_components/soular/brand` and delete the flat `icon*.png` first.
