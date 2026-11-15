# Brand assets

The icon is a soul of sunlight striking a solar panel. The logo is the wordmark
with that soul as its "o".

Home Assistant 2026.8 and later read brand images straight from
`custom_components/soular/brand/`, so they ship with the integration. What ships
there is the glowing set: icon and logo, each in a light-theme and a dark-theme
variant.

The flat, iconographic set here is the alternative:

| File                   |                                               |
| ---------------------- | --------------------------------------------- |
| `soular-icon.svg`      | The flat icon, as a vector.                   |
| `soular-logo.svg`      | The flat wordmark, for light backgrounds.     |
| `soular-logo-dark.svg` | The flat wordmark, for dark backgrounds.      |
| `flat/`                | The flat set rendered as Home Assistant PNGs. |

## Regenerating

Both sets are generated, so change the generator rather than the files.

```bash
uv run python -m tools.brand_icon
```

renders the glowing icon and logo into `custom_components/soular/brand/`.

```bash
uv run python -m tools.brand_vector --png docs/brand/flat
```

writes the flat SVGs here and renders them into `flat/`. It needs `rsvg-convert`
(`brew install librsvg`). To ship the flat set instead, copy `flat/*.png` over the
files in `custom_components/soular/brand/` and delete its `dark_icon*.png`.
