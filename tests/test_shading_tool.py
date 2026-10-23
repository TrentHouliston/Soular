"""Tests for the shading-map builder.

The builder is checked end to end against a sky whose answer is known: a wall
that blocks the sun below 25 degrees across the western quarter, and open sky
everywhere else. Production is generated through the core's own optics, so the
only thing the builder has to recover is the wall.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from custom_components.soular.core.clearsky import clear_sky
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.irradiance import plane_of_array
from custom_components.soular.core.shading import from_npz
from tools.shading import grid
from tools.shading.cli import _npz_bytes
from tools.shading.horizon import fit, read_horizon
from tools.shading.samples import build_samples, read_power
from tools.site import load_site

SITE = """
[site]
latitude = -33.12
longitude = 151.53
timezone = "Australia/Sydney"

[arrays.roof]
azimuth = 354
tilt = 25
dc_watts = 6000
"""

WALL_AZIMUTH = (240.0, 330.0)
WALL_HEIGHT_DEG = 25.0


def _blocked(azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    return (azimuth >= WALL_AZIMUTH[0]) & (azimuth <= WALL_AZIMUTH[1]) & (elevation < WALL_HEIGHT_DEG)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pd.DataFrame, pd.Series]:
    """Generate a year of clear days behind the wall, and a satellite that agrees it was clear."""
    path = tmp_path_factory.mktemp("site") / "site.toml"
    path.write_text(SITE)
    site = load_site(path, ac_limit_w=1.0)
    array = site.arrays[0]

    index = pd.date_range("2025-01-01", "2025-12-31 23:55", freq="5min", tz="UTC")
    times = index.tz_localize(None).to_numpy().astype("datetime64[s]")
    geometry = solar_geometry(times, site.spec)
    sky = clear_sky(times, site.spec, geometry)
    poa = plane_of_array(array, site.spec, geometry, sky.ghi, sky.dni, sky.dhi)
    beam = np.where(_blocked(geometry.azimuth, geometry.apparent_elevation), 0.0, poa.poa_beam)
    # The same irradiance-driven cell-temperature derate the builder assumes.
    derate = 1.0 + array.gamma_pdc * 30.0 * (poa.poa_beam + poa.poa_diffuse) / 800.0
    watts = 0.85 * derate * array.dc_capacity_w * (beam + poa.poa_diffuse) / 1000.0
    power = pd.DataFrame({"roof": watts}, index=index)
    return path, power, pd.Series(sky.ghi, index=index)


def test_power_csv_is_averaged_onto_the_grid(tmp_path: Path) -> None:
    """One-minute exports describe the same interval as five-minute ones."""
    site_path = tmp_path / "site.toml"
    site_path.write_text(SITE)
    csv = tmp_path / "power.csv"
    csv.write_text(
        "time,roof,soc\n"
        + "".join(f"2026-01-01T00:0{minute}:00Z,{100 * minute},50\n" for minute in range(5))
        + "2026-01-01T00:05:00+00:00,1000,50\n"
    )
    power = read_power(csv, load_site(site_path, ac_limit_w=1.0))
    assert list(power.columns) == ["roof", "soc"]
    assert power["roof"].tolist() == [200.0, 1000.0]


def test_a_missing_array_column_is_named(tmp_path: Path) -> None:
    """A CSV that does not cover the site file fails loudly, naming the gap."""
    site_path = tmp_path / "site.toml"
    site_path.write_text(SITE)
    csv = tmp_path / "power.csv"
    csv.write_text("time,east\n2026-01-01T00:00:00Z,0\n")
    with pytest.raises(ValueError, match="roof"):
        read_power(csv, load_site(site_path, ac_limit_w=1.0))


def test_the_wall_is_recovered(synthetic: tuple[Path, pd.DataFrame, pd.Series]) -> None:
    """Blocked behind the wall, open above it and everywhere else."""
    path, power, satellite = synthetic
    site = load_site(path, ac_limit_w=1.0)
    samples = build_samples(site, power, satellite).arrays["roof"]
    train = samples.loc[~samples["holdout"].to_numpy(dtype=bool)]

    horizon = fit(train)
    field = grid.build(train, horizon)
    loaded = from_npz(_npz_bytes({"roof": field}), "roof")

    def t(azimuth: float, elevation: float) -> float:
        return float(loaded.lookup(np.array([azimuth]), np.array([elevation]))[0])

    # Inside the wall's span, well clear of its edges.
    assert t(285.0, 12.0) < 0.15
    # Above the wall, and in open sky beside it.
    assert t(285.0, 40.0) > 0.85
    assert t(60.0, 15.0) > 0.85
    # The fitted line sits at the wall's height across its span.
    assert abs(float(horizon.at(np.array([285.0]))[0]) - WALL_HEIGHT_DEG) < 4.0


def test_a_horizon_file_round_trips() -> None:
    """A written horizon reads back as the same line, for seeding a refit."""
    text = "0.0\t5.00\n90.0\t10.00\n180.0\t0.00\n270.0\t20.00\n360.0\t5.00\n"
    horizon = read_horizon(text)
    again = read_horizon(horizon.to_text(step=90.0))
    np.testing.assert_allclose(again.at(np.array([0.0, 90.0, 270.0])), [5.0, 10.0, 20.0])


def test_the_written_map_has_the_axes_the_integration_expects() -> None:
    """1-degree axes over the whole sky, one field per array."""
    payload = np.load(io.BytesIO(_npz_bytes({"roof": np.zeros((361, 91))})))
    assert payload["azimuth_deg"][0] == 0.0
    assert payload["azimuth_deg"][-1] == 360.0
    assert payload["elevation_deg"][-1] == 90.0
    assert payload["T_roof"].shape == (361, 91)
