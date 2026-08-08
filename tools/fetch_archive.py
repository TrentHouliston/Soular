"""Build a local, replayable archive of what the weather models said, and when.

Measuring forecast skill needs forecasts, not reanalysis. The distinction is easy
to lose: an archive of "what the weather was" scores a model that never existed,
and flatters it enormously. So everything here is stored against both a valid
time and a lead time, and the backtest can only ever ask for data at a lead it
would genuinely have had.

Three feeds, with materially different histories:

* **NWP** via the previous-runs API. ``shortwave_radiation_previous_day2`` at
  valid time T is what the run two days before T predicted for T. Lead time is
  resolved in whole days, which is the granularity the endpoint offers.
* **Satellite** via the satellite-radiation archive: JMA/JAXA Himawari at its
  native 10-minute resolution. Observations, not forecasts -- the nowcast is
  built from them in :mod:`custom_components.soular.core.blend`.
* **Ensemble** via the ensemble API. This one has a trap. The endpoint *accepts*
  historical date ranges and returns a correctly shaped response for them, but
  the values are all null: it serves the current run, not an archive of past
  runs. Only the few days around now carry data. So ensemble spread cannot be
  backtested historically at all, and this module says so loudly rather than
  storing an empty table that looks like a fetch that merely went quiet.

Raw payloads are retained gzipped so a parser fix can be replayed offline
without re-fetching, and every request is keyed so re-running is idempotent.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import datetime as dt
import gzip
import hashlib
import json
import pathlib
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

NWP_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SATELLITE_URL = "https://satellite-api.open-meteo.com/v1/archive"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Everything the forward model consumes. Requested together so one response
# carries a self-consistent set rather than a mixture of runs.
NWP_VARIABLES = (
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "temperature_2m",
    "wind_speed_10m",
    "cloud_cover",
)

SATELLITE_VARIABLES = (
    "shortwave_radiation",
    # Open-Meteo's own clear-sky, useful as an independent check on ours.
    "shortwave_radiation_clear_sky",
)

# What the incumbent integration actually asks for. It does not transpose: it
# requests plane-of-array irradiance from Open-Meteo and consumes the answer, so
# a faithful emulation has to consume the same numbers rather than a
# reimplementation of them.
INCUMBENT_VARIABLES = (
    "global_tilted_irradiance",
    "diffuse_radiation",
    "direct_radiation",
    "temperature_2m",
)

ENSEMBLE_VARIABLES = ("shortwave_radiation",)
ENSEMBLE_MODEL = "ecmwf_ifs025"
SATELLITE_MODEL = "jma_jaxa_himawari"

# Lead 0 is the freshest run covering a valid time; lead N is the run N days
# earlier. Three days covers the horizon the integration actually forecasts.
DEFAULT_MAX_LEAD_DAYS = 3

# Chunk sizes chosen so a single response stays a few megabytes. Requests are
# billed by data volume rather than count, so fewer, larger calls are cheaper.
NWP_CHUNK_DAYS = 45
SATELLITE_CHUNK_DAYS = 30
ENSEMBLE_CHUNK_DAYS = 15

REQUEST_PAUSE_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 180

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_responses (
    request_key TEXT PRIMARY KEY,
    feed        TEXT NOT NULL,
    url         TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    payload     BLOB NOT NULL
) WITHOUT ROWID;

-- lead_day 0 is the freshest run; N is the run N days earlier. Storing the lead
-- rather than an issue time is what makes a leak structurally impossible: the
-- backtest asks for a lead, and a lead it did not have does not exist here.
CREATE TABLE IF NOT EXISTS nwp (
    valid_utc TEXT NOT NULL,
    lead_day  INTEGER NOT NULL,
    variable  TEXT NOT NULL,
    value     REAL,
    PRIMARY KEY (valid_utc, lead_day, variable)
) WITHOUT ROWID;

-- Plane-of-array irradiance as the incumbent integration receives it, per array.
CREATE TABLE IF NOT EXISTS gti (
    array_name TEXT NOT NULL,
    valid_utc  TEXT NOT NULL,
    lead_day   INTEGER NOT NULL,
    variable   TEXT NOT NULL,
    value      REAL,
    PRIMARY KEY (array_name, valid_utc, lead_day, variable)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS satellite (
    valid_utc TEXT NOT NULL,
    variable  TEXT NOT NULL,
    value     REAL,
    PRIMARY KEY (valid_utc, variable)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS ensemble (
    valid_utc TEXT NOT NULL,
    member    INTEGER NOT NULL,
    variable  TEXT NOT NULL,
    value     REAL,
    PRIMARY KEY (valid_utc, member, variable)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
"""


class FetchError(RuntimeError):
    """Open-Meteo rejected a request, or returned something unparseable."""


def connect(path: Path) -> sqlite3.Connection:
    """Open the cache, creating its schema if needed."""
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def request_key(feed: str, params: dict[str, str]) -> str:
    """Stable identity for a request, so re-running skips completed work."""
    payload = json.dumps({"feed": feed, **params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def fetch_json(url: str, params: dict[str, str]) -> dict[str, Any]:
    """GET a JSON payload, turning API-level errors into exceptions."""
    query = urllib.parse.urlencode(params)
    full = f"{url}?{query}"
    try:
        with urllib.request.urlopen(full, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            body = response.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:400]
        msg = f"HTTP {err.code} from {url}: {detail}"
        raise FetchError(msg) from err
    except urllib.error.URLError as err:
        msg = f"could not reach {url}: {err.reason}"
        raise FetchError(msg) from err

    payload: dict[str, Any] = json.loads(body)
    if payload.get("error"):
        msg = f"{url} rejected the request: {payload.get('reason', 'no reason given')}"
        raise FetchError(msg)
    return payload


def store_raw(connection: sqlite3.Connection, key: str, feed: str, url: str, payload: dict[str, Any]) -> None:
    """Retain the response verbatim so a parser fix can replay without re-fetching."""
    connection.execute(
        "INSERT OR REPLACE INTO raw_responses (request_key, feed, url, fetched_at, payload) VALUES (?, ?, ?, ?, ?)",
        (
            key,
            feed,
            url,
            dt.datetime.now(dt.UTC).isoformat(),
            gzip.compress(json.dumps(payload).encode()),
        ),
    )


def already_fetched(connection: sqlite3.Connection, key: str) -> bool:
    """Report whether this exact request has been stored before."""
    row = connection.execute("SELECT 1 FROM raw_responses WHERE request_key = ?", (key,)).fetchone()
    return row is not None


def date_chunks(start: dt.date, end: dt.date, days: int) -> Iterator[tuple[dt.date, dt.date]]:
    """Split an inclusive date range into chunks of at most ``days``."""
    cursor = start
    while cursor <= end:
        stop = min(cursor + dt.timedelta(days=days - 1), end)
        yield cursor, stop
        cursor = stop + dt.timedelta(days=1)


def _series(payload: dict[str, Any]) -> dict[str, list[Any]]:
    """Return whichever time series section a response carries."""
    for section in ("minutely_15", "hourly"):
        block = payload.get(section)
        if isinstance(block, dict) and "time" in block:
            return block
    msg = f"response carried no time series; keys were {sorted(payload)}"
    raise FetchError(msg)


def fetch_nwp(
    connection: sqlite3.Connection,
    latitude: float,
    longitude: float,
    start: dt.date,
    end: dt.date,
    max_lead_days: int,
) -> int:
    """Pull fixed-lead-time NWP forecasts for the whole range."""
    requested: list[str] = []
    for variable in NWP_VARIABLES:
        requested.append(variable)
        requested.extend(f"{variable}_previous_day{lead}" for lead in range(1, max_lead_days + 1))

    rows = 0
    for chunk_start, chunk_end in date_chunks(start, end, NWP_CHUNK_DAYS):
        params = {
            "latitude": f"{latitude:.6f}",
            "longitude": f"{longitude:.6f}",
            "hourly": ",".join(requested),
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "timezone": "UTC",
            # Default is km/h, which would silently over-cool the array by a
            # factor of 3.6 in the Faiman model.
            "wind_speed_unit": "ms",
        }
        key = request_key("nwp", params)
        if already_fetched(connection, key):
            print(f"  nwp {chunk_start}..{chunk_end}: cached")
            continue

        print(f"  nwp {chunk_start}..{chunk_end}: fetching {len(requested)} variables")
        payload = fetch_json(NWP_URL, params)
        store_raw(connection, key, "nwp", NWP_URL, payload)
        rows += _ingest_nwp(connection, payload, max_lead_days)
        connection.commit()
        time.sleep(REQUEST_PAUSE_SECONDS)
    return rows


def _ingest_nwp(connection: sqlite3.Connection, payload: dict[str, Any], max_lead_days: int) -> int:
    """Flatten a previous-runs response into (valid time, lead, variable) rows."""
    block = _series(payload)
    times: list[str] = block["time"]
    records: list[tuple[str, int, str, float | None]] = []

    for variable in NWP_VARIABLES:
        for lead in range(max_lead_days + 1):
            column = variable if lead == 0 else f"{variable}_previous_day{lead}"
            values = block.get(column)
            if values is None:
                continue
            records.extend(
                (stamp, lead, variable, value) for stamp, value in zip(times, values, strict=True) if value is not None
            )

    connection.executemany("INSERT OR REPLACE INTO nwp VALUES (?, ?, ?, ?)", records)
    return len(records)


def fetch_gti(
    connection: sqlite3.Connection,
    latitude: float,
    longitude: float,
    start: dt.date,
    end: dt.date,
    max_lead_days: int,
    arrays: list[tuple[str, float, float]],
) -> int:
    """Pull tilted irradiance per array, as the incumbent integration receives it.

    Open-Meteo measures panel azimuth from south, positive westward; every Home
    Assistant integration uses compass degrees from north. The incumbent converts
    with ``azimuth - 180``, and so does this.
    """
    requested: list[str] = []
    for variable in INCUMBENT_VARIABLES:
        requested.append(variable)
        requested.extend(f"{variable}_previous_day{lead}" for lead in range(1, max_lead_days + 1))

    rows = 0
    for name, tilt, compass_azimuth in arrays:
        for chunk_start, chunk_end in date_chunks(start, end, NWP_CHUNK_DAYS):
            params = {
                "latitude": f"{latitude:.6f}",
                "longitude": f"{longitude:.6f}",
                "tilt": f"{tilt:.1f}",
                "azimuth": f"{compass_azimuth - 180.0:.1f}",
                "hourly": ",".join(requested),
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "timezone": "UTC",
            }
            key = request_key(f"gti:{name}", params)
            if already_fetched(connection, key):
                print(f"  gti {name} {chunk_start}..{chunk_end}: cached")
                continue

            print(f"  gti {name} {chunk_start}..{chunk_end}: fetching")
            payload = fetch_json(NWP_URL, params)
            store_raw(connection, key, "gti", NWP_URL, payload)

            block = _series(payload)
            times: list[str] = block["time"]
            records: list[tuple[str, str, int, str, float | None]] = []
            for variable in INCUMBENT_VARIABLES:
                for lead in range(max_lead_days + 1):
                    column = variable if lead == 0 else f"{variable}_previous_day{lead}"
                    values = block.get(column)
                    if values is None:
                        continue
                    records.extend(
                        (name, stamp, lead, variable, value)
                        for stamp, value in zip(times, values, strict=True)
                        if value is not None
                    )
            connection.executemany("INSERT OR REPLACE INTO gti VALUES (?, ?, ?, ?, ?)", records)
            connection.commit()
            rows += len(records)
            time.sleep(REQUEST_PAUSE_SECONDS)
    return rows


def fetch_satellite(
    connection: sqlite3.Connection,
    latitude: float,
    longitude: float,
    start: dt.date,
    end: dt.date,
) -> int:
    """Pull observed irradiance at the satellite's native 10-minute cadence."""
    rows = 0
    for chunk_start, chunk_end in date_chunks(start, end, SATELLITE_CHUNK_DAYS):
        params = {
            "latitude": f"{latitude:.6f}",
            "longitude": f"{longitude:.6f}",
            "hourly": ",".join(SATELLITE_VARIABLES),
            "models": SATELLITE_MODEL,
            # Without this the response is resampled to hourly, throwing away
            # exactly the sub-hourly variability the nowcast exists to capture.
            "temporal_resolution": "native",
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "timezone": "UTC",
        }
        key = request_key("satellite", params)
        if already_fetched(connection, key):
            print(f"  satellite {chunk_start}..{chunk_end}: cached")
            continue

        print(f"  satellite {chunk_start}..{chunk_end}: fetching")
        payload = fetch_json(SATELLITE_URL, params)
        store_raw(connection, key, "satellite", SATELLITE_URL, payload)

        block = _series(payload)
        times: list[str] = block["time"]
        records = [
            (stamp, variable, value)
            for variable in SATELLITE_VARIABLES
            if block.get(variable) is not None
            for stamp, value in zip(times, block[variable], strict=True)
            if value is not None
        ]
        connection.executemany("INSERT OR REPLACE INTO satellite VALUES (?, ?, ?)", records)
        connection.commit()
        rows += len(records)
        time.sleep(REQUEST_PAUSE_SECONDS)
    return rows


ALLOWED_RANGE = re.compile(r"allowed range from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})")


def ensemble_window(latitude: float, longitude: float) -> tuple[dt.date, dt.date] | None:
    """Discover the ensemble archive's window by asking for something outside it.

    The endpoint names its own limits in the rejection message, which beats
    hardcoding a date that will drift.
    """
    params = {
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "hourly": "shortwave_radiation",
        "models": ENSEMBLE_MODEL,
        "start_date": "2000-01-01",
        "end_date": "2000-01-02",
        "timezone": "UTC",
    }
    try:
        fetch_json(ENSEMBLE_URL, params)
    except FetchError as err:
        match = ALLOWED_RANGE.search(str(err))
        if match:
            return dt.date.fromisoformat(match.group(1)), dt.date.fromisoformat(match.group(2))
    return None


def fetch_ensemble(
    connection: sqlite3.Connection,
    latitude: float,
    longitude: float,
    start: dt.date,
    end: dt.date,
) -> int:
    """Pull ensemble members, clamped to whatever archive actually exists."""
    window = ensemble_window(latitude, longitude)
    if window is None:
        print("  ensemble: could not determine the archive window; skipping")
        return 0

    available_start, available_end = window
    clamped_start = max(start, available_start)
    clamped_end = min(end, available_end, dt.date.today())  # noqa: DTZ011
    if clamped_start > clamped_end:
        print(f"  ensemble: archive is {available_start}..{available_end}, which does not overlap the request")
        return 0

    if clamped_start > start:
        missing = (clamped_start - start).days
        # Say this out loud. A silently shortened window would make the quantile
        # calibration look like it was measured over the full period.
        print(f"  ensemble: archive starts {available_start}; {missing} days of the requested range are unavailable")

    connection.execute(
        "INSERT OR REPLACE INTO meta VALUES (?, ?)",
        ("ensemble_window", json.dumps({"start": clamped_start.isoformat(), "end": clamped_end.isoformat()})),
    )

    rows = 0
    empty_chunks = 0
    for chunk_start, chunk_end in date_chunks(clamped_start, clamped_end, ENSEMBLE_CHUNK_DAYS):
        params = {
            "latitude": f"{latitude:.6f}",
            "longitude": f"{longitude:.6f}",
            "hourly": ",".join(ENSEMBLE_VARIABLES),
            "models": ENSEMBLE_MODEL,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "timezone": "UTC",
        }
        key = request_key("ensemble", params)
        if already_fetched(connection, key):
            print(f"  ensemble {chunk_start}..{chunk_end}: cached")
            continue

        payload = fetch_json(ENSEMBLE_URL, params)
        store_raw(connection, key, "ensemble", ENSEMBLE_URL, payload)
        added = _ingest_ensemble(connection, payload)
        connection.commit()
        rows += added

        if added:
            print(f"  ensemble {chunk_start}..{chunk_end}: {added:,} values")
        else:
            # A well-formed response full of nulls. Keep going in case a later
            # chunk overlaps the live window, but do not pretend this worked.
            empty_chunks += 1
            print(f"  ensemble {chunk_start}..{chunk_end}: response carried no values (outside the served window)")
        time.sleep(REQUEST_PAUSE_SECONDS)

    stored = connection.execute("SELECT MIN(valid_utc), MAX(valid_utc) FROM ensemble").fetchone()
    if stored and stored[0]:
        connection.execute(
            "INSERT OR REPLACE INTO meta VALUES (?, ?)",
            ("ensemble_data_window", json.dumps({"start": stored[0], "end": stored[1]})),
        )
    if empty_chunks:
        print(
            f"  ensemble: {empty_chunks} of the requested chunks returned nulls. The ensemble endpoint\n"
            f"    serves the current run rather than an archive, so historical spread is not\n"
            f"    retrievable. Quantile calibration has to be measured online, or accumulated\n"
            f"    by running this fetcher on a schedule from here on."
        )
    return rows


MEMBER_SUFFIX = re.compile(r"_member(\d+)$")


def _ingest_ensemble(connection: sqlite3.Connection, payload: dict[str, Any]) -> int:
    """Flatten ensemble member columns into (valid time, member, variable) rows.

    Open-Meteo names the control run after the plain variable and the perturbed
    members ``<variable>_member01`` upward, so the control becomes member 0.
    """
    block = _series(payload)
    times: list[str] = block["time"]
    records: list[tuple[str, int, str, float | None]] = []

    for column, values in block.items():
        if column == "time":
            continue
        match = MEMBER_SUFFIX.search(column)
        member = int(match.group(1)) if match else 0
        variable = MEMBER_SUFFIX.sub("", column)
        if variable not in ENSEMBLE_VARIABLES:
            continue
        records.extend(
            (stamp, member, variable, value) for stamp, value in zip(times, values, strict=True) if value is not None
        )

    connection.executemany("INSERT OR REPLACE INTO ensemble VALUES (?, ?, ?, ?)", records)
    return len(records)


def summarise(connection: sqlite3.Connection) -> None:
    """Print what the cache now holds, per feed."""
    print("\nCache contents:")
    for table, extra in (
        ("nwp", ", COUNT(DISTINCT lead_day)"),
        ("satellite", ""),
        ("ensemble", ", COUNT(DISTINCT member)"),
    ):
        row = connection.execute(
            f"SELECT COUNT(*), MIN(valid_utc), MAX(valid_utc){extra} FROM {table}"  # noqa: S608
        ).fetchone()
        count, first, last = row[0], row[1], row[2]
        if not count:
            print(f"  {table:10s} empty")
            continue
        detail = f"  ({row[3]} {'leads' if table == 'nwp' else 'members'})" if extra else ""
        print(f"  {table:10s} {count:>9,} rows  {first} .. {last}{detail}")

    size = connection.execute("SELECT SUM(LENGTH(payload)) FROM raw_responses").fetchone()[0] or 0
    calls = connection.execute("SELECT COUNT(*) FROM raw_responses").fetchone()[0]
    print(f"  {'raw':10s} {calls} responses retained, {size / 1e6:.1f} MB compressed")


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch the archive feeds into a local cache."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--cache", type=Path, default=Path("backtest_cache.db"))
    parser.add_argument(
        "--feeds",
        default="nwp,satellite,ensemble",
        help="comma-separated subset of nwp, satellite, ensemble, gti",
    )
    parser.add_argument(
        "--arrays",
        help="path to arrays.toml, required for the gti feed",
    )
    parser.add_argument("--max-lead-days", type=int, default=DEFAULT_MAX_LEAD_DAYS)
    args = parser.parse_args(argv)

    if args.start > args.end:
        parser.error("--start must not be after --end")

    feeds = {feed.strip() for feed in args.feeds.split(",") if feed.strip()}
    unknown = feeds - {"nwp", "satellite", "ensemble", "gti"}
    if unknown:
        parser.error(f"unknown feeds: {sorted(unknown)}")

    connection = connect(args.cache)
    print(f"Cache: {args.cache}   range: {args.start} .. {args.end}   site: {args.latitude}, {args.longitude}")

    try:
        if "nwp" in feeds:
            print("\nNWP (previous-runs, fixed lead times):")
            fetch_nwp(connection, args.latitude, args.longitude, args.start, args.end, args.max_lead_days)
        if "satellite" in feeds:
            print("\nSatellite (Himawari, native 10-minute):")
            fetch_satellite(connection, args.latitude, args.longitude, args.start, args.end)
        if "gti" in feeds:
            if not args.arrays:
                parser.error("--arrays is required for the gti feed")
            import tomllib  # noqa: PLC0415

            config = tomllib.loads(pathlib.Path(args.arrays).read_text())
            arrays = [
                (name, float(entry["tilt"]), float(entry["azimuth"]))
                for name, entry in sorted(config["arrays"].items())
            ]
            print("\nTilted irradiance (as the incumbent integration receives it):")
            fetch_gti(connection, args.latitude, args.longitude, args.start, args.end, args.max_lead_days, arrays)
        if "ensemble" in feeds:
            print("\nEnsemble (ECMWF IFS, 51 members):")
            fetch_ensemble(connection, args.latitude, args.longitude, args.start, args.end)
    except FetchError as err:
        print(f"\nfailed: {err}", file=sys.stderr)
        return 1
    finally:
        connection.commit()

    summarise(connection)
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
