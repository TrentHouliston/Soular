"""Build a graded shading map for each array from its own measured output.

Each array is treated as a very slow, very low-resolution camera. On a clear day
the only thing between the modelled clear-sky output and the measured output is
whatever blocked the sun, so over a year of sun positions the ratio paints a map
of the sky each array can see. The result is the ``.npz`` the integration reads.

Inputs are deliberately generic, so any inverter's data will do:

* a site file (see :mod:`tools.site`) with each array's geometry, and
* a CSV of measured DC power per array at five-minute resolution::

      time,east,west,north,south,soc
      2026-01-01T00:00:00Z,0,0,0,0,54
      ...

  ``time`` is ISO 8601 (a timezone, or UTC if none). One column per array, named
  as in the site file, in watts. ``soc`` (battery %) is optional; when present a
  full battery is treated as curtailment, which it almost always is.

The map is measured through Soular's own physics -- the same solar position,
clear-sky and transposition code the integration forecasts with -- so it is
consistent with the model that consumes it by construction.

Usage::

    uv run shading-map --site site.toml --power power.csv --out outputs/
"""
