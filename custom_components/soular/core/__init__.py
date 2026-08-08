"""Home Assistant free forecasting core.

Everything in this package is pure computation over numpy arrays. It must never
import Home Assistant, which is enforced by the "Core is standalone" import-linter
contract. That boundary is what lets ``tools/backtest.py`` replay the exact code
path the integration runs, so measured skill numbers describe the shipped model.
"""
