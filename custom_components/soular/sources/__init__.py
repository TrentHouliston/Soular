"""Weather data sources.

These fetch and parse; they do no physics. Each source polls independently and
fails independently, so losing one degrades the forecast rather than breaking it.
"""
