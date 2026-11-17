"""When measured output is not telling the truth about the sky.

Curtailment is the dominant problem on a battery site. When the battery is full
and export is limited, the inverter throttles the array, and it does so at the
sunniest part of the day. Counted, that reads as the weather forecast promising
sun that never came, and the correction would mark every sunny day down.
"""

from typing import Final

# A battery this full is either curtailing or about to. Instants above it are
# left out of both sides of every comparison with measured output.
CURTAILMENT_SOC_PCT: Final = 97.0
