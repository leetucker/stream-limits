"""``python -m stream_limits``: what was found, as JSON."""

import json

from . import describe

print(json.dumps(describe(), indent=2))
