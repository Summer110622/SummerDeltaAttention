"""CPU scaffold for a ~10M-parameter causal LM with Summer Delta Attention blocks.

Not a language-model result. See README.md and DESIGN.md in this directory.
"""

try:
    from .model import SDALM, SDALMConfig, count_parameters
except ImportError:  # script-style path, parent package not loaded
    from model import SDALM, SDALMConfig, count_parameters

__all__ = ["SDALM", "SDALMConfig", "count_parameters"]
