"""Hide the container's flash-attn builds from TE under the PyTorch-nightly overlay.

They are compiled against the container's torch 2.13 and fail to load (undefined
c10 symbols) with torch nightly; TE imports them unconditionally whenever their
package metadata exists, so report them as not installed. TE then uses cuDNN
fused attention.
"""

import importlib.metadata as _md

_HIDDEN = {"flash-attn", "flash_attn", "flash-attn-3", "flash_attn_3", "flash-attn-4", "flash_attn_4"}
_orig_version = _md.version


def _version(name):
    if name in _HIDDEN:
        raise _md.PackageNotFoundError(name)
    return _orig_version(name)


_md.version = _version

# flash-linear-attention's Triton autotune configs are rejected by the nightly's
# Triton 3.9 at import time; MCore only imports it under `except ImportError`.
import sys as _sys

_sys.modules["fla"] = None
