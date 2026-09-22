"""Optional in-process capture hook for ``--profile-only``."""

from __future__ import annotations

import importlib
import os
from contextlib import nullcontext


def profile_capture_scope(*, backend: str, case_id: str):
    module_name = os.environ.get("KERNELGEN_PROFILE_HOOK_MODULE", "").strip()
    if not module_name:
        return nullcontext()
    factory = getattr(importlib.import_module(module_name), "profile_capture_scope", None)
    if not callable(factory):
        raise RuntimeError(f"profile hook {module_name!r} has no profile_capture_scope()")
    return factory(backend=backend, case_id=case_id)


__all__ = ["profile_capture_scope"]
