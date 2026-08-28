# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Process-local candidate loading used by FlagGems' pytest integration."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Callable, Iterator, Optional


_CANDIDATE_LOCK = RLock()
_ACTIVE_CANDIDATE: Optional["_CandidateSession"] = None


@dataclass
class _CandidateSession:
    source_path: Path
    source_sha256: str
    function: Callable
    module_name: str
    current_case: Callable[[Optional[str]], Optional[str]]
    track_calls: bool = True
    operator: Optional[str] = None
    calls: Counter[tuple[Optional[str], Optional[str]]] = field(
        default_factory=Counter
    )
    wrapped: Callable = field(init=False)

    def __post_init__(self) -> None:
        @functools.wraps(self.function)
        def tracked(*args, **kwargs):
            current_test = os.environ.get("PYTEST_CURRENT_TEST")
            nodeid = current_test.rsplit(" (", 1)[0] if current_test else None
            with _CANDIDATE_LOCK:
                operator = self.operator
            case_id = self.current_case(operator)
            with _CANDIDATE_LOCK:
                self.calls[(nodeid, case_id)] += 1
            return self.function(*args, **kwargs)

        self.wrapped = tracked

    def bind(self, operator: str) -> Callable:
        with _CANDIDATE_LOCK:
            if self.operator is None:
                self.operator = operator
            elif self.operator != operator:
                raise RuntimeError(
                    "--candidate-code-path may target only one public operator "
                    f"per pytest session: already bound to {self.operator!r}, "
                    f"then resolved {operator!r}. Select one operator explicitly."
                )
            if self.track_calls:
                self.wrapped.__name__ = operator
                return self.wrapped
            return self.function

    def set_call_tracking(self, enabled: bool) -> None:
        with _CANDIDATE_LOCK:
            self.track_calls = enabled

    def report(self, expected_nodeids: tuple[str, ...] = ()) -> dict:
        with _CANDIDATE_LOCK:
            records = [
                {"nodeid": nodeid, "case_id": case_id, "count": count}
                for (nodeid, case_id), count in sorted(
                    self.calls.items(),
                    key=lambda item: (item[0][0] or "", item[0][1] or ""),
                )
            ]
            observed_nodeids = {
                nodeid for nodeid, _ in self.calls if isinstance(nodeid, str)
            }
            total_calls = sum(self.calls.values())
            operator = self.operator
            call_tracking = self.track_calls
        missing_nodeids = sorted(set(expected_nodeids) - observed_nodeids)
        return {
            "schema_version": "flaggems.candidate-code-report/v1",
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "entrypoint": "run",
            "operator": operator,
            "call_tracking": call_tracking,
            "total_calls": total_calls,
            "missing_nodeids": missing_nodeids,
            "records": records,
        }


def _load_candidate(path: Path) -> tuple[Callable, str, str]:
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read candidate code: {path}: {exc}") from exc
    digest = hashlib.sha256(source).hexdigest()
    module_name = f"_flag_gems_candidate_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate code: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    function = getattr(module, "run", None)
    if not callable(function):
        sys.modules.pop(module_name, None)
        raise RuntimeError(
            f"candidate code must export a callable run(): {path}"
        )
    return function, digest, module_name


@contextmanager
def candidate_code(
    path: str | os.PathLike[str],
    *,
    current_case: Callable[[Optional[str]], Optional[str]],
    track_calls: bool = True,
) -> Iterator[Callable]:
    """Load ``path::run`` as the candidate for one pytest process.

    The public operator is bound lazily by the first ``resolve_gems_op`` call,
    which keeps the user-facing pytest interface limited to one path option.
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise RuntimeError(f"candidate code path is not a file: {source_path}")
    parent = str(source_path.parent)
    sys.path.insert(0, parent)
    try:
        function, digest, module_name = _load_candidate(source_path)
    except Exception:
        sys.path.remove(parent)
        raise
    session = _CandidateSession(
        source_path=source_path,
        source_sha256=digest,
        function=function,
        module_name=module_name,
        current_case=current_case,
        track_calls=track_calls,
    )
    global _ACTIVE_CANDIDATE
    with _CANDIDATE_LOCK:
        if _ACTIVE_CANDIDATE is not None:
            sys.modules.pop(module_name, None)
            sys.path.remove(parent)
            raise RuntimeError("a candidate code session is already active")
        _ACTIVE_CANDIDATE = session
    try:
        yield session.wrapped
    finally:
        with _CANDIDATE_LOCK:
            if _ACTIVE_CANDIDATE is session:
                _ACTIVE_CANDIDATE = None
        sys.modules.pop(module_name, None)
        try:
            sys.path.remove(parent)
        except ValueError:
            pass


def resolve_candidate(operator: str) -> Optional[Callable]:
    """Resolve the active path candidate, binding it to ``operator`` once."""

    with _CANDIDATE_LOCK:
        session = _ACTIVE_CANDIDATE
    return None if session is None else session.bind(operator)


def is_candidate(operator: str, function: Callable) -> bool:
    with _CANDIDATE_LOCK:
        session = _ACTIVE_CANDIDATE
        return (
            session is not None
            and session.operator == operator
            and (function is session.wrapped or function is session.function)
        )


def set_candidate_call_tracking(enabled: bool) -> None:
    """Enable or disable the per-invocation wrapper for the active session."""

    with _CANDIDATE_LOCK:
        session = _ACTIVE_CANDIDATE
    if session is None:
        raise RuntimeError("no candidate code session is active")
    session.set_call_tracking(enabled)


def candidate_report(expected_nodeids: tuple[str, ...] = ()) -> Optional[dict]:
    with _CANDIDATE_LOCK:
        session = _ACTIVE_CANDIDATE
    return None if session is None else session.report(expected_nodeids)


def write_candidate_report(
    path: str | os.PathLike[str], expected_nodeids: tuple[str, ...] = ()
) -> dict:
    report = candidate_report(expected_nodeids)
    if report is None:
        raise RuntimeError("no candidate code session is active")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "candidate_code",
    "candidate_report",
    "is_candidate",
    "resolve_candidate",
    "set_candidate_call_tracking",
    "write_candidate_report",
]
