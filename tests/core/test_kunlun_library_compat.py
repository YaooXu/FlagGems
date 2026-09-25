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

"""Kunlunxin ATen overrides retain their behavior on older Torch versions."""

import runpy
from pathlib import Path

import pytest


_MODULE = (
    Path(__file__).resolve().parents[2]
    / "src/flag_gems/runtime/backend/_kunlunxin/ops/_torch_library_compat.py"
)
impl_with_override = runpy.run_path(str(_MODULE))["impl_with_override"]


def test_modern_library_requests_override():
    calls = []

    class Library:
        def impl(self, name, fn, dispatch_key="", *, allow_override=False):
            calls.append((name, fn, dispatch_key, allow_override))

    fn = lambda: None
    impl_with_override(Library(), "range.step", fn, "CUDA")
    assert calls == [("range.step", fn, "CUDA", True)]


def test_legacy_library_registers_without_unsupported_keyword():
    calls = []

    class Library:
        def impl(self, name, fn, dispatch_key=""):
            calls.append((name, fn, dispatch_key))

    fn = lambda: None
    impl_with_override(Library(), "range", fn)
    assert calls == [("range", fn, "")]


def test_other_type_error_is_not_hidden_or_retried():
    calls = []

    class Library:
        def impl(self, name, fn, dispatch_key="", *, allow_override=False):
            calls.append((name, dispatch_key, allow_override))
            raise TypeError("invalid registration")

    with pytest.raises(TypeError, match="invalid registration"):
        impl_with_override(Library(), "range", lambda: None)
    assert calls == [("range", "", True)]
