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

"""Register Kunlunxin's intentional ATen overrides on old and new Torch."""


def impl_with_override(library, name, fn, dispatch_key=""):
    try:
        library.impl(name, fn, dispatch_key, allow_override=True)
    except TypeError as exc:
        if "unexpected keyword argument 'allow_override'" not in str(exc):
            raise
        library.impl(name, fn, dispatch_key)
