# Correctness assertions and stateful benchmarks

`tests/test_utils.py` supplies the regular-operator shape/range grid. Its
`assert_result_equal` is for value-preserving operations; `assert_result_close`
uses the original output dtype's shared arithmetic tolerance. Both check dtype
and shape. FP8 comparison conversion lives in `flag_gems.testing`, after the
original dtype checks; individual tests do not need FP8 comparison adapters.

Create an independent oracle input with
`accuracy_utils.to_reference(input, independent=True)`. Dense snapshots preserve
strides, storage offsets, conjugate/negative flags and storage alias relationships
within one `flag_gems.testing.clone_inputs(...)` call. Copies are independent
leaves; this utility does not copy an autograd graph or a tensor version counter.
Scalar metadata oracles such as `_version` need their own counter-aware setup.

Resolve correctness candidates through `test_utils.resolve_gems_op` with one
public operator name, including for out overloads. The wrapper calls that exact
candidate and checks its nonmutable tensor inputs afterwards. It permits changes
to an in-place receiver and explicit output buffers (including aliased buffers).
It never implements an out overload or repairs a candidate result. If the native
reference changes dense input metadata, a test may supply
`expected_input_metadata={argument_index: reference_input_after_call}`;
this expectation must come from the independent reference. Stored values remain
protected by the pre-call snapshot.

`special_value_cases(dtypes)` enumerates nan-only, inf-only and mixed scenarios
when representable. Finite-only FP8 formats still receive nan-only coverage.
`make_special_input` constructs the corresponding payload. Default/core tests
select these scenarios; quick tests retain supported dtypes without these sweeps.

The generic dtype probe is for a unary rank-1 default overload. Other signatures
must provide an explicit valid probe. Input-construction failures and unexpected
native errors are inconclusive and propagate, rather than removing a dtype.
Custom boolean probes are responsible for distinguishing capability failures
from bad signatures or environmental failures.

In benchmarks, explicitly supplying `gems_op` (even `None`) requires a resolvable
direct candidate. Its absence raises instead of timing a native fallback. Legacy
benchmarks without this argument retain their dispatcher route.

`is_inplace=True` restores independent inputs before every warmup/sample and
before each candidate profiling capture. Restoration and synchronization precede
the measured call. Kernel mode uses device events, operator mode measures through
completion, and wrapper mode measures the host call. Each side uses the same
original state; the caller's template is unchanged. State-changing CUDA graph and
backward measurements are rejected because this implementation cannot express
their reset/measurement boundary correctly. No latency is fabricated for them.

Regression checks:

```bash
PYTHONPATH=src:. python -m pytest -q --import-mode=importlib \
  tests/test_validation_primitives.py tests/test_registered_op_override.py \
  tests/test_candidate_code.py benchmark/test_benchmark_case_api.py
```
