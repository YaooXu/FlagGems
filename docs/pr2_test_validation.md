# PR #2 test repair and validation

This revision repairs false acceptance, candidate routing, and coverage gaps in
70 operator correctness files and their benchmark entry points. The reusable
comparison/input-state/measurement support lives in dependency
[PR #4](https://github.com/YaooXu/FlagGems/pull/4), commit `6df89d652` on
`fix/test-validation-primitives`. Merge that dependency first, then synchronize
this still-open `feat/new-api-for-kernelgen-server-test-gen` branch with the
updated target. `tests/test_utils.py` is owned by the dependency, so this PR no
longer adds a conflicting copy.

## Repairs

| Review finding | Result |
| --- | --- |
| R1–R2: contaminated oracle and loose copy/view comparisons | Independent reference inputs, candidate input snapshots, zero-tolerance value-preserving comparisons, shared dtype-aware FP8 compatibility. |
| R3–R4: overload aliases and fabricated out support | One public candidate name per operator; real out arguments go directly to the candidate. Default-plus-copy adapters are removed. |
| R5: optional candidate backward checks | Differentiable tests require candidate gradients and compare their values. Detached or wrong-gradient candidates fail. |
| R6–R7: native fallback and consumed benchmark inputs | The shared dependency requires direct candidates and restores state before each state-changing invocation outside timing. |
| R8: removed or magnitude-capped extremes | Five ranges restored for convolution; chain inputs retain their declared magnitude. Extreme reference calls preserve native intermediate overflow. |
| R9: missing dtype/parameter/broadcast cases | add includes int8/uint8 and alpha=0; broadcast baselines use (20,320,15). chain_matmul includes supported float64. |
| R10: quick boundaries | Dedicated broadcast/backward/special-value and add tensor/scalar expansions are selected in default mode. Quick keeps dtype coverage, including complex32; alpha/reduce_range use defaults. |
| R11: FP8 special values and stale comments | Separate representable nan-only/inf-only/mixed cases, including FP8 views/copies/duals; corrected bf16 explanations and removed duplicated unsigned generators. |
| R12: incomplete shape × range grid | Expanded detach_copy, dual and related view range grids to the selected required shapes. |
| R13: ambiguous dtype probe failures | Shared probing propagates input-generation and unexpected signature failures instead of classifying them as unsupported dtype. |
| R14: reference-only negative test | `_coalesce` FP8 rejection now invokes and checks the candidate. |

For `chain_matmul`, each intermediate matrix product has the tested dtype's
rounding and overflow. Upcasting the whole chain to fp64 changes that behavior;
the reference now invokes the same native operator on independent original-dtype
inputs. This was checked with native candidates over the full chain grid, without
increasing comparison tolerances.

The current CUDA `slow_conv_transpose2d` mutates the shape of an unbatched input
from (C,H,W) to (1,C,H,W). Tests obtain the expected post-call metadata from the
independent native reference and still verify that stored values did not change.
The out test also uses a separate input for its preliminary output-shape call,
so that call cannot change the workload subsequently presented to the oracle.

Known entry branches in add include tensor/tensor, tensor/scalar, scalar/tensor,
scalar/scalar and complex operand combinations; the corresponding tests identify
these forms explicitly. Pointwise view/layout cases exercise contiguous,
non-contiguous, empty and broadcast inputs. For newly exposed operators without
an available FlagGems implementation, shape/rank coverage documents observable
native contracts, not a claim that an unknown future kernel's dispatch thresholds
have already been covered.

## Validation scope

Validation used H20-3e physical GPU 3, Torch
`2.8.0a0+5228986c39.nv25.06`, and the default reference device (`TO_CPU=False`).
Only public operator names were overridden with native ATen packets. This checks
that the tests accept valid native behavior; it does not certify optimized
implementations or performance speedups.

- Final complete default suite: 50,908 passed, zero failures/skips, including
  all 162 newly added special-scenario cases.
- Quick suite: 9,207 passed, zero failures/skips. The previous four empty
  complex32 parameter cases were repaired rather than counted as coverage.
- Shared assertion, candidate API and benchmark API regressions: 63 passed.
- Wrong-candidate and actual measurement acceptance probes: 10 passed. They reject
  changed copy values, input mutation, missing/wrong gradients, forward-only out
  callables and permissive negative candidates. Actual kernel/operator measurements
  start every reference/candidate invocation with the declared nonempty sparse
  input; input preparation occurs outside the timed region.

The input-restoring benchmark implementation explicitly rejects state-changing
CUDA graph/backward timing; it does not claim those modes have been validated.
Cross-backend execution and full performance benchmarks were not part of this
repair's verification. Torch/Triton/vendor packages were not installed or changed;
pre-commit initialized its own isolated hook environments.
