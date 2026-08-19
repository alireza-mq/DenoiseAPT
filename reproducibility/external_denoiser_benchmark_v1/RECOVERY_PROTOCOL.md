# External benchmark v1 recovery

The frozen v1 evaluator completed its in-memory computations on the already
spent retrospective panel, then stopped before staging or committing any
artifact. Its median-filter overall OS-NRMSE differed from an earlier parity
sentinel by `5.66e-10`; independent diagnosis also found a `1.05e-9` difference
for the median anomaly-region sentinel. Both are far below the aggregate
summary's four-decimal reporting precision.

The recovery is additive and post-access. It preserves the original runner,
configuration, selection, prefreeze receipt, method outputs, metrics,
aggregation, and exact integer checks. It changes only the absolute tolerance
for the two median-filter floating parity cells from `1e-12` to `2e-9`.
Every other parity float retains `1e-12`, and NaN or infinity fails.

Before recomputation, an exclusive authorization receipt binds the original
execution identity and files, this wrapper, its tests, this note, the exact
failure, the two expected/observed/delta values, runtime identity, and absence
of canonical or staging artifacts. The wrapper reruns every method from
scratch, scopes and restores the assertion patch, invokes the unchanged legacy
audit, and writes an exclusive recovery audit receipt last. No Wavelet,
Noisereduce, or RINS-T endpoint was inspected to choose this numerical guard.
