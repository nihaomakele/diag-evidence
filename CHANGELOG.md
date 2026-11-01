# DiagEvidence changelog

This file records changes to the public research-preview source tree.

## 0.2.0.dev4 — Record support screening

- Add source-only record-median support prototypes and a fixed threshold from
  48 leave-type-out record maxima; preserve the existing frozen classifier.
- Add pinned HIT-SM source metadata, bounded binary API download support and
  vector-layout validation for 42 recordings from two independent rigs.
- Compare score, distance and joint screening at window and record levels,
  with complete decision traces, attributed results and 20 additional tests.
- Retain the default configuration and prioritize device-specific validation
  and adaptation based on the measured retained counts and errors.

## 0.2.0.dev3 — Development, not released

- Add source-only drift intervals and exact linear healthy-margin decomposition;
  controlled resampling does not reproduce the previous complete collapse.
- Add an offline, noncausal 44-feature envelope/order study with encoder checks,
  matched physical windows, a time-domain control and frozen numeric models.
- Freeze source-only learner selection before downloading the pinned Ottawa B
  condition. Report its negative paired comparison; do not replace the baseline.
- Correct informational encoder fields in the A/B manifests, preserving all
  source file identities, hashes, algorithms and predictions.
- Add numerical modulation checks and public attributed aggregate results.

## 0.2.0.dev2 — Local development, not released

- Add nested leave-type-out validation with per-fold scaling, original-record
  isolation, frozen inner selection and separate development model selection.
- Add a pinned Ottawa v2 profile-A subset and frozen cross-source evaluation,
  with explicit antialias resampling and shape-only transfer.
- Add numeric JSON inference, bounded tree traversal and exact comparisons with
  library predictions. No target fitting or threshold adjustment.
- Report poor worst-type performance and complete external fault-to-healthy
  misses; do not promote the candidate or claim zero-coverage accuracy.

## 0.2.0.dev1 — Local development, not released

- Add six fixed feature/model comparisons, validation-only selection and a
  frozen 6204 partial holdout plus unsupported-combination stress test.
- Report the candidate's mixed generalization and ineffective distance-based
  rejection. Keep the original default baseline unchanged.
- Add explicit abstention to the classifier contract and independent report
  checks (policy 1.3). Preserve model availability without emitting a category.
- Add a guarded live experimental adapter that refuses categories when its
  declared validation quality target is unmet; zero coverage is not accuracy.

## 0.2.0.dev0 — Local development, not released

- Add opt-in historical reference applicability checks for declared units,
  sampling rate, optional preprocessing revision and timezone-aware cutoff.
- Exclude failed/unknown references from label and count support while retaining
  them in reports with deterministic reason codes. Preserve cross-asset history.
- Recompute checks in the report using the same explicit context. Update policy
  to diagnosis-evidence/1.2 and reject contradictory applicability metadata on
  reused case identities. Legacy calls mark this check not applicable.
- These checks do not authenticate metadata or establish real diagnostic gains.
- Add opt-in bearing-health-v1 (healthy class 3), independently rechecked in the
  report; keep default faults-v1 unchanged.
- Add a pinned, licensed 48-record HUST benchmark, type-group split before
  windowing, training-only centroid model and retrieval, frozen outputs and
  matched-coverage comparisons. Real signals and simulated-clock faults are
  explicitly separate. Report poor generalization and healthy false alarms.

## 0.1.1 — Review fixes (2026-10-21)

- Namespace case identities by source, exclude retrieval score metadata from
  content identities, and reject conflicting payloads for one case identity.
- Reject unknown Python convenience-API options; strengthen malformed check
  handling and numeric boundaries in direct verification and report APIs.
- Check effective report strength and consistency in the synthetic evaluator.
- Remove URL query credentials from retained source links, redact sensitive
  log keys, and avoid reflecting invalid CLI arguments.
- Reject HTTP redirects and bound HTTP/RAG response consumption.
- Add targeted regressions and a staged research roadmap. These checks do not
  establish real diagnostic accuracy or multi-agent coordination benefits.

## 0.1.0 — Research preview (unreleased)

- Renamed the original `harness-agent` work to DiagEvidence before its first
  public release: distribution/CLI `diag-evidence`, import `diag_evidence`,
  environment prefix `DIAG_EVIDENCE_`. Original provenance and MIT attribution
  are retained.
- Package metadata, an installable command-line entry point and optional
  dependency groups.
- An offline synthetic demonstration and reproducible scenario evaluation.
- Validation of signals and service responses, constrained conclusions and
  explicit failure behavior.
- Per-run artifacts with recursive redaction and trace isolation.
- Real loopback HTTP integration tests and a configured Python 3.10–3.13 CI matrix.
- Community, provenance and release documentation. Model weights and private
  data are excluded from the release source and distributions.

The synthetic scenarios exercise expected behavior. They do not establish
accuracy on real equipment, multi-agent coordination performance or a
calibrated probability of diagnostic correctness.
