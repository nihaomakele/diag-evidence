# DiagEvidence changelog

This file records changes to the public research-preview source tree.

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
