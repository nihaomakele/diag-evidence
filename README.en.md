# DiagEvidence

> **Development version: 0.2.0.dev3, not a release.** Adds [feature-drift analysis and an offline envelope/order comparison](docs/feature_drift_and_order_study.md). On the previously unread B operating profile, angular accuracy is 29.17% versus 35.42% for matched time-domain features. No candidate is promoted. Earlier [cross-source evaluation](docs/grouped_and_external_validation.md) also failed; this is unsuitable for operational diagnosis.

[中文](README.md)

A research toolkit for contract checks, conclusion constraints, and replayable evaluation of vibration diagnostic tool workflows. It checks numeric input, calls external classification and retrieval services, validates their contracts, applies a versioned policy, and produces a constrained report with isolated run records.

**The published v0.1.1 is a research preview; this directory is subsequent development.** `high` denotes completion of declared rule checks, not calibrated accuracy. An optional NumPy baseline is trained from licensed laboratory signals. No production validation or unmanned multi-agent coordination algorithm is included. Healthy is an explicit opt-in class, not an unknown/OOD state.

## Quick start

Python >=3.10:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
diag-evidence demo --scenario all --no-write
diag-evidence evaluate --output-dir artifacts/evaluation
```

On Windows activate `.venv\Scripts\Activate.ps1`. Installation requires dependencies; the installed synthetic demo runs offline, without keys or model weights. `passed` counts expected contract behavior, not diagnosis accuracy.

Features include strict finite signal validation before network access, classifier response validation, retrieval deduplication and explicit score provenance, four-state checks, enforced report strength limits, escaped external quotations, unique run artifacts, bounded audit summaries and visible persistence failures.

For real services, configure `DIAG_EVIDENCE_RAG_BASE_URL` and `DIAG_EVIDENCE_FD_BASE_URL`, then run `diag-evidence run input.json --output-dir artifacts/run`. See [contracts](docs/contracts.md), [policy](docs/policy.md), [architecture](docs/architecture.md), [limitations](docs/limitations.md), and [validation](VALIDATION.md). `--no-write` disables all report/trace/audit writes. `.env` is not loaded automatically.

```bash
python -m unittest discover -s diag_evidence/tests -t . -v
python -m pip install build
python -m build
python scripts/check_release.py dist
```

Integration tests use temporary local HTTP servers and need loopback permission; no public network is required. CI is configured for Python 3.10–3.13. The recorded local environment is documented separately; no unexecuted CI success is claimed.

The first public release uses `DiagEvidence` as the project name, `diag-evidence` for the distribution and CLI, `diag_evidence` for imports, and `DIAG_EVIDENCE_` for configuration variables. The earlier local prototype name is retained in provenance records. The project has not been uploaded to PyPI; the commands above install this source tree.

MIT licensed project code; third-party dependencies have their own licenses. Unverified legacy weights and training data are excluded. See [asset provenance](DATA_PROVENANCE.md), [third-party notices](THIRD_PARTY_NOTICES.md), and [contribution guide](CONTRIBUTING.md).

The next milestones are an authorized real workflow, a frozen-output comparison against simple baselines, and evidence applicability checks. See the [research and maintenance roadmap](docs/roadmap.md) for acceptance criteria; planned multi-agent work is separate from the current implementation.
