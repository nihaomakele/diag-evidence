# Contributing to DiagEvidence

DiagEvidence v0.1.1 is a research preview. Contributions should make evidence
checks, failure behavior and reproducibility easier to inspect.

## Local setup

Use Python 3.10 or newer and an isolated environment:

```sh
python -m venv .venv
# macOS / Linux
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s diag_evidence/tests -t . -v
diag-evidence demo --scenario all --no-write
```

The tests use injected responses and a temporary HTTP server on 127.0.0.1.
They require no external API, API key, GPU, model checkpoint or network service.
Optional HTTP examples use `python -m pip install -e '.[demo]'`; CSV generation
uses the `data` extra. The `plots` extra is reserved for optional local plots.

## Before a pull request

- Explain the concrete input or service behavior that caused the problem.
- Add a regression test for a correctness or failure-handling change.
- Preserve the distinction between tool execution, evidence support and model
  confidence. A rule-based support level is not a calibrated probability.
- Use synthetic fixtures. Do not submit real signals, credentials, logs, model
  weights or third-party data without documented redistribution permission.
- Update documentation when a public command or response contract changes.
- Run the tests and offline demo above. For packaging changes also run:

  ```sh
  python -m pip install build
  python -m build
  python scripts/check_release.py dist
  ```

Use focused pull requests. Describe the resulting behavior and the checks that
passed. Do not claim a performance improvement from the offline scenario suite;
it verifies contracts and failure handling, not real equipment accuracy.

## Scope and discussion

Open an issue for a reproducible bug or a proposed feature. Include the package
version (`diag-evidence --version`), Python version, operating system, minimal synthetic input and expected
behavior. Remove credentials and equipment-identifying information first.
Use the issue templates where they apply. Security-sensitive reports follow
[SECURITY.md](SECURITY.md).

New model integrations must state their source, version, license, input/output
contract and evaluation protocol. Downloading a checkpoint must be explicit;
the offline demo must remain reproducible without it.
