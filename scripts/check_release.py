"""Inspect release archives without extracting or executing their contents.

This is a bounded check for disallowed artifacts and common credential formats,
not a proof of data ownership or the absence of all possible secrets.
"""

from __future__ import annotations

import argparse
import configparser
from email.parser import Parser
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".bin", ".onnx", ".pkl", ".pickle",
    ".pdf", ".csv", ".mat", ".npy", ".npz", ".jsonl", ".log", ".pyc", ".pyo", ".key", ".pem",
}
FORBIDDEN_PARTS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "private_data",
    "logs", "reports", "secrets", ".ci",
}
SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
# Explicitly synthetic regression fixtures are not live credentials.
SYNTHETIC_PREFIXES = ("sk-TEST_ONLY_", "sk-test-", "sk-fake-")
SYNTHETIC_CSV_SHA256 = "71ce1eef099ca3e43ba5241c46d0e1e1cf15e52db942027a8107c508122268a9"


def _members(path: Path):
    if path.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if not entry.is_dir():
                    yield entry.filename, archive.read(entry)
    else:
        with tarfile.open(path, "r:gz") as archive:
            for entry in archive.getmembers():
                if entry.issym() or entry.islnk():
                    raise ValueError(f"archive contains a link: {entry.name}")
                if entry.isfile():
                    source = archive.extractfile(entry)
                    if source is not None:
                        yield entry.name, source.read()


def inspect_archive(path: Path) -> dict:
    names = []
    wheel_metadata = {}
    violations = []
    for name, content in _members(path):
        names.append(name)
        posix = PurePosixPath(name)
        if any(part == "harness_agent" or part.startswith(("harness_agent-", "harness_agent.", "harness-agent-"))
               for part in posix.parts):
            violations.append(f"obsolete package artifact: {name}")
        if posix.is_absolute() or ".." in posix.parts:
            violations.append(f"unsafe archive path: {name}")
        if any(part in FORBIDDEN_PARTS for part in posix.parts):
            violations.append(f"runtime/private directory: {name}")
        verified_synthetic_csv = (
            path.suffix != ".whl"
            and posix.parts[-2:] == ("examples", "synthetic_signal_1025.csv")
            and hashlib.sha256(content).hexdigest() == SYNTHETIC_CSV_SHA256
        )
        if posix.suffix.lower() in FORBIDDEN_SUFFIXES and not verified_synthetic_csv:
            violations.append(f"disallowed artifact: {name}")
        if posix.name == ".env" or posix.name.endswith(".env"):
            violations.append(f"environment file: {name}")
        text = content.decode("utf-8", errors="ignore")
        if path.suffix == ".whl" and ".dist-info/" in name:
            wheel_metadata[name] = text
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                synthetic_test_fixture = ("tests" in posix.parts
                                          and match.group().startswith(SYNTHETIC_PREFIXES))
                if not synthetic_test_fixture:
                    violations.append(f"possible credential in: {name}")
                    break
    if path.suffix == ".whl":
        required = {"diag_evidence/cli.py", "diag_evidence/skills/fault_diagnosis.md"}
        for name in sorted(required - set(names)):
            violations.append(f"missing installed package resource: {name}")
        if any("diag_evidence/tests/" in name for name in names):
            violations.append("wheel contains development tests")
        entry_points = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if not entry_points:
            violations.append("missing CLI entry point metadata")
        else:
            try:
                entry_config = configparser.ConfigParser()
                entry_config.read_string(wheel_metadata[entry_points[0]])
                scripts = dict(entry_config["console_scripts"])
                if scripts != {"diag-evidence": "diag_evidence.cli:main"}:
                    violations.append("CLI entry point does not match diag-evidence = diag_evidence.cli:main")
            except (configparser.Error, KeyError):
                violations.append("invalid CLI entry point metadata")
        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            violations.append("expected one wheel METADATA file")
        elif Parser().parsestr(wheel_metadata[metadata_files[0]]).get("Name") != "diag-evidence":
            violations.append("wheel distribution name is not diag-evidence")
    elif path.name.endswith(".tar.gz"):
        required = {"pyproject.toml", "LICENSE", "README.md", "README.en.md",
                    "CITATION.cff", "scripts/check_release.py",
                    "examples/make_input.py", "examples/fd_stub.py", "examples/rag_stub.py",
                    "examples/generate_synthetic_signal.py", "examples/synthetic_signal_1025.csv"}
        relative = {"/".join(PurePosixPath(name).parts[1:]) for name in names}
        for name in sorted(required - relative):
            violations.append(f"missing source file: {name}")
        if not any(name.startswith("diag_evidence/tests/") for name in relative):
            violations.append("source distribution omits regression tests")
    return {"archive": path.name, "files": len(names), "passed": not violations,
            "violations": sorted(set(violations))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a wheel, source archive, source ZIP, or dist directory")
    args = parser.parse_args()
    paths = ([args.path] if args.path.is_file() else
             sorted(list(args.path.glob("*.whl")) + list(args.path.glob("*.tar.gz"))))
    if not paths:
        parser.error("no release archives found")
    results = []
    for path in paths:
        try:
            results.append(inspect_archive(path))
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
            results.append({"archive": path.name, "passed": False,
                            "violations": [f"cannot inspect archive: {type(exc).__name__}"]})
    print(json.dumps({"passed": all(item["passed"] for item in results),
                      "checks": "artifact names and common credential formats; not a full secret or license audit",
                      "results": results}, indent=2, ensure_ascii=False))
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
