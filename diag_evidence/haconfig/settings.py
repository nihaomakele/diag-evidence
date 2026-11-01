"""Instance configuration; read environment variables with Settings.from_env()."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlparse

ENV_KEY_NAME = "DASHSCOPE_API_KEY"
RAG_HEALTH_PATH, RAG_SEARCH_PATH = "/health", "/v1/search/text"
FD_HEALTH_PATH, FD_DIAGNOSE_PATH = "/", "/diagnose"
HEALTH_TIMEOUT, RAG_SEARCH_TIMEOUT, FD_DIAGNOSE_TIMEOUT = 5.0, 30.0, 120.0
MAX_RETRIES, RETRY_BACKOFF = 2, 1.5
SIGNAL_WINDOW_SIZE, SIGNAL_SAMPLING_RATE = 1024, 50000
FAULT_LABELS = {
    0: {"name": "内圈故障", "name_en": "Inner Race Fault"},
    1: {"name": "外圈故障", "name_en": "Outer Race Fault"},
    2: {"name": "滚动体故障", "name_en": "Ball Fault"},
}


def class_labels(schema: str = "faults-v1") -> dict:
    """Return a fresh, explicit ontology; healthy is opt-in, never unknown/OOD."""
    if not isinstance(schema, str) or schema not in ("faults-v1", "bearing-health-v1"):
        raise ValueError("不支持的 label_schema")
    labels = {key: dict(value) for key, value in FAULT_LABELS.items()}
    if schema == "bearing-health-v1":
        labels[3] = {"name": "健康", "name_en": "Healthy"}
    return labels


class MissingAPIKeyError(RuntimeError):
    """A required optional provider credential was not supplied."""


def get_api_key(required: bool = False) -> Optional[str]:
    key = os.environ.get(ENV_KEY_NAME, "").strip()
    if not key:
        if required:
            raise MissingAPIKeyError(f"环境变量 {ENV_KEY_NAME} 未设置；请参考 .env.example")
        return None
    return key


@dataclass(frozen=True)
class ServiceEndpoint:
    name: str
    base_url: str
    health_path: str
    health_timeout: float = HEALTH_TIMEOUT
    extra: dict = field(default_factory=dict)

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")

    @property
    def health_url(self) -> str:
        return self.url(self.health_path)

    def validate(self) -> None:
        if not isinstance(self.base_url, str):
            raise ValueError(f"{self.name} base_url 必须为 HTTP(S) URL 字符串")
        try:
            parsed = urlparse(self.base_url)
        except ValueError as exc:
            raise ValueError(f"{self.name} base_url 格式非法") from exc
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError(f"{self.name} base_url 必须是无凭据、查询参数和片段的 HTTP(S) URL")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError(f"{self.name} base_url 端口非法") from exc


def _output_path(name: str) -> str:
    return str(Path.cwd() / "diag-evidence-output" / name)


@dataclass(frozen=True)
class Settings:
    rag_base_url: str = "http://127.0.0.1:8000"
    fd_base_url: str = "http://127.0.0.1:5000"
    ui_base_url: str = "http://127.0.0.1:6006"
    signal_window_size: int = SIGNAL_WINDOW_SIZE
    sampling_rate: float = SIGNAL_SAMPLING_RATE
    reports_dir: str = field(default_factory=lambda: _output_path("reports"))
    logs_dir: str = field(default_factory=lambda: _output_path("logs"))
    health_timeout: float = HEALTH_TIMEOUT
    rag_search_timeout: float = RAG_SEARCH_TIMEOUT
    fd_diagnose_timeout: float = FD_DIAGNOSE_TIMEOUT
    max_retries: int = MAX_RETRIES
    retry_backoff: float = RETRY_BACKOFF
    label_schema: str = "faults-v1"

    def __post_init__(self) -> None:
        class_labels(self.label_schema)
        for endpoint in self.services:
            endpoint.validate()
        if (isinstance(self.signal_window_size, bool)
                or not isinstance(self.signal_window_size, int) or self.signal_window_size < 16):
            raise ValueError("signal_window_size 必须是至少 16 的整数")
        for name in ("sampling_rate", "health_timeout", "rag_search_timeout", "fd_diagnose_timeout"):
            value = getattr(self, name)
            try:
                valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0
            except (OverflowError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"{name} 必须为正的有限数值")
        if self.sampling_rate > 1e12:
            raise ValueError("sampling_rate 超出支持范围（1e12 Hz）")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries 必须是 0..10 的整数")
        if (isinstance(self.retry_backoff, bool) or not isinstance(self.retry_backoff, (int, float))
                or not 0 <= self.retry_backoff <= 60 or not math.isfinite(self.retry_backoff)):
            raise ValueError("retry_backoff 必须在 0..60 秒之间")
        if not self.reports_dir or not self.logs_dir:
            raise ValueError("reports_dir 和 logs_dir 不能为空")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env
        kwargs = {}
        converters = {"signal_window_size": int, "sampling_rate": float,
                      "health_timeout": float, "rag_search_timeout": float,
                      "fd_diagnose_timeout": float, "max_retries": int, "retry_backoff": float}
        for name in cls.__dataclass_fields__:
            key = "DIAG_EVIDENCE_" + name.upper()
            if key in source:
                try:
                    kwargs[name] = converters.get(name, str)(source[key])
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError(f"环境变量 {key} 格式不合法") from exc
        return cls(**kwargs)

    @property
    def rag_service(self) -> ServiceEndpoint:
        return ServiceEndpoint("rag", self.rag_base_url, RAG_HEALTH_PATH, self.health_timeout)

    @property
    def fd_service(self) -> ServiceEndpoint:
        return ServiceEndpoint("fd", self.fd_base_url, FD_HEALTH_PATH, self.health_timeout)

    @property
    def ui_service(self) -> ServiceEndpoint:
        return ServiceEndpoint("ui", self.ui_base_url, "/", self.health_timeout)

    @property
    def services(self) -> tuple[ServiceEndpoint, ...]:
        return self.rag_service, self.fd_service, self.ui_service

    def ensure_dirs(self) -> None:
        for directory in (self.reports_dir, self.logs_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        return {"rag_base_url": self.rag_base_url, "fd_base_url": self.fd_base_url,
                "ui_base_url": self.ui_base_url, "signal_window_size": self.signal_window_size,
                "sampling_rate": self.sampling_rate, "label_schema": self.label_schema,
                "reports_dir": str(self.reports_dir),
                "logs_dir": str(self.logs_dir), "api_key_present": bool(get_api_key())}


def resolve_config(config: Settings | None = None) -> Settings:
    return config if config is not None else Settings.from_env()


# Compatibility names for v0.0 callers. New code passes a Settings instance.
# Legacy constants describe defaults. Reading the process environment here would
# make even importing the package fail before CLI configuration errors are caught.
_LEGACY = Settings()
RAG_BASE_URL, FD_BASE_URL, UI_BASE_URL = _LEGACY.rag_base_url, _LEGACY.fd_base_url, _LEGACY.ui_base_url
RAG_SERVICE, FD_SERVICE, UI_SERVICE = _LEGACY.services
SERVICES = _LEGACY.services
REPORTS_DIR, LOGS_DIR = _LEGACY.reports_dir, _LEGACY.logs_dir
PROJECT_ROOT = os.getenv("DIAG_EVIDENCE_PROJECT_ROOT", "")
AGENT_ROOT = str(Path(__file__).resolve().parents[1])


def ensure_dirs() -> None:
    resolve_config().ensure_dirs()


def describe() -> dict:
    return resolve_config().describe()
