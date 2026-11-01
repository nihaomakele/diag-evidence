"""Diagnostic evidence workflow research toolkit."""
__version__ = "0.2.0.dev4"

# Keep import cheap; avoid loading NumPy and requests until the API is used.
def __getattr__(name):
    if name in {"DiagnosisAgent", "run_agent"}:
        from .agent import DiagnosisAgent, run_agent
        return {"DiagnosisAgent": DiagnosisAgent, "run_agent": run_agent}[name]
    if name == "Settings":
        from .haconfig.settings import Settings
        return Settings
    raise AttributeError(name)

__all__ = ["DiagnosisAgent", "run_agent", "Settings", "__version__"]
