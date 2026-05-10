from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModuleConfig:
    enabled: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResearchConfig:
    dreamcoder: ModuleConfig = field(default_factory=ModuleConfig)
    world_model: ModuleConfig = field(default_factory=ModuleConfig)
    meta_harness: ModuleConfig = field(default_factory=ModuleConfig)
    planner: ModuleConfig = field(default_factory=ModuleConfig)
    # B1: default ModuleConfig has ``enabled=False`` but the store is
    # treated as enabled-by-default when this field is *absent* (legacy
    # behaviour); the explicit flag lets callers opt out via
    # ``hypothesis_store.enabled=False``. ``ModuleRegistry.load`` reads
    # ``config.hypothesis_store.enabled`` with a ``True`` default.
    hypothesis_store: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(enabled=True, params={})
    )
    log_dir: str = "research_logs"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchConfig":
        cfg = cls()
        for name in ("dreamcoder", "world_model", "meta_harness", "planner"):
            if name in data:
                value = data[name] or {}
                setattr(
                    cfg,
                    name,
                    ModuleConfig(
                        enabled=bool(value.get("enabled", False)),
                        params=dict(value.get("params", {})),
                    ),
                )
        # hypothesis_store follows the same shape but defaults to enabled=True
        # when absent, so existing YAML configs keep the M1–M7 chain live.
        if "hypothesis_store" in data:
            hs_value = data["hypothesis_store"] or {}
            cfg.hypothesis_store = ModuleConfig(
                enabled=bool(hs_value.get("enabled", True)),
                params=dict(hs_value.get("params", {})),
            )
        cfg.log_dir = str(data.get("log_dir", cfg.log_dir))
        return cfg

    def active_modules(self) -> list[str]:
        return [
            name
            for name in ("dreamcoder", "world_model", "meta_harness", "planner")
            if getattr(self, name).enabled
        ]


def load_research_config(
    config_path: str | os.PathLike[str] | None = None,
) -> ResearchConfig:
    env_json = os.environ.get("ARC_RESEARCH_CONFIG", "").strip()
    if env_json:
        return ResearchConfig.from_dict(json.loads(env_json))

    env_path = os.environ.get("ARC_RESEARCH_CONFIG_PATH", "").strip()
    if env_path:
        config_path = env_path

    if config_path is None:
        config_path = (
            Path(__file__).resolve().parent / "config" / "default.yaml"
        )
    path = Path(config_path)
    if not path.exists():
        return ResearchConfig()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return ResearchConfig.from_dict(data)
