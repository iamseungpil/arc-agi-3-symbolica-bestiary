"""Arcgentica-native research extensions.

These modules are optional guests on top of the upstream Arcgentica host.
When all modules are disabled, the runtime should behave like the native
Arcgentica baseline except for no-op hook dispatch.
"""

from .config import ModuleConfig, ResearchConfig, load_research_config
from .registry import ModuleRegistry, ResearchRuntimeContext

__all__ = [
    "ModuleConfig",
    "ResearchConfig",
    "ResearchRuntimeContext",
    "ModuleRegistry",
    "load_research_config",
]
