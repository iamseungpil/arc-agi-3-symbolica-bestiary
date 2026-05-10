from typing import Type
import logging

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm

load_dotenv()
logger = logging.getLogger(__name__)

# Safe imports — only load templates that actually work in this environment
def _safe_import(name: str, module_path: str, cls_names: list[str]) -> dict[str, Type[Agent]]:
    try:
        import importlib
        mod = importlib.import_module(module_path, package="agents")
        return {n.lower(): getattr(mod, n) for n in cls_names if hasattr(mod, n)}
    except Exception as e:
        logger.debug(f"Skipping agent template {name}: {e}")
        return {}

_agents: dict[str, Type[Agent]] = {}
_agents.update(_safe_import("random", ".templates.random_agent", ["Random"]))
_agents.update(_safe_import("llm", ".templates.llm_agents", ["LLM", "FastLLM", "GuidedLLM", "ReasoningLLM"]))
_agents.update(_safe_import("reasoning", ".templates.reasoning_agent", ["ReasoningAgent"]))
_agents.update(_safe_import("multimodal", ".templates.multimodal", ["MultiModalLLM"]))
_agents.update(_safe_import("agentica", ".templates.agentica", ["Arcgentica"]))
_agents.update(_safe_import("agentica_simple", ".templates.agentica_simple", ["ArcgenticaSimple"]))
_agents.update(_safe_import("agentica_v57", ".templates.agentica_v57", ["ArcgenticaV57"]))
_agents.update(_safe_import("langgraph_func", ".templates.langgraph_functional_agent", ["LangGraphFunc", "LangGraphTextOnly"]))
_agents.update(_safe_import("langgraph_random", ".templates.langgraph_random_agent", ["LangGraphRandom"]))
_agents.update(_safe_import("langgraph_thinking", ".templates.langgraph_thinking", ["LangGraphThinking"]))
_agents.update(_safe_import("smolagents", ".templates.smolagents", ["SmolCodingAgent", "SmolVisionAgent"]))
_agents.update(
    _safe_import("arcgentica_research", ".templates.arcgentica_research", ["ArcgenticaResearch"])
)
_agents.update(
    _safe_import("agentica_lite", ".templates.agentica_lite.adapter", ["ArcgenticaLiteAgent"])
)

AVAILABLE_AGENTS: dict[str, Type[Agent]] = _agents

if "arcgenticaresearch" in AVAILABLE_AGENTS:
    AVAILABLE_AGENTS["research"] = AVAILABLE_AGENTS["arcgenticaresearch"]
    AVAILABLE_AGENTS["arcgentica_research"] = AVAILABLE_AGENTS["arcgenticaresearch"]
if "arcgenticasimple" in AVAILABLE_AGENTS:
    AVAILABLE_AGENTS["simple"] = AVAILABLE_AGENTS["arcgenticasimple"]
    AVAILABLE_AGENTS["arcgentica_simple"] = AVAILABLE_AGENTS["arcgenticasimple"]
    AVAILABLE_AGENTS["symbolica_simple"] = AVAILABLE_AGENTS["arcgenticasimple"]
if "arcgenticav57" in AVAILABLE_AGENTS:
    AVAILABLE_AGENTS["v57"] = AVAILABLE_AGENTS["arcgenticav57"]
    AVAILABLE_AGENTS["agentica_v57"] = AVAILABLE_AGENTS["arcgenticav57"]
if "arcgenticaliteagent" in AVAILABLE_AGENTS:
    AVAILABLE_AGENTS["lite"] = AVAILABLE_AGENTS["arcgenticaliteagent"]
    AVAILABLE_AGENTS["agentica_lite"] = AVAILABLE_AGENTS["arcgenticaliteagent"]
    AVAILABLE_AGENTS["v601"] = AVAILABLE_AGENTS["arcgenticaliteagent"]
    AVAILABLE_AGENTS["v602"] = AVAILABLE_AGENTS["arcgenticaliteagent"]

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

__all__ = [
    "Swarm",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
