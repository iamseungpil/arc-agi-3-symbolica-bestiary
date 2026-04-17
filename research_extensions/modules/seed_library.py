"""Abstract-primitive seed library for DreamCoder.

Why this exists
---------------
DreamCoder's library should not start empty. The original DreamCoder paper
(Ellis et al., PLDI 2021) seeds its library with primitive combinators
(map, fold, +, ...) so the wake-sleep loop has named building blocks to
compose with. Without seeded primitives, the agent re-derives the same
basic discovery routines from scratch on every run and wastes turn budget.

What the user asked for
-----------------------
"BFS나 각 버튼의 의미 같이 추상화된 개념을 skill에 넣는 거야."
=> Bake BFS and per-button-semantic-discovery into the library as
abstract primitives the agent can reference by name.

Seeded entries
--------------
1. ``bfs-explore-grid`` — generic frontier exploration controller. The
   agent does not have a Python interpreter, so the body is an
   action-grounded controller: pick the least-tried available action,
   advance one step, and record outcome. This is BFS over the observable
   state graph (not raw grid coords).
2. ``discover-ACTIONk-semantics`` for k in 1..7 — one per possible button.
   Each is a controller that takes ACTIONk once from a stable state and
   asks the agent to write a one-line semantic label (move/toggle/no-op/
   rotate/...) into world-model notes.
3. ``probe-rival-discriminator`` — falsification-first primitive: pick the
   action that would maximally separate the current best vs second-best
   world hypothesis. CWM's REx tree-search uses Thompson sampling over
   rival programs; this primitive is the agent-facing analogue.
4. ``commit-validated-opening`` — wrapper that points at the most
   reliable validated exact spine, with a falsification branch when its
   expected effect is weaker than recently observed.

These are *abstract*: they reference action families and the agent's own
running notes rather than fixed action tuples. Because the routing logic
in ``dreamcoder.py`` only fires on action-grounded controllers, each
seeded entry includes one or two example ACTION tokens in its
``action_spine`` so the routing layer recognizes them as reusable.
"""
from __future__ import annotations

from typing import Any


def _bfs_skill() -> dict[str, Any]:
    return {
        "name": "bfs-explore-grid",
        "kind": "abstract_primitive",
        "description": (
            "Breadth-first exploration over observable states. From the current "
            "observable state, advance with the least-used available action, treat "
            "the resulting observation family as the next frontier node, and only "
            "deepen along an action that has produced a new family at least once."
        ),
        "precondition": (
            "Current state has 2 or more available actions and at least one of them "
            "has not been tried yet from this observable state, OR the recent action "
            "window keeps re-entering the same observation family."
        ),
        "controller": (
            "1) List untried actions from this state. "
            "2) If any exist, pick the lexicographically first untried one and submit it. "
            "3) Else pick the action with the lowest local count and the highest "
            "branch-escape rate from the world-model bucket. "
            "4) After acting, register a one-line note about the observed family change "
            "so future runs can route here without re-exploring."
        ),
        "action_spine": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        "expected_effect": (
            "Reach a previously-unseen observation family within at most k steps where "
            "k = number of unseen actions from this state."
        ),
        "failure_recovery": (
            "If 3 consecutive steps return to the same family without new signature, "
            "stop the BFS expansion and switch to `probe-rival-discriminator` so the "
            "next action falsifies a hypothesis instead of expanding the same frontier."
        ),
    }


def _discover_button_semantics_skills() -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    for k in range(1, 8):
        action = f"ACTION{k}"
        skills.append(
            {
                "name": f"discover-{action}-semantics",
                "kind": "abstract_primitive",
                "description": (
                    f"Characterize what {action} does in this game. The agent does not "
                    f"know in advance whether {action} is a move, a toggle, a rotate, "
                    "a select, or a no-op. This primitive runs one isolated probe "
                    f"and records the observed semantic label for {action}."
                ),
                "precondition": (
                    f"World-model notes do not yet contain a semantic label for {action} "
                    "in the current game, and the current state is stable (no recent "
                    "surprise pending) so the probe outcome is interpretable."
                ),
                "controller": (
                    f"1) Submit {action} exactly once from the current stable state. "
                    "2) Compare the resulting grid against the previous one: same/move/"
                    "toggle/rotate/grow/shrink/select/no-op. "
                    "3) Append a `world_update` line of the form "
                    f"`semantic[{action}] = <label>` so subsequent skills can compose "
                    "on top of the named meaning."
                ),
                "action_spine": [action],
                "expected_effect": (
                    f"World-model notes gain a `semantic[{action}]` entry and the agent "
                    "can subsequently reference that label in higher-level skill bodies."
                ),
                "failure_recovery": (
                    "If the observed effect is ambiguous (state already changing from a "
                    "lingering animation, or no diff at all), retry once after one "
                    "neutral step; if still ambiguous, mark the semantic as "
                    f"`semantic[{action}] = unknown-needs-context` and move on."
                ),
            }
        )
    return skills


def _rival_discriminator_skill() -> dict[str, Any]:
    return {
        "name": "probe-rival-discriminator",
        "kind": "abstract_primitive",
        "description": (
            "When two or more rival world hypotheses fit the observations equally, take "
            "the action that would falsify the stronger of the two. This is the "
            "agent-facing analogue of CWM's REx Thompson sampling over rival programs: "
            "spend the next action on falsification rather than confirmation."
        ),
        "precondition": (
            "At least one entry in `rival_predictions` is unresolved on the most "
            "recent prediction, OR `world_model` has two drafts whose simulator "
            "predictions disagree for some action available right now."
        ),
        "controller": (
            "1) Inspect rival predictions and shortlist actions for which the "
            "predictions diverge. "
            "2) Pick the action whose outcome would falsify the higher-scoring rival. "
            "3) Submit that action and register the outcome as either "
            "`falsified <rival_name>` or `supported <rival_name>` in world notes."
        ),
        "action_spine": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        "expected_effect": (
            "Either the leading hypothesis is falsified (rare but high-value) or it is "
            "promoted to `supported`. In either case, the world-model draft becomes "
            "more selective for the next action."
        ),
        "failure_recovery": (
            "If the agent cannot articulate two rival predictions, fall back to "
            "`bfs-explore-grid` so at least the action carries information value."
        ),
    }


def _commit_validated_opening_wrapper() -> dict[str, Any]:
    return {
        "name": "commit-validated-opening",
        "kind": "abstract_primitive",
        "description": (
            "Wrapper around the highest-scoring validated exact spine. Reuse the "
            "stored opening on a fresh run, but compare its observed effect to "
            "expectations and branch out if it underperforms. Equivalent to "
            "DreamCoder's library reuse + recognition prior on a previously fit "
            "program."
        ),
        "precondition": (
            "Fresh run with no stronger opening evidence, AND the available actions "
            "match the action_spine of the most recently validated exact spine."
        ),
        "controller": (
            "1) Look up the stored opening spine in this library. "
            "2) Replay it action-by-action, comparing each step's observed diff to "
            "the spine's recorded `expected_effect`. "
            "3) If 2 consecutive steps are weaker than expected, abort and switch to "
            "`probe-rival-discriminator`."
        ),
        "action_spine": ["ACTION1", "ACTION2"],
        "expected_effect": (
            "Land in the same opening checkpoint region as a prior validated run "
            "while leaving the agent free to branch the moment evidence diverges."
        ),
        "failure_recovery": (
            "If no validated spine exists yet, propose `bfs-explore-grid` instead and "
            "retry next turn."
        ),
        "subskills": ["skill:bfs-explore-grid"],
    }


def abstract_primitive_seeds() -> list[dict[str, Any]]:
    """Return the full ordered list of seed payloads.

    Order matters because DreamCoder's display ranking puts earlier-added
    structured skills near the top of the prompt overlay when scores tie.
    BFS comes first because it is the safest default for a fresh game,
    then per-button discovery, then the rival discriminator and the
    opening wrapper."""
    seeds: list[dict[str, Any]] = [_bfs_skill()]
    seeds.extend(_discover_button_semantics_skills())
    seeds.append(_rival_discriminator_skill())
    seeds.append(_commit_validated_opening_wrapper())
    return seeds
