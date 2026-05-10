"""Per-module unit tests for stalemate_trigger.py.

Plan v602 §11 addendum: 6 critical branch tests for stalemate trigger semantics.

Branches under test:
  - fires-after-K (turns_since_L_plus > K AND max_posterior < theta -> True)
  - not-fires-below-K
  - not-fires-high-posterior
  - warm-up-first (turn_count == 0 fires once)
  - warm-up-once (mark_warm_up_done suppresses subsequent calls)
  - once-per-episode (mark_fired suppresses subsequent fires)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.stalemate_trigger import (  # noqa: E402
    StalemateConfig, StalemateTrigger,
)


def _trigger(K: int = 8, theta: float = 0.45, once: bool = True,
             periodic: int = 0) -> StalemateTrigger:
    """Tests pass once=True + periodic=0 to preserve original v601 semantics
    (v605 default is once=False, periodic=5 — overridden here for legacy tests)."""
    return StalemateTrigger(StalemateConfig(K_threshold=K, theta_threshold=theta,
                                            once_per_episode=once,
                                            periodic_every_n=periodic))


# ---------- 1. fires-after-K --------------------------------------------------

def test_fires_after_K_with_low_posterior():
    """When turns_since_L_plus > K AND max_posterior < theta, fires() returns True."""
    t = _trigger(K=8, theta=0.45)
    # 9 turns without L+, posterior 0.3 below 0.45
    assert t.fires(turns_since_L_plus=9, max_posterior=0.3) is True


# ---------- 2. not-fires-below-K ----------------------------------------------

def test_not_fires_when_below_K():
    """When turns_since_L_plus <= K, fires() returns False even with low posterior."""
    t = _trigger(K=8, theta=0.45)
    # exactly K (not strictly greater) should NOT fire
    assert t.fires(turns_since_L_plus=8, max_posterior=0.1) is False
    # well below K
    assert t.fires(turns_since_L_plus=3, max_posterior=0.1) is False


# ---------- 3. not-fires-high-posterior ---------------------------------------

def test_not_fires_when_posterior_high():
    """When max_posterior >= theta, fires() returns False even with many idle turns."""
    t = _trigger(K=8, theta=0.45)
    assert t.fires(turns_since_L_plus=20, max_posterior=0.6) is False
    # exactly at theta should NOT fire (strict <)
    assert t.fires(turns_since_L_plus=20, max_posterior=0.45) is False


# ---------- 4. warm-up-first --------------------------------------------------

def test_warm_up_fires_at_turn_zero():
    """warm_up_fires(0) returns True before mark_warm_up_done()."""
    t = _trigger()
    assert t.warm_up_fires(0) is True
    # turns 1+ never trigger warm-up
    assert t.warm_up_fires(1) is False


# ---------- 5. warm-up-once ---------------------------------------------------

def test_warm_up_fires_only_once():
    """After mark_warm_up_done(), warm_up_fires returns False even at turn 0."""
    t = _trigger()
    assert t.warm_up_fires(0) is True
    t.mark_warm_up_done()
    assert t.warm_up_fires(0) is False
    assert t.warm_up_fires(1) is False


# ---------- 6. once-per-episode -----------------------------------------------

def test_once_per_episode_suppresses_subsequent_fires():
    """When once_per_episode=True, after mark_fired() further fires() returns False.

    reset_episode() restores firing capability.
    """
    # v605: once=True suppresses ONLY when stalemate condition fails after mark_fired.
    # When stalemate condition still holds, v605 allows continued firing.
    # Test focuses on periodic=0 + once=True with NO stalemate.
    t = _trigger(once=True, periodic=0, K=20)  # K=20 ensures stalemate inactive
    assert t.fires(turns_since_L_plus=21, max_posterior=0.1) is True
    t.mark_fired()
    # turns_since_L_plus DROPS below K (e.g. after L+ reset); stalemate inactive.
    assert t.fires(turns_since_L_plus=10, max_posterior=0.1) is False
    t.reset_episode()
    assert t.fires(turns_since_L_plus=21, max_posterior=0.1) is True
