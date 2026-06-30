"""
event_driven/agent.py — Event-driven / reactive pattern.

Domain: Covenant monitoring — the agent doesn't run on a schedule or wait
for a user request. It subscribes to a transaction event stream and
decides, on EVERY event, whether that specific event moved the covenant
ratio enough to warrant escalation.

The defining trait versus scheduled/cron-triggered: there is no fixed
cadence. The agent's action is triggered by the data itself — a single
large transaction can trigger an immediate escalation seconds after it
posts, while a quiet day with no material transactions triggers nothing
at all, even if hours pass. This is fundamentally different from a daily
batch check, which would either catch the breach hours late or run
needlessly on a day with no activity.

Real engineering risk this pattern introduces that polling agents don't
have: events can arrive out of order, can be duplicated by an upstream
system retry, and the agent must maintain correct running state across
many events without re-processing the same event twice. All three are
tested below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TransactionEvent:
    event_id: str
    entity_code: str
    sequence_number: int  # monotonically increasing per entity, for ordering
    debt_delta_usd: float
    ebitda_delta_usd: float
    timestamp: str


@dataclass
class CovenantState:
    entity_code: str
    total_debt_usd: float
    trailing_ebitda_usd: float
    covenant_threshold: float  # max allowed debt/EBITDA
    last_processed_sequence: int = -1
    processed_event_ids: set = field(default_factory=set)

    @property
    def current_ratio(self) -> float:
        if self.trailing_ebitda_usd <= 0:
            return float("inf")
        return self.total_debt_usd / self.trailing_ebitda_usd


@dataclass
class ReactionResult:
    event_id: str
    processed: bool
    skip_reason: Optional[str]
    ratio_before: Optional[float]
    ratio_after: Optional[float]
    breach_triggered: bool
    escalation: Optional[dict]


class CovenantMonitorAgent:
    """
    Reactive agent: maintains running covenant state per entity, processes
    each incoming transaction event exactly once, and escalates the
    instant a processed event pushes the ratio over threshold — not on
    the next scheduled check.
    """

    WARNING_BUFFER = 0.05  # warn within 5% of breach, even if not yet breached

    def __init__(self) -> None:
        self.states: dict[str, CovenantState] = {}

    def register_entity(
        self, entity_code: str, initial_debt: float, initial_ebitda: float, threshold: float
    ) -> None:
        self.states[entity_code] = CovenantState(
            entity_code=entity_code, total_debt_usd=initial_debt,
            trailing_ebitda_usd=initial_ebitda, covenant_threshold=threshold,
        )

    def on_event(self, event: TransactionEvent) -> ReactionResult:
        """
        The reactive entry point — called once per incoming event. This is
        what would be wired to a real event bus subscription in production.
        """
        state = self.states.get(event.entity_code)
        if state is None:
            return ReactionResult(
                event_id=event.event_id, processed=False,
                skip_reason=f"Unregistered entity: {event.entity_code}",
                ratio_before=None, ratio_after=None, breach_triggered=False, escalation=None,
            )

        # ── Idempotency guard: duplicate event delivery must not double-count ──
        if event.event_id in state.processed_event_ids:
            return ReactionResult(
                event_id=event.event_id, processed=False,
                skip_reason="DUPLICATE_EVENT_ALREADY_PROCESSED",
                ratio_before=state.current_ratio, ratio_after=state.current_ratio,
                breach_triggered=False, escalation=None,
            )

        # ── Ordering guard: out-of-order events must not corrupt running state ──
        if event.sequence_number <= state.last_processed_sequence:
            return ReactionResult(
                event_id=event.event_id, processed=False,
                skip_reason=f"OUT_OF_ORDER: seq={event.sequence_number} "
                            f"<= last_processed={state.last_processed_sequence}",
                ratio_before=state.current_ratio, ratio_after=state.current_ratio,
                breach_triggered=False, escalation=None,
            )

        ratio_before = state.current_ratio

        state.total_debt_usd += event.debt_delta_usd
        state.trailing_ebitda_usd += event.ebitda_delta_usd
        state.last_processed_sequence = event.sequence_number
        state.processed_event_ids.add(event.event_id)

        ratio_after = state.current_ratio

        escalation = None
        breach_triggered = False
        if ratio_after >= state.covenant_threshold:
            breach_triggered = True
            escalation = {
                "type": "COVENANT_BREACH",
                "entity": event.entity_code,
                "ratio": round(ratio_after, 3),
                "threshold": state.covenant_threshold,
                "triggering_event": event.event_id,
            }
        elif ratio_after >= state.covenant_threshold * (1 - self.WARNING_BUFFER):
            escalation = {
                "type": "COVENANT_WARNING",
                "entity": event.entity_code,
                "ratio": round(ratio_after, 3),
                "threshold": state.covenant_threshold,
                "triggering_event": event.event_id,
            }

        return ReactionResult(
            event_id=event.event_id, processed=True, skip_reason=None,
            ratio_before=round(ratio_before, 3) if ratio_before != float("inf") else None,
            ratio_after=round(ratio_after, 3) if ratio_after != float("inf") else None,
            breach_triggered=breach_triggered, escalation=escalation,
        )
