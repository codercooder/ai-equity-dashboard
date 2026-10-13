from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_events import MARKET_EVENTS

BASE_DIR = Path(__file__).resolve().parent
AUTO_EVENTS_PATH = BASE_DIR / "data" / "auto_market_events.json"

CATEGORY_STYLE = {
    "利率": ("rgba(220,38,38,0.42)", "rgba(220,38,38,0.055)", "rgba(220,38,38,0.25)"),
    "宏观": ("rgba(107,114,128,0.42)", "rgba(107,114,128,0.07)", "rgba(107,114,128,0.25)"),
    "AI": ("rgba(124,58,237,0.44)", "rgba(124,58,237,0.06)", "rgba(124,58,237,0.24)"),
    "AI监管": ("rgba(168,85,247,0.44)", "rgba(168,85,247,0.06)", "rgba(168,85,247,0.24)"),
    "地缘政治": ("rgba(244,63,94,0.42)", "rgba(244,63,94,0.05)", "rgba(244,63,94,0.22)"),
    "能源": ("rgba(234,88,12,0.44)", "rgba(234,88,12,0.06)", "rgba(234,88,12,0.24)"),
    "贸易": ("rgba(202,138,4,0.44)", "rgba(234,179,8,0.07)", "rgba(202,138,4,0.28)"),
    "金融稳定": ("rgba(245,158,11,0.44)", "rgba(245,158,11,0.07)", "rgba(245,158,11,0.30)"),
}


def _coerce_event(event: dict[str, Any], *, auto_generated: bool) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    kind = str(event.get("kind") or "point").lower()
    if kind not in {"point", "range"}:
        return None

    e = dict(event)
    e["kind"] = kind
    e.setdefault("category", "其他")
    e.setdefault("importance", "High")
    e.setdefault("status", "confirmed" if not auto_generated else "auto_high")
    e.setdefault("auto_generated", auto_generated)
    e.setdefault("confidence", 1.0 if not auto_generated else 0.8)

    point_color, range_color, line_color = CATEGORY_STYLE.get(
        e["category"],
        ("rgba(75,85,99,0.42)", "rgba(107,114,128,0.07)", "rgba(107,114,128,0.25)"),
    )
    if kind == "point":
        e.setdefault("color", point_color)
        date = e.get("date")
        if not date:
            return None
        try:
            e["date"] = pd.Timestamp(date).strftime("%Y-%m-%d")
        except Exception:
            return None
    else:
        start = e.get("start")
        if not start:
            return None
        try:
            e["start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
        except Exception:
            return None
        if e.get("end"):
            try:
                e["end"] = pd.Timestamp(e["end"]).strftime("%Y-%m-%d")
            except Exception:
                e["end"] = None
        e.setdefault("color", range_color)
        e.setdefault("line_color", line_color)

    e.setdefault("source", "")
    e.setdefault("source_url", "")
    e.setdefault("label", "重要事件")
    return e


def load_auto_event_payload() -> dict[str, Any]:
    if not AUTO_EVENTS_PATH.exists():
        return {"updated_at": None, "events": [], "scheduled_events": []}
    try:
        payload = json.loads(AUTO_EVENTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        payload.setdefault("updated_at", None)
        payload.setdefault("events", [])
        payload.setdefault("scheduled_events", [])
        return payload
    except Exception:
        return {"updated_at": None, "events": [], "scheduled_events": []}


def _event_key(event: dict[str, Any]) -> str:
    if event.get("event_id"):
        return str(event["event_id"])
    when = event.get("date") or f"{event.get('start')}|{event.get('end')}"
    return f"{event.get('category','')}|{when}|{event.get('label','')}".lower()


def get_market_events(
    *,
    include_auto: bool = True,
    high_only: bool = True,
) -> list[dict[str, Any]]:
    """Return curated + persisted auto-discovered events for chart overlays.

    Scheduled calendar events are intentionally not returned here; the overlay is
    meant to explain realized market history rather than crowd the chart with
    future dates. They remain available in the JSON payload for future UI use.
    """
    merged: dict[str, dict[str, Any]] = {}

    for raw in MARKET_EVENTS:
        event = _coerce_event(raw, auto_generated=False)
        if event:
            merged[_event_key(event)] = event

    if include_auto:
        payload = load_auto_event_payload()
        for raw in payload.get("events", []):
            event = _coerce_event(raw, auto_generated=True)
            if not event:
                continue
            if str(event.get("status", "")).lower() not in {"auto_high", "confirmed"}:
                continue
            if high_only and str(event.get("importance", "High")).lower() != "high":
                continue
            merged.setdefault(_event_key(event), event)

    events = list(merged.values())

    def sort_key(e: dict[str, Any]):
        raw_date = e.get("date") or e.get("start") or "1900-01-01"
        try:
            return pd.Timestamp(raw_date)
        except Exception:
            return pd.Timestamp("1900-01-01")

    return sorted(events, key=sort_key)


def auto_event_metadata() -> dict[str, Any]:
    payload = load_auto_event_payload()
    return {
        "updated_at": payload.get("updated_at"),
        "event_count": len(payload.get("events", []) or []),
        "scheduled_count": len(payload.get("scheduled_events", []) or []),
    }
