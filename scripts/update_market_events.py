from __future__ import annotations

import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "auto_market_events.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AIEquityDashboardEventBot/1.0; +https://github.com/)"
}
TIMEOUT = 20

PRIORITY_SOURCES = (
    "reuters", "bloomberg", "financial times", "wall street journal", "cnbc",
    "associated press", "ap news", "white house", "federal reserve", "bls",
    "bureau of labor statistics", "commerce department", "federal register",
    "nvidia", "openai", "deepseek", "european commission",
)

HIGH_IMPACT_WORDS = (
    "launch", "launches", "launched", "unveil", "unveils", "unveiled", "release", "released",
    "rate cut", "cuts rates", "rate hike", "raises rates", "hikes rates",
    "executive order", "final rule", "export control", "ban", "sanction", "tariff",
    "attack", "strikes", "invasion", "war", "ceasefire", "hormuz", "blockade",
    "oil surge", "oil jumps", "oil spikes", "supply disruption", "production cut",
    "hotter than expected", "cooler than expected", "inflation surprise", "cpi surprise",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(category: str, date: str, label: str) -> str:
    raw = f"{category}|{date}|{label}".lower().encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def request_text(url: str, *, params: dict[str, Any] | None = None) -> str:
    r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def request_bytes(url: str, *, params: dict[str, Any] | None = None) -> bytes:
    r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def load_existing() -> dict[str, Any]:
    if not DATA_PATH.exists():
        return {"updated_at": None, "events": [], "scheduled_events": []}
    try:
        payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
        payload.setdefault("events", [])
        payload.setdefault("scheduled_events", [])
        return payload
    except Exception:
        return {"updated_at": None, "events": [], "scheduled_events": []}


def google_news_items(query: str, days: int = 2) -> list[dict[str, Any]]:
    q = f"({query}) when:{days}d"
    try:
        content = request_bytes(
            "https://news.google.com/rss/search",
            params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        )
        root = ET.fromstring(content)
    except Exception:
        return []

    out = []
    for item in root.findall(".//item"):
        title = re.sub(r"\s+", " ", (item.findtext("title") or "").strip())
        if not title:
            continue
        source_node = item.find("source")
        source = (source_node.text or "Google News").strip() if source_node is not None else "Google News"
        link = (item.findtext("link") or "").strip()
        pub_raw = item.findtext("pubDate") or ""
        try:
            pub = parsedate_to_datetime(pub_raw).astimezone(timezone.utc)
        except Exception:
            pub = datetime.now(timezone.utc)
        out.append({"title": title, "source": source, "url": link, "published": pub})
    return out


def news_score(title: str, source: str, extra_keywords: tuple[str, ...] = ()) -> int:
    text = title.lower()
    src = source.lower()
    score = 35
    if any(x in src for x in PRIORITY_SOURCES):
        score += 25
    score += min(30, 10 * sum(1 for x in HIGH_IMPACT_WORDS if x in text))
    score += min(20, 10 * sum(1 for x in extra_keywords if x in text))
    return min(score, 100)


def choose_one_per_day(items: list[dict[str, Any]], *, category: str, label: str, extra_keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    best: dict[str, tuple[int, dict[str, Any]]] = {}
    for item in items:
        score = news_score(item["title"], item["source"], extra_keywords)
        if score < 70:
            continue
        day = item["published"].date().isoformat()
        if day not in best or score > best[day][0]:
            best[day] = (score, item)

    events = []
    for day, (score, item) in best.items():
        events.append({
            "event_id": stable_id(category, day, label),
            "kind": "point",
            "date": day,
            "label": label,
            "category": category,
            "importance": "High",
            "status": "auto_high",
            "confidence": round(score / 100.0, 2),
            "auto_generated": True,
            "source": item["source"],
            "source_url": item["url"],
            "headline": item["title"],
            "detected_at": now_iso(),
        })
    return events


def discover_news_events() -> list[dict[str, Any]]:
    themes = [
        {
            "query": 'DeepSeek model launch release R1 V3 artificial intelligence',
            "category": "AI",
            "label": "DeepSeek 重要发布 / 市场冲击",
            "keywords": ("deepseek", "r1", "v3", "model"),
        },
        {
            "query": 'NVIDIA new AI chip GPU Blackwell Rubin launch unveil',
            "category": "AI",
            "label": "NVIDIA 重要新品 / 架构发布",
            "keywords": ("nvidia", "blackwell", "rubin", "gpu", "chip"),
        },
        {
            "query": 'AI regulation export control chips executive order AI Act final rule',
            "category": "AI监管",
            "label": "AI监管 / 出口管制重大变化",
            "keywords": ("regulation", "rule", "export", "control", "ai act", "executive order"),
        },
        {
            "query": 'Federal Reserve FOMC rate cut hike interest rates decision',
            "category": "利率",
            "label": "FOMC 利率政策重大变化",
            "keywords": ("federal reserve", "fomc", "rate cut", "rate hike"),
        },
        {
            "query": 'US CPI hotter colder than expected inflation surprise markets',
            "category": "宏观",
            "label": "美国 CPI shock",
            "keywords": ("cpi", "inflation", "hotter", "cooler", "surprise"),
        },
        {
            "query": 'Iran Israel US war attack Hormuz oil supply disruption Brent',
            "category": "地缘政治",
            "label": "中东地缘 / 油价重大冲击",
            "keywords": ("iran", "israel", "hormuz", "oil", "attack", "strike"),
        },
        {
            "query": 'Russia Ukraine war sanctions energy oil escalation ceasefire',
            "category": "地缘政治",
            "label": "俄乌战争重大升级 / 缓和",
            "keywords": ("russia", "ukraine", "sanction", "ceasefire", "attack"),
        },
        {
            "query": 'US China tariffs trade war semiconductors export controls',
            "category": "贸易",
            "label": "贸易 / 关税 / 出口管制冲击",
            "keywords": ("tariff", "trade", "china", "export control", "semiconductor"),
        },
    ]

    events: list[dict[str, Any]] = []
    for theme in themes:
        items = google_news_items(theme["query"], days=2)
        events.extend(
            choose_one_per_day(
                items,
                category=theme["category"],
                label=theme["label"],
                extra_keywords=theme["keywords"],
            )
        )
    return events


def discover_nvidia_official() -> list[dict[str, Any]]:
    """Official NVIDIA Newsroom releases; only product/architecture launches qualify."""
    try:
        root = ET.fromstring(request_bytes("https://nvidianews.nvidia.com/releases.xml"))
    except Exception:
        return []

    product_words = ("blackwell", "rubin", "gpu", "platform", "accelerator", "ai factory", "dgx", "grace")
    action_words = ("launch", "unveil", "introduc", "announce")
    cutoff = datetime.now(timezone.utc).timestamp() - 3 * 86400
    out = []
    for item in root.findall(".//item"):
        title = re.sub(r"\s+", " ", (item.findtext("title") or "").strip())
        lower = title.lower()
        if not title or not any(x in lower for x in product_words) or not any(x in lower for x in action_words):
            continue
        pub_raw = item.findtext("pubDate") or ""
        try:
            pub = parsedate_to_datetime(pub_raw).astimezone(timezone.utc)
        except Exception:
            continue
        if pub.timestamp() < cutoff:
            continue
        day = pub.date().isoformat()
        out.append({
            "event_id": stable_id("AI", day, "NVIDIA 重要新品 / 架构发布"),
            "kind": "point",
            "date": day,
            "label": "NVIDIA 重要新品 / 架构发布",
            "category": "AI",
            "importance": "High",
            "status": "confirmed",
            "confidence": 0.98,
            "auto_generated": True,
            "source": "NVIDIA Newsroom",
            "source_url": (item.findtext("link") or "https://nvidianews.nvidia.com/").strip(),
            "headline": title,
            "detected_at": now_iso(),
        })
    return out


def parse_fomc_schedule() -> list[dict[str, Any]]:
    """Store upcoming FOMC decision dates from the official Fed calendar.

    These are persisted as scheduled events and are not drawn on the historical
    sector chart until a separate high-impact event is detected/confirmed.
    """
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        text = BeautifulSoup(request_text(url), "html.parser").get_text(" ", strip=True)
    except Exception:
        return []

    now = datetime.now(timezone.utc)
    current_year = now.year
    months = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    }
    out = []
    for year in (current_year, current_year + 1):
        start_marker = f"{year} FOMC Meetings"
        pos = text.find(start_marker)
        if pos < 0:
            continue
        tail = text[pos + len(start_marker):]
        # Stop before the next year's heading if present.
        next_marker = f"{year + 1} FOMC Meetings"
        next_pos = tail.find(next_marker)
        section = tail[:next_pos] if next_pos >= 0 else tail[:5000]
        pattern = re.compile(
            r"\b(" + "|".join(months) + r")\s+(\d{1,2})(?:-(\d{1,2}))?\*?\b"
        )
        seen = set()
        for match in pattern.finditer(section):
            month_name, first_day, second_day = match.groups()
            # Prevent statement/minutes release dates from being mistaken for meetings:
            # a meeting date should appear near the beginning of the month block and
            # is unique per month in the regular calendar.
            month = months[month_name]
            if month in seen:
                continue
            seen.add(month)
            end_day = int(second_day or first_day)
            try:
                d = datetime(year, month, end_day, tzinfo=timezone.utc)
            except ValueError:
                continue
            if d.date() < now.date():
                continue
            date_str = d.date().isoformat()
            out.append({
                "event_id": stable_id("利率", date_str, "FOMC 决议日"),
                "kind": "point",
                "date": date_str,
                "label": "FOMC 决议日",
                "category": "利率",
                "importance": "High",
                "status": "scheduled",
                "confidence": 1.0,
                "auto_generated": True,
                "source": "Federal Reserve",
                "source_url": url,
            })
    return out


def parse_cpi_schedule() -> list[dict[str, Any]]:
    url = "https://www.bls.gov/schedule/news_release/cpi.htm"
    try:
        soup = BeautifulSoup(request_text(url), "html.parser")
    except Exception:
        return []
    now = datetime.now(timezone.utc)
    out = []
    for row in soup.find_all("tr"):
        cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in row.find_all(["th", "td"])]
        if len(cells) < 3 or "Release Date" in cells:
            continue
        date_text = cells[1]
        parsed = None
        for fmt in ("%b. %d, %Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                parsed = datetime.strptime(date_text, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                pass
        if parsed is None or parsed.date() < now.date():
            continue
        date_str = parsed.date().isoformat()
        out.append({
            "event_id": stable_id("宏观", date_str, "美国 CPI 发布"),
            "kind": "point",
            "date": date_str,
            "label": "美国 CPI 发布",
            "category": "宏观",
            "importance": "Medium",
            "status": "scheduled",
            "confidence": 1.0,
            "auto_generated": True,
            "source": "U.S. Bureau of Labor Statistics",
            "source_url": url,
        })
    return out


def merge_events(old_events: list[dict[str, Any]], new_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for event in old_events + new_events:
        if not isinstance(event, dict):
            continue
        key = str(event.get("event_id") or stable_id(str(event.get("category")), str(event.get("date") or event.get("start")), str(event.get("label"))))
        if key not in merged:
            merged[key] = event
            continue
        old_conf = float(merged[key].get("confidence") or 0)
        new_conf = float(event.get("confidence") or 0)
        if new_conf >= old_conf:
            merged[key] = {**merged[key], **event}
    return sorted(merged.values(), key=lambda e: str(e.get("date") or e.get("start") or ""))


def main() -> int:
    existing = load_existing()
    discovered = []
    failures = []

    try:
        discovered.extend(discover_nvidia_official())
    except Exception as exc:
        failures.append(f"nvidia:{exc}")
    try:
        discovered.extend(discover_news_events())
    except Exception as exc:
        failures.append(f"news:{exc}")

    scheduled = []
    try:
        scheduled.extend(parse_fomc_schedule())
    except Exception as exc:
        failures.append(f"fomc:{exc}")
    try:
        scheduled.extend(parse_cpi_schedule())
    except Exception as exc:
        failures.append(f"cpi:{exc}")

    payload = {
        "updated_at": now_iso(),
        "events": merge_events(existing.get("events", []), discovered),
        "scheduled_events": merge_events(existing.get("scheduled_events", []), scheduled),
        "last_run": {
            "discovered_count": len(discovered),
            "scheduled_count": len(scheduled),
            "failures": failures,
        },
    }
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["last_run"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
