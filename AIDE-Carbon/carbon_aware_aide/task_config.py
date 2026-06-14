"""
Task configuration utilities for configurable AIDE task packs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import yaml


DEFAULT_SUMMARY_SECTIONS = [
    "Challenge 1", "Challenge 2", "Challenge 3",
    "Experimental Evidence", "Key Performance Summary",
    "The Bottom Line",
]


def load_task_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    data["_config_path"] = str(cfg_path)
    data["_config_dir"] = str(cfg_path.parent)
    return data


def resolve_path(base_dir: str | Path, maybe_path: str | None) -> str | None:
    if not maybe_path:
        return None
    p = Path(maybe_path)
    if not p.is_absolute():
        p = Path(base_dir) / p
    return str(p.resolve())


def extract_summary_content(summary_path: str | None, sections_to_keep: list[str] | None = None) -> str:
    if not summary_path:
        return ""
    path = Path(summary_path)
    if not path.exists():
        return ""
    raw_summary = path.read_text(encoding="utf-8")
    keywords = sections_to_keep or DEFAULT_SUMMARY_SECTIONS
    lines = raw_summary.split("\n")
    collecting = False
    kept_lines = []
    for line in lines:
        if any(kw in line for kw in keywords):
            collecting = True
        if collecting:
            kept_lines.append(line)
    summary_content = "\n".join(kept_lines).strip()
    return summary_content or raw_summary


def render_text_template(text: str, variables: dict[str, Any]) -> str:
    rendered = text
    for key, value in variables.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


def load_markdown_prompt(path: str | None, variables: dict[str, Any] | None = None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    content = p.read_text(encoding="utf-8")
    return render_text_template(content, variables or {})
