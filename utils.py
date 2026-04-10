"""
utils.py - Shared utilities: logging setup, text helpers, and misc tools.
"""

import logging
import re
import json
from pathlib import Path
from typing import Any

import config


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console and a log file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass  # If log dir doesn't exist, skip file logging

    return logger


def clean_text(text: str) -> str:
    """Strip excessive whitespace and non-printable characters."""
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate a string for display purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"


def pretty_json(obj: Any) -> str:
    """Return a pretty-printed JSON string."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


def sanitize_filename(name: str) -> str:
    """Remove characters that are unsafe in filenames."""
    return re.sub(r"[^\w\-. ]", "_", name).strip()


def file_extension(path: str) -> str:
    """Return lower-case extension including the dot, e.g. '.pdf'."""
    return Path(path).suffix.lower()
