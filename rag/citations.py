"""Normalize citation typography without inventing IDs or attaching new evidence."""

import re


_GROUP = re.compile(r"\[\s*[sS]\d+(?:\s*[,;]\s*[sS]\d+)*\s*\]|【\s*[sS]\d+(?:\s*[,;]\s*[sS]\d+)*\s*】")


def normalize_citations(answer):
    return _GROUP.sub(
        lambda match: " ".join(f"[{sid}]" for sid in dict.fromkeys(
            sid.upper() for sid in re.findall(r"[sS]\d+", match.group())
        )), answer,
    )
