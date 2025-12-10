import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional


class FeedbackRepository:
    """Persists user feedback insights to disk for long-term learning."""

    def __init__(self, store_path: Optional[str] = None):
        default_path = os.getenv("FEEDBACK_STORE_PATH", "knowledge_db/feedback_history.jsonl")
        self._path = Path(store_path or default_path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def save_feedback(self, feedback_items: List[Dict]) -> None:
        """Append feedback items to the JSONL store."""
        if not feedback_items:
            return

        serialized_lines = [json.dumps(item, ensure_ascii=False) + "\n" for item in feedback_items]

        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.writelines(serialized_lines)

    def load_recent(self, limit: int = 100) -> List[Dict]:
        """Return the most recent feedback items (best-effort)."""
        if not self._path.exists() or limit <= 0:
            return []

        try:
            with self._path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return []

        recent = lines[-limit:]
        parsed: List[Dict] = []
        for line in recent:
            line = line.strip()
            if not line:
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return parsed
