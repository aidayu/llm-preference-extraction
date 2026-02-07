"""Data saving utilities for dialogue logs.

元ファイル: preference_kg/experiments2/data_saver.py
"""

import json
from pathlib import Path

# Base data directory (app/data/)
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "output_samples"


def get_user_data_dir(user_id: str) -> Path:
    """Get the data directory for a specific user."""
    user_dir = DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def save_dialogue_log(session_data: dict, filename: str = "dialogue_logs.jsonl") -> str:
    """Append a dialogue session to the user's JSONL log file.

    Args:
        session_data: Dict with user_id, timestamp, dialogue_history
        filename: Name of the log file

    Returns:
        Path to the saved file
    """
    user_id = session_data.get("user_id", "anonymous")
    user_dir = get_user_data_dir(user_id)
    filepath = user_dir / filename

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(session_data, ensure_ascii=False) + "\n")

    return str(filepath)


def load_dialogue_logs(user_id: str, filename: str = "dialogue_logs.jsonl") -> list[dict]:
    """Load all dialogue logs for a specific user.

    Args:
        user_id: The user's ID
        filename: Name of the log file

    Returns:
        List of session data dicts
    """
    user_dir = get_user_data_dir(user_id)
    filepath = user_dir / filename

    if not filepath.exists():
        return []

    logs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))

    return logs


def list_users() -> list[str]:
    """List all user IDs with data."""
    if not DATA_DIR.exists():
        return []

    return [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
