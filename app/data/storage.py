"""
app/data/storage.py
───────────────────
Toàn bộ I/O file JSON: đáp án TN, đáp án TL, và điểm học sinh.
Muốn đổi sang SQLite hay database khác → chỉ sửa file này.
"""

import json
import os

_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR      = os.path.join(_PROJECT_ROOT, "data")
KEYS_DB_PATH   = os.path.join(_DATA_DIR, "keys_db.json")
TL_KEYS_PATH   = os.path.join(_DATA_DIR, "tl_keys_db.json")
SCORES_DB_PATH = os.path.join(_DATA_DIR, "student_scores.json")


# ── Đáp án Trắc nghiệm ───────────────────────────────────────────────────────

def load_answer_keys() -> dict:
    """Trả về {ma_de: {q_num(int): ans(str)}}"""
    if not os.path.exists(KEYS_DB_PATH):
        return {}
    try:
        with open(KEYS_DB_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {made: {int(q): ans for q, ans in keys.items()}
                for made, keys in raw.items()}
    except Exception as e:
        print(f"[storage] Không thể đọc keys_db: {e}")
        return {}


def save_answer_keys(db: dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    try:
        with open(KEYS_DB_PATH, "w", encoding="utf-8") as f:
            lines = [
                f'  "{made}": {json.dumps({str(k): v for k, v in keys.items()})}'
                for made, keys in db.items()
            ]
            f.write("{\n" + ",\n".join(lines) + "\n}")
    except Exception as e:
        print(f"[storage] Không thể lưu keys_db: {e}")


# ── Đáp án Tự luận ───────────────────────────────────────────────────────────
# Format: {ma_de: [ {q: int, dap_an: str, diem: float} ]}

def load_tl_keys() -> dict:
    """Trả về {ma_de: [ {q, dap_an, diem} ]}"""
    if not os.path.exists(TL_KEYS_PATH):
        return {}
    try:
        with open(TL_KEYS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[storage] Không thể đọc tl_keys_db: {e}")
        return {}


def save_tl_keys(db: dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    try:
        with open(TL_KEYS_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[storage] Không thể lưu tl_keys_db: {e}")


# ── Điểm học sinh ─────────────────────────────────────────────────────────────

def append_student_score(record: dict) -> None:
    """Thêm 1 bản ghi điểm vào cuối file."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    scores = _load_scores()
    scores.append(record)
    _write_scores(scores)


def _load_scores() -> list:
    if not os.path.exists(SCORES_DB_PATH):
        return []
    try:
        with open(SCORES_DB_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return json.loads(content) if content else []
    except Exception as e:
        print(f"[storage] Không thể đọc scores: {e}")
        return []


def _write_scores(scores: list) -> None:
    try:
        with open(SCORES_DB_PATH, "w", encoding="utf-8") as f:
            lines = [json.dumps(s, ensure_ascii=False) for s in scores]
            f.write("[\n  " + ",\n  ".join(lines) + "\n]")
    except Exception as e:
        print(f"[storage] Không thể lưu scores: {e}")