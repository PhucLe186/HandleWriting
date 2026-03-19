"""
app/core/scorer.py
──────────────────
Logic tính điểm thuần túy — không phụ thuộc UI hay file I/O.
Dễ unit-test độc lập.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScoreResult:
    """Kết quả chấm điểm hoàn chỉnh của 1 học sinh."""
    sbd:      str   = "---"
    ma_de:    str   = "---"
    score_tn: float = 0.0   # điểm trắc nghiệm  (thang 10)
    score_tl: float = 0.0   # điểm tự luận      (thang 10)
    has_tl:   bool  = False # True khi đã chấm tự luận

    @property
    def total(self) -> float:
        """Tổng điểm: TN + TL nếu có TL, chỉ TN nếu chưa có."""
        return self.score_tn + self.score_tl if self.has_tl else self.score_tn

    def to_record(self) -> dict:
        """Chuyển thành dict để lưu vào storage."""
        return {
            "sbd":      self.sbd,
            "ma_de":    self.ma_de,
            "score_tn": round(self.score_tn, 2),
            "score_tl": round(self.score_tl, 2),
            "score":    round(self.total, 2),
        }


def build_score_result(
    tn_result: dict,
    tl_result: Optional[dict] = None,
) -> ScoreResult:
    """
    Tạo ScoreResult từ output của omr_processor.

    Parameters
    ----------
    tn_result : dict  — kết quả process_image() cho bài TN
    tl_result : dict  — kết quả process_image() cho bài TL (None nếu chưa có)
    """
    score_tn = tn_result.get("score", 0.0)

    result = ScoreResult(
        sbd      = tn_result.get("sbd", "---"),
        ma_de    = tn_result.get("ma_de", "---"),
        score_tn = score_tn,
    )

    if tl_result is not None:
        result.score_tl = tl_result.get("score_tl", tl_result.get("score", 0.0))
        result.has_tl   = True

    return result