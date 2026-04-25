"""
core/essay_processor.py
───────────────────────
Module xử lý chấm điểm tự luận end-to-end.
  1. Nhận ảnh bài tự luận (numpy array)
  2. Dùng YOLO để cắt các khung câu trả lời
  3. Dùng TrOCR để đọc chữ viết tay
  4. Dùng LLM (Groq / Llama) để so sánh với đáp án chuẩn
  5. Trả về điểm số, chi tiết từng câu, và ảnh đã đánh dấu kết quả
"""

import cv2
import numpy as np


class EssayProcessor:
    """
    Bộ xử lý chấm tự luận — tương đương OMRProcessor cho trắc nghiệm.

    Parameters
    ----------
    detector   : YoloCropper  — cắt vùng câu trả lời
    recognizer : TrOCRReader  — đọc chữ viết tay
    grader     : LLMGrader    — chấm điểm bằng AI
    """

    def __init__(self, detector, recognizer, grader):
        self.detector   = detector
        self.recognizer = recognizer
        self.grader     = grader

    def process_image(
        self,
        image: np.ndarray,
        essay_key: dict[int, str],
        question_key: dict[int, str] | None = None,
        max_essay_score: float = 6.0,
    ) -> dict:
        """
        Chấm điểm tự luận từ ảnh.

        Parameters
        ----------
        image           : ảnh bài tự luận (BGR numpy array)
        essay_key       : {idx: đáp_án_chuẩn} — index bắt đầu từ 0
        question_key    : {idx: câu_hỏi_gốc}  — index bắt đầu từ 0 (tuỳ chọn)
        max_essay_score : điểm tối đa phần tự luận

        Returns
        -------
        dict với các key:
            "score"           : float  — điểm tự luận
            "details"         : list[dict] — chi tiết từng câu
            "annotated_image" : np.ndarray — ảnh đã vẽ khung kết quả
        """
        if question_key is None:
            question_key = {}

        crops = self.detector.detect_and_crop(image, save_debug=False)

        if not crops:
            raise ValueError("Không tìm thấy khung trả lời tự luận nào trên ảnh!")

        details = []
        correct_count = 0
        annotated_img = image.copy()

        for item in crops:
            idx = item["order"]
            roi = item["image"]
            box = item.get("box", (0, 0, 0, 0))

            ref_text      = essay_key.get(idx, "")
            question_text = question_key.get(idx, "")
            ocr_text      = self.recognizer.extract_text(roi)
            grade_res     = self.grader.grade(ref_text, ocr_text, question=question_text)

            is_correct = grade_res.get("result", "SAI")
            if is_correct == "ĐÚNG":
                correct_count += 1
                color = (0, 255, 0)   # Xanh lá
            else:
                color = (0, 0, 255)   # Đỏ

            # Vẽ khung và nhãn kết quả lên ảnh
            if box != (0, 0, 0, 0):
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    annotated_img,
                    f"Cau {idx+1}: {is_correct}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3,
                )

            details.append({
                "Câu":          idx + 1,
                "Câu hỏi":     question_text,
                "Đáp án chuẩn": ref_text,
                "Học sinh viết": ocr_text,
                "Kết quả":      is_correct,
                "Lý do":        grade_res.get("reason", ""),
            })

        total_questions = len(essay_key) if len(essay_key) > 0 else 1
        score = (correct_count / total_questions) * max_essay_score

        return {
            "score":           score,
            "details":         details,
            "annotated_image": annotated_img,
        }
