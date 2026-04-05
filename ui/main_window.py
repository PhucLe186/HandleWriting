"""
app/ui/main_window.py (Giao diện mới từ ui_test, kết hợp core cũ)
"""
import os
import sys
import threading
import tempfile
import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image
from tkinter import messagebox

# Thêm đường dẫn gốc
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- BACKEND CŨ (Original from main_window.py) ---
from core.yolo_cropper import YoloCropper
from core.trocr_reader import TrOCRReader
from core.llm_grader import LLMGrader
from core.excel_exporter import ExcelExporter
from core.omr_processor import OMRProcessor
from ui.key_editor import KeyEditorWindow
from ui.camera_window import CameraWindow
from ui.auto_mcq_window import AutoMCQWindow

LLM_API_KEY = "" 
YOLO_WEIGHTS = os.path.join(current_dir, "models", "yolo_weights", "best.pt")

MAX_MCQ_SCORE = 4.0
MAX_ESSAY_SCORE = 6.0

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SystemCamera(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hệ Thống Chấm Điểm Chuyên Nghiệp")
        self.geometry("1200x800")
        self.minsize(900, 600)
        self.after(0, lambda: self.state("zoomed"))

        # --- State ---
        self.current_mssv = "?"
        self.current_mcq_score = 0.0
        self.current_essay_score = 0.0
        self.current_essay_details = []
        
        self.answer_key_db = {}
        self.tl_key_db = {}

        self.tn_annotated = None
        self.tl_annotated = None
        
        self._display_source = None
        self._temp_path = None

        self._build_ui()
        
        # Khởi động AI trong luồng phụ
        threading.Thread(target=self.init_ai_models, daemon=True).start()

    def init_ai_models(self):
        self._set_info("⏳ Đang khởi động bộ máy AI... (Khoảng 5-10 giây)")
        try:
            self.detector = YoloCropper(model_path=YOLO_WEIGHTS)
            self.recognizer = TrOCRReader()
            self.grader = LLMGrader(api_key=LLM_API_KEY)
            self.omr = OMRProcessor()
            self.exporter = ExcelExporter(output_dir=os.path.join(current_dir, "data", "output"))
            
            self._set_info("✅ AI ĐÃ SẴN SÀNG! Đưa phiếu trắc nghiệm vào trước.", "#2ecc71")
            self.btn_cam_tn.configure(state="normal")
            self.btn_auto_tn.configure(state="normal")
            self.btn_cam_tl.configure(state="normal")
        except Exception as e:
            self._set_info(f"❌ Lỗi tải AI: {e}", "#e74c3c")
            messagebox.showerror("Error", f"Lỗi khởi động AI:\n{e}")

    # ── UI BUILDING ──────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left panel
        left = ctk.CTkFrame(self, width=340, corner_radius=15)
        left.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        left.grid_propagate(False)
        self._left = left

        ctk.CTkLabel(
            left, text="HỆ THỐNG CHẤM OMR",
            font=ctk.CTkFont(family="Roboto", size=20, weight="bold"),
        ).pack(pady=(20, 4))

        self._build_step1(left)
        self._build_step2(left)
        self._build_tools(left)
        self._build_results(left)

        # Right panel — chia thành 2 hàng: ảnh (trên) + kết quả OCR (dưới)
        right = ctk.CTkFrame(self, corner_radius=15, fg_color="gray15")
        right.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        right.grid_rowconfigure(0, weight=3)   # ảnh chiếm 75%
        right.grid_rowconfigure(1, weight=1)   # bảng kết quả chiếm 25%
        right.grid_columnconfigure(0, weight=1)
        self._right = right

        # -- Vùng hiển thị ảnh --
        self._img_frame = ctk.CTkFrame(right, fg_color="gray15", corner_radius=0)
        self._img_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.image_label = ctk.CTkLabel(
            self._img_frame, text="📷  Chờ khởi động hệ thống...",
            font=ctk.CTkFont(size=16), text_color="gray50",
        )
        self.image_label.pack(expand=True, fill="both", padx=10, pady=10)
        self._img_frame.bind("<Configure>", self._on_resize)

        # -- Vùng kết quả OCR tự luận (ẩn ban đầu) --
        self._ocr_panel = ctk.CTkTabview(right, fg_color="gray20", corner_radius=10)
        self._ocr_panel.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._ocr_panel.grid_remove()  # Ẩn cho đến khi có kết quả

        self._tab_details = self._ocr_panel.add("Chi tiết chấm")
        self._tab_text = self._ocr_panel.add("Chữ đã quét")

        # Scrollable frame chứa các hàng kết quả
        self._ocr_scroll = ctk.CTkScrollableFrame(
            self._tab_details, fg_color="gray20", corner_radius=0,
            label_text="",
        )
        self._ocr_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._ocr_scroll.grid_columnconfigure(0, minsize=55)
        self._ocr_scroll.grid_columnconfigure(1, weight=2)
        self._ocr_scroll.grid_columnconfigure(2, weight=2)
        self._ocr_scroll.grid_columnconfigure(3, minsize=80)

        # Textbox cho Chữ đã quét
        self._ocr_textbox = ctk.CTkTextbox(
            self._tab_text, fg_color="gray15", corner_radius=5, font=ctk.CTkFont(size=14, family="Consolas")
        )
        self._ocr_textbox.pack(fill="both", expand=True, padx=8, pady=8)
        self._ocr_textbox.configure(state="disabled")

    def _build_step1(self, parent):
        box = ctk.CTkFrame(parent, fg_color="gray20", corner_radius=12)
        box.pack(fill="x", padx=14, pady=(8, 4))
        ctk.CTkLabel(
            box, text="BƯỚC 1 — TRẮC NGHIỆM",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#3498DB",
        ).pack(anchor="w", padx=14, pady=(10, 4))

        self.btn_cam_tn = ctk.CTkButton(
            box, text="📷  Chụp bài trắc nghiệm",
            command=self._open_camera_tn,
            height=42, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#8e44ad", hover_color="#6c3483", corner_radius=10,
            state="disabled",
        )
        self.btn_cam_tn.pack(pady=(0, 6), padx=12, fill="x")

        self.btn_cham_tn = ctk.CTkButton(
            box, text="⚡  Chấm trắc nghiệm",
            command=self._process_tn,
            height=42, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1ABC9C", hover_color="#16A085", corner_radius=10,
            state="disabled",
        )
        self.btn_cham_tn.pack(pady=(0, 6), padx=12, fill="x")

        self.btn_auto_tn = ctk.CTkButton(
            box, text="🤖  Chấm Tự Động (Chỉ TN)",
            command=self._open_auto_mcq_window,
            height=42, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2980B9", hover_color="#1A5276", corner_radius=10,
            state="disabled",
        )
        self.btn_auto_tn.pack(pady=(0, 10), padx=12, fill="x")

    def _build_step2(self, parent):
        box = ctk.CTkFrame(parent, fg_color="gray20", corner_radius=12)
        box.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(
            box, text="BƯỚC 2 — TỰ LUẬN",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#e67e22",
        ).pack(anchor="w", padx=14, pady=(10, 4))

        self.btn_cam_tl = ctk.CTkButton(
            box, text="📷  Chụp bài tự luận",
            command=self._open_camera_tl,
            height=42, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#8e44ad", hover_color="#6c3483", corner_radius=10,
            state="disabled",
        )
        self.btn_cam_tl.pack(pady=(0, 6), padx=12, fill="x")

        self.btn_cham_tl = ctk.CTkButton(
            box, text="📝  Chấm tự luận",
            command=self._process_tl,
            height=42, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#e67e22", hover_color="#ca6f1e", corner_radius=10,
            state="disabled",
        )
        self.btn_cham_tl.pack(pady=(0, 10), padx=12, fill="x")

    def _build_tools(self, parent):
        ctk.CTkButton(
            parent, text="⚙  Quản lý đáp án",
            command=self._open_key_editor,
            height=38, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#34495E", hover_color="#2C3E50", corner_radius=10,
        ).pack(pady=(6, 4), padx=14, fill="x")

        ctk.CTkFrame(parent, height=2, fg_color="gray30").pack(fill="x", padx=14, pady=8)

    def _build_results(self, parent):
        # SBD
        info_row = ctk.CTkFrame(parent, fg_color="transparent")
        info_row.pack(fill="x", padx=14)
        self.lbl_sbd = ctk.CTkLabel(info_row, text="SBD: ---", font=ctk.CTkFont(family="Consolas", size=15))
        self.lbl_sbd.pack(side="left")

        # Score box
        score_box = ctk.CTkFrame(parent, fg_color="gray20", corner_radius=12)
        score_box.pack(fill="x", padx=14, pady=8)

        def _score_row(label_text, color, attr):
            row = ctk.CTkFrame(score_box, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=(8, 2))
            ctk.CTkLabel(row, text=label_text, font=ctk.CTkFont(size=13), text_color="gray70").pack(side="left")
            lbl = ctk.CTkLabel(row, text="---", font=ctk.CTkFont(size=15, weight="bold"), text_color=color)
            lbl.pack(side="right")
            setattr(self, attr, lbl)

        _score_row("Trắc nghiệm:", "#3498DB", "lbl_score_tn")
        _score_row("Tự luận:",     "#e67e22", "lbl_score_tl")

        ctk.CTkFrame(score_box, height=1, fg_color="gray35").pack(fill="x", padx=14, pady=6)

        total_row = ctk.CTkFrame(score_box, fg_color="transparent")
        total_row.pack(fill="x", padx=14, pady=(2, 12))
        ctk.CTkLabel(total_row, text="TỔNG ĐIỂM:", font=ctk.CTkFont(size=14, weight="bold"), text_color="gray80").pack(side="left")
        self.lbl_score_total = ctk.CTkLabel(
            total_row, text="---", font=ctk.CTkFont(size=28, weight="bold"), text_color="#2ecc71"
        )
        self.lbl_score_total.pack(side="right")

        # Lưu + info
        self.btn_save = ctk.CTkButton(
            parent, text="💾  LƯU KẾT QUẢ",
            command=self._save,
            height=44, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2980B9", hover_color="#1F618D",
            corner_radius=12, state="disabled",
        )
        self.btn_save.pack(pady=(4, 6), padx=14, fill="x")

        self.lbl_info = ctk.CTkLabel(
            parent, text="",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="#95A5A6", justify="left", wraplength=300,
        )
        self.lbl_info.pack(side="bottom", fill="x", pady=10, padx=10)

    # ── CAMERA ───────────────────────────────────────────────────────────────

    def _open_camera_tn(self):
        CameraWindow(self, on_capture_callback=self._on_tn_captured)

    def _open_auto_mcq_window(self):
        AutoMCQWindow(self, main_app=self)

    def _open_camera_tl(self):
        CameraWindow(self, on_capture_callback=self._on_tl_captured)

    def _on_tn_captured(self, frame: np.ndarray):
        self.tn_image_array = frame
        self._reset_display()
        self._show(frame)
        self.btn_cham_tn.configure(state="normal")
        self.btn_cham_tl.configure(state="disabled")
        self._set_info("Đã chụp bài TN. Nhấn 'Chấm trắc nghiệm' để tiếp tục.")

    def _on_tl_captured(self, frame: np.ndarray):
        self.tl_image_array = frame
        self._show(frame)
        self.btn_cham_tl.configure(state="normal")
        self._set_info("Đã chụp bài TL. Nhấn 'Chấm tự luận' để hoàn tất.")

    # ── PROCESSING ───────────────────────────────────────────────────────────

    def _process_tn(self):
        if self.tn_image_array is None:
            messagebox.showwarning("Thiếu ảnh", "Vui lòng chụp bài trắc nghiệm trước!")
            return
            
        self.lbl_score_tn.configure(text="Đang xử lý...", text_color="gray")
        self.btn_cham_tn.configure(state="disabled")
        threading.Thread(target=self._run_omr_bg, daemon=True).start()

    def _run_omr_bg(self):
        self._set_info("⏳ Đang quét trắc nghiệm...")
        try:
            tmp_path = self._to_tempfile(self.tn_image_array)
            result = self.omr.process_image(tmp_path, answer_key_db=self.answer_key_db)
            self._cleanup_temp()
            
            sbd_str = result.get('sbd', '?')
            self.current_mssv = sbd_str if sbd_str != "" else "?"
            
            raw_score = result.get('score', 0.0)
            self.current_mcq_score = (raw_score / 10.0) * MAX_MCQ_SCORE
            self.tn_annotated = result.get('annotated_image', self.tn_image_array)

            self.after(0, self._on_tn_done)
        except Exception as e:
            self.after(0, lambda e=e: self._on_tn_error(e))

    def _on_tn_done(self):
        self._refresh_score_ui()
        self._show(self.tn_annotated)
        self._set_info(f"✔ Đã chấm Trắc nghiệm. Sinh viên: {self.current_mssv}\nChụp tự luận hoặc lưu ngay.", "#2ecc71")
        self.btn_cham_tn.configure(state="normal")
        self.btn_cam_tl.configure(state="normal")
        self.btn_save.configure(state="normal")

    def _on_tn_error(self, err):
        messagebox.showerror("Lỗi OMR", f"Không chấm được trắc nghiệm:\n{err}")
        self.lbl_score_tn.configure(text="LỖI", text_color="#e74c3c")
        self.btn_cham_tn.configure(state="normal")
        self._set_info("❌ Lỗi Trắc nghiệm. Vui lòng chụp lại.", "#e74c3c")

    def _process_tl(self):
        if self.tl_image_array is None:
            messagebox.showwarning("Thiếu ảnh", "Vui lòng chụp bài tự luận trước!")
            return
            
        self.lbl_score_tl.configure(text="Đang xử lý...", text_color="gray")
        self.btn_cham_tl.configure(state="disabled")
        threading.Thread(target=self._run_essay_bg, daemon=True).start()

    def _run_essay_bg(self):
        self._set_info("⏳ Đang nhận diện chữ và chấm Tự luận bằng AI...")
        try:
            # Lấy đáp án của mã đề tương ứng (nếu có), mặc định lấy TL 1
            essay_key = {}
            question_key = {}
            # Nếu answer_key_db chưa tích hợp đầy đủ TL, dùng cách cũ từ gui_app:
            # "Lấy dict đáp án tl_key_db" -> Hiện tại chúng ta chưa implement load_tl vào format phù hợp cho AI_grader.
            # Tạm thời cứ giả sử tl_key_db chứa list dicts, chuyển đổi nó cho Llama:
            code = self.current_mssv # Hoặc mã đề nếu bạn có trường mã đề
            if len(self.tl_key_db) > 0:
                # Lấy ngẫu nhiên key đầu tiên nếu ko có mã đề
                k = next(iter(self.tl_key_db))
                tl_list = self.tl_key_db[k]
                for item in tl_list:
                    essay_key[item['q'] - 1] = item['dap_an'] # -1 vì YOLO trả order từ 0
                    question_key[item['q'] - 1] = item.get('cau_hoi', '')
                    
            crops = self.detector.detect_and_crop(self.tl_image_array, save_debug=False)
            self.current_essay_details = []
            correct_count = 0
            annotated_essay_img = self.tl_image_array.copy()
            
            if not crops:
                raise ValueError("Không tìm thấy khung trả lời tự luận nào trên ảnh!")
                
            for item in crops:
                idx = item["order"]
                roi = item["image"]
                box = item.get("box", (0,0,0,0))
                
                ref_text = essay_key.get(idx, "")
                question_text = question_key.get(idx, "")
                ocr_text = self.recognizer.extract_text(roi)
                grade_res = self.grader.grade(ref_text, ocr_text, question=question_text)
                
                is_correct = grade_res.get("result", "SAI")
                if is_correct == "ĐÚNG":
                    correct_count += 1
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                    
                if box != (0,0,0,0):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated_essay_img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(annotated_essay_img, f"Cau {idx+1}: {is_correct}", (x1, max(30, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                                
                self.current_essay_details.append({
                    "Câu": idx + 1,
                    "Câu hỏi": question_text,
                    "Đáp án chuẩn": ref_text,
                    "Học sinh viết": ocr_text,
                    "Kết quả": is_correct,
                    "Lý do": grade_res.get("reason", "")
                })
                
            total_essay_questions = len(essay_key) if len(essay_key) > 0 else 1
            self.current_essay_score = (correct_count / total_essay_questions) * MAX_ESSAY_SCORE
            self.tl_annotated = annotated_essay_img
            
            self.after(0, self._on_tl_done)
        except Exception as e:
            self.after(0, lambda e=e: self._on_tl_error(e))

    def _on_tl_done(self):
        self._refresh_score_ui()
        self._show(self.tl_annotated)
        self._render_ocr_table(self.current_essay_details)
        self._set_info("✔ Đã chấm xong TN + TL. Nhấn LƯU để lưu kết quả.", "#2ecc71")
        self.btn_cham_tl.configure(state="normal")
        self.btn_save.configure(state="normal")

    def _on_tl_error(self, err):
        messagebox.showerror("Lỗi Tự Luận", f"Không chấm được tự luận:\n{err}")
        self.lbl_score_tl.configure(text="LỖI", text_color="#e74c3c")
        self.btn_cham_tl.configure(state="normal")
        self._set_info("❌ Lỗi Tự luận. Vui lòng chụp lại.", "#e74c3c")

    # ── SAVE & MISC ──────────────────────────────────────────────────────────

    def _save(self):
        total = self.current_mcq_score + self.current_essay_score
        try:
            self.exporter.export(
                mssv=self.current_mssv,
                mcq_score=self.current_mcq_score,
                essay_score=self.current_essay_score,
                total_score=total,
                essay_details=self.current_essay_details
            )
            messagebox.showinfo("Đã Lưu", f"Đã lưu thành công SBD {self.current_mssv} — Tổng: {total:.2f}")
            self._set_info("💾 Đã lưu bảng điểm. SẴN SÀNG CHẤM BÀI MỚI!", "#3498DB")
            self._reset_display()
        except Exception as e:
            messagebox.showerror("Lỗi Export", f"Không thể lưu kết quả Excel:\n{e}")

    def _open_key_editor(self):
        def on_save(ma_de: str, tn_dict: dict, tl_list: list):
            self.answer_key_db[ma_de] = tn_dict
            self.tl_key_db[ma_de] = tl_list

        def on_bulk_save(db: dict):
            self.answer_key_db = db

        KeyEditorWindow(
            self,
            answer_key_db=self.answer_key_db,
            tl_key_db=self.tl_key_db,
            on_save=on_save,
            on_bulk_save=on_bulk_save,
        )

    # ── UI HELPERS ───────────────────────────────────────────────────────────

    def _refresh_score_ui(self):
        self.lbl_sbd.configure(text=f"SBD: {self.current_mssv}")
        self.lbl_score_tn.configure(text=f"{self.current_mcq_score:.2f} / {MAX_MCQ_SCORE}", text_color="#3498DB")
        self.lbl_score_tl.configure(text=f"{self.current_essay_score:.2f} / {MAX_ESSAY_SCORE}", text_color="#e67e22")
        self.lbl_score_total.configure(text=f"{(self.current_mcq_score + self.current_essay_score):.2f}", text_color="#2ecc71")

    def _reset_display(self):
        self.current_mssv = "?"
        self.current_mcq_score = 0.0
        self.current_essay_score = 0.0
        self.current_essay_details = []
        
        self.lbl_sbd.configure(text="SBD: ---")
        self.lbl_score_tn.configure(text="---")
        self.lbl_score_tl.configure(text="---")
        self.lbl_score_total.configure(text="---", text_color="#2ecc71")
        self.btn_save.configure(state="disabled")
        # Ẩn panel OCR khi reset
        self._ocr_panel.grid_remove()

    def _set_info(self, text: str, color: str = "#95A5A6"):
        self.lbl_info.configure(text=text, text_color=color)

    def _show(self, source):
        self._display_source = source
        self._render_image(source)

    def _render_ocr_table(self, details: list):
        """Xóa và vẽ lại bảng kết quả OCR trong _ocr_scroll."""
        # Xóa nội dung cũ
        for w in self._ocr_scroll.winfo_children():
            w.destroy()

        self._ocr_textbox.configure(state="normal")
        self._ocr_textbox.delete("1.0", "end")

        if not details:
            ctk.CTkLabel(
                self._ocr_scroll, text="Không có dữ liệu OCR.",
                text_color="gray60", font=ctk.CTkFont(size=11)
            ).grid(row=0, column=0, columnspan=4, pady=10)
            self._ocr_textbox.configure(state="disabled")
            self._ocr_panel.grid()  # Vẫn hiện panel
            return

        # Header hàng
        headers = ["Câu", "OCR đọc được", "Đáp án chuẩn", "Kết quả"]
        head_colors = ["gray60", "gray60", "gray60", "gray60"]
        for c, (h, hc) in enumerate(zip(headers, head_colors)):
            ctk.CTkLabel(
                self._ocr_scroll, text=h,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=hc,
            ).grid(row=0, column=c, padx=(6, 4), pady=(2, 4), sticky="w")

        # Separator
        sep = ctk.CTkFrame(self._ocr_scroll, height=1, fg_color="gray35")
        sep.grid(row=1, column=0, columnspan=4, sticky="ew", padx=4, pady=2)

        # Các hàng dữ liệu
        for r, item in enumerate(details, start=2):
            is_correct = item.get("Kết quả", "SAI")
            row_color = "#1a3a1a" if is_correct == "ĐÚNG" else "#3a1a1a"
            result_color = "#2ecc71" if is_correct == "ĐÚNG" else "#e74c3c"
            result_icon = "✅ ĐÚNG" if is_correct == "ĐÚNG" else "❌ SAI"

            # Nền hàng
            row_bg = ctk.CTkFrame(self._ocr_scroll, fg_color=row_color, corner_radius=6)
            row_bg.grid(row=r, column=0, columnspan=4, sticky="ew", padx=2, pady=1)
            row_bg.grid_columnconfigure(0, minsize=55)
            row_bg.grid_columnconfigure(1, weight=2)
            row_bg.grid_columnconfigure(2, weight=2)
            row_bg.grid_columnconfigure(3, minsize=80)

            # Cột câu số
            ctk.CTkLabel(
                row_bg,
                text=f"Câu {item.get('Câu', r-1)}",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="gray80",
            ).grid(row=0, column=0, padx=(8, 4), pady=4, sticky="w")

            # Cột OCR đọc được
            ocr_txt = item.get("Học sinh viết", "") or "(trống)"
            ctk.CTkLabel(
                row_bg,
                text=ocr_txt,
                font=ctk.CTkFont(family="Consolas", size=11),
                text_color="#f0e68c",
                wraplength=160, justify="left",
                anchor="w",
            ).grid(row=0, column=1, padx=4, pady=4, sticky="ew")

            # Cột đáp án chuẩn
            ref_txt = item.get("Đáp án chuẩn", "") or "(chưa có)"
            ctk.CTkLabel(
                row_bg,
                text=ref_txt,
                font=ctk.CTkFont(family="Consolas", size=11),
                text_color="#87ceeb",
                wraplength=160, justify="left",
                anchor="w",
            ).grid(row=0, column=2, padx=4, pady=4, sticky="ew")

            # Cột kết quả
            reason = item.get("Lý do", "")
            ctk.CTkLabel(
                row_bg,
                text=result_icon,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=result_color,
            ).grid(row=0, column=3, padx=(4, 8), pady=4, sticky="w")

            # Nếu có lý do, thêm dòng phụ
            if reason:
                ctk.CTkLabel(
                    row_bg,
                    text=f"  ↳ {reason}",
                    font=ctk.CTkFont(size=10, slant="italic"),
                    text_color="gray55",
                    anchor="w",
                ).grid(row=1, column=1, columnspan=3, padx=4, pady=(0, 4), sticky="w")

        # Nạp dữ liệu văn bản vào textbox "Chữ đã quét"
        full_text = ""
        for item in details:
            q_num = item.get("Câu", "?")
            ocr_txt = item.get("Học sinh viết", "")
            if ocr_txt.strip() != "":
                full_text += f"Câu {q_num}:\n{ocr_txt}\n\n"
        
        if full_text.strip() == "":
            full_text = "(Không đọc được chữ nào)"
            
        self._ocr_textbox.insert("1.0", full_text.strip())
        self._ocr_textbox.configure(state="disabled")

        # Hiện panel
        self._ocr_panel.grid()

    def _on_resize(self, _event):
        if self._display_source is not None:
            self._render_image(self._display_source)

    def _render_image(self, source):
        if source is None: return
        self.update_idletasks()
        if isinstance(source, str):
            img = Image.open(source)
        else:
            img = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))

        fw = self._img_frame.winfo_width()
        fh = self._img_frame.winfo_height()
        if fw < 10 or fh < 10: return

        ratio = min((fw - 40) / img.width, (fh - 40) / img.height)
        nw = int(img.width * ratio)
        nh = int(img.height * ratio)
        if nw <= 0 or nh <= 0: return

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(nw, nh))
        self.image_label.configure(image=ctk_img, text="")

    # ── TEMP FILE ────────────────────────────────────────────────────────────

    def _to_tempfile(self, arr: np.ndarray) -> str:
        # Save temp image for OMR since it takes a string path
        tmp_path = os.path.join(current_dir, "data", "temp_crops", "temp_mcq.jpg")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        cv2.imwrite(tmp_path, arr)
        self._temp_path = tmp_path
        return tmp_path

    def _cleanup_temp(self):
        # We can leave temp file, no strict need to delete right now, but we can implement it if needed
        pass
