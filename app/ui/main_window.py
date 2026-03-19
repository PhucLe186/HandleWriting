"""
app/ui/main_window.py
─────────────────────
Cửa sổ chính của ứng dụng.
Chỉ làm 3 việc:
  1. Dựng UI
  2. Điều phối flow (TN → TL → Lưu)
  3. Gọi sang core/ và data/ — không tự xử lý logic hay I/O
"""

import os
import tempfile
from tkinter import messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from app.core.omr_processor import OMRProcessor
from app.core.scorer        import ScoreResult, build_score_result
from app.data.storage       import load_answer_keys, save_answer_keys, load_tl_keys, save_tl_keys, append_student_score
from app.ui.camera_window   import CameraWindow
from app.ui.key_editor      import KeyEditorWindow

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class OMRApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hệ Thống Chấm Điểm Chuyên Nghiệp")
        self.geometry("1200x750")
        self.minsize(900, 600)
        self.after(0, lambda: self.state("zoomed"))

        # ── Core ─────────────────────────────────────────────────────────────
        self.processor    = OMRProcessor()
        self.answer_key_db: dict = load_answer_keys()
        self.tl_key_db: dict     = load_tl_keys()

        # ── State: ảnh và kết quả ────────────────────────────────────────────
        self.tn_image_array: np.ndarray | None = None
        self.tn_raw_result:  dict | None       = None   # output thô từ processor
        self.tn_annotated:   np.ndarray | None = None

        self.tl_image_array: np.ndarray | None = None
        self.tl_raw_result:  dict | None       = None
        self.tl_annotated:   np.ndarray | None = None

        self.score_result: ScoreResult | None  = None   # kết quả tổng hợp
        self._display_source = None                     # ảnh đang hiển thị bên phải
        self._temp_path: str | None = None

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

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

        # Right panel
        right = ctk.CTkFrame(self, corner_radius=15, fg_color="gray15")
        right.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        self._right = right

        self.image_label = ctk.CTkLabel(
            right, text="📷  Nhấn 'Chụp bài trắc nghiệm' để bắt đầu.",
            font=ctk.CTkFont(size=16), text_color="gray50",
        )
        self.image_label.pack(expand=True, fill="both", padx=10, pady=10)
        right.bind("<Configure>", self._on_resize)

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
        )
        self.btn_cam_tn.pack(pady=(0, 6), padx=12, fill="x")

        self.btn_cham_tn = ctk.CTkButton(
            box, text="⚡  Chấm trắc nghiệm",
            command=self._process_tn,
            height=42, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1ABC9C", hover_color="#16A085", corner_radius=10,
            state="disabled",
        )
        self.btn_cham_tn.pack(pady=(0, 10), padx=12, fill="x")

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
        # SBD + Mã đề
        info_row = ctk.CTkFrame(parent, fg_color="transparent")
        info_row.pack(fill="x", padx=14)
        self.lbl_sbd  = ctk.CTkLabel(info_row, text="SBD: ---",
                                     font=ctk.CTkFont(family="Consolas", size=15))
        self.lbl_sbd.pack(side="left")
        self.lbl_made = ctk.CTkLabel(info_row, text="  MÃ ĐỀ: ---",
                                     font=ctk.CTkFont(family="Consolas", size=15))
        self.lbl_made.pack(side="left")

        # Score box
        score_box = ctk.CTkFrame(parent, fg_color="gray20", corner_radius=12)
        score_box.pack(fill="x", padx=14, pady=8)

        def _score_row(label_text, color, attr):
            row = ctk.CTkFrame(score_box, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=(8, 2))
            ctk.CTkLabel(row, text=label_text,
                         font=ctk.CTkFont(size=13), text_color="gray70").pack(side="left")
            lbl = ctk.CTkLabel(row, text="---",
                               font=ctk.CTkFont(size=15, weight="bold"), text_color=color)
            lbl.pack(side="right")
            setattr(self, attr, lbl)

        _score_row("Trắc nghiệm:", "#3498DB", "lbl_score_tn")
        _score_row("Tự luận:",     "#e67e22", "lbl_score_tl")

        ctk.CTkFrame(score_box, height=1, fg_color="gray35").pack(fill="x", padx=14, pady=6)

        total_row = ctk.CTkFrame(score_box, fg_color="transparent")
        total_row.pack(fill="x", padx=14, pady=(2, 12))
        ctk.CTkLabel(total_row, text="TỔNG ĐIỂM:",
                     font=ctk.CTkFont(size=14, weight="bold"), text_color="gray80"
                     ).pack(side="left")
        self.lbl_score_total = ctk.CTkLabel(
            total_row, text="---",
            font=ctk.CTkFont(size=28, weight="bold"), text_color="#2ecc71",
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
        self.lbl_info.pack(side="bottom", pady=10, padx=10)

    # ── Camera callbacks ──────────────────────────────────────────────────────

    def _open_camera_tn(self):
        CameraWindow(self, on_capture_callback=self._on_tn_captured)

    def _open_camera_tl(self):
        CameraWindow(self, on_capture_callback=self._on_tl_captured)

    def _on_tn_captured(self, frame: np.ndarray):
        self.tn_image_array = frame
        self.tn_raw_result  = None
        self.tn_annotated   = None
        self.tl_image_array = None   # reset TL khi chụp lại TN
        self.tl_raw_result  = None
        self.tl_annotated   = None
        self.score_result   = None

        self._reset_display()
        self._show(frame)
        self.btn_cham_tn.configure(state="normal")
        self.btn_cam_tl.configure(state="disabled")
        self.btn_cham_tl.configure(state="disabled")
        self._set_info("Đã chụp bài TN. Nhấn 'Chấm trắc nghiệm' để tiếp tục.")

    def _on_tl_captured(self, frame: np.ndarray):
        self.tl_image_array = frame
        self.tl_raw_result  = None
        self.tl_annotated   = None

        self._show(frame)
        self.btn_cham_tl.configure(state="normal")
        self._set_info("Đã chụp bài TL. Nhấn 'Chấm tự luận' để hoàn tất.")

    # ── Processing ────────────────────────────────────────────────────────────

    def _process_tn(self):
        if self.tn_image_array is None:
            messagebox.showwarning("Thiếu ảnh", "Vui lòng chụp bài trắc nghiệm trước!")
            return
        try:
            self.lbl_score_tn.configure(text="Đang xử lý...", text_color="gray")
            self.btn_cham_tn.configure(state="disabled")
            self.update_idletasks()

            result = self.processor.process_image(
                self._to_tempfile(self.tn_image_array),
                answer_key_db=self.answer_key_db,
            )
            self._cleanup_temp()

            self.tn_raw_result = result
            self.tn_annotated  = result["annotated_image"]
            self.score_result  = build_score_result(tn_result=result)

            self._refresh_score_ui()
            self._show(self.tn_annotated)

            ma_de = self.score_result.ma_de
            if ma_de in self.answer_key_db:
                self._set_info(f"✔ TN xong (mã đề '{ma_de}'). Chụp tự luận hoặc lưu ngay.", "#2ecc71")
            else:
                self._set_info(f"⚠ Mã '{ma_de}' chưa có đáp án!", "#e74c3c")

            self.btn_cham_tn.configure(state="normal")
            self.btn_cam_tl.configure(state="normal")
            self.btn_save.configure(state="normal")

        except Exception as e:
            messagebox.showerror("Lỗi", "Lỗi chấm TN:\n" + str(e))
            self.lbl_score_tn.configure(text="LỖI", text_color="#e74c3c")
            self.btn_cham_tn.configure(state="normal")

    def _process_tl(self):
        if self.tl_image_array is None:
            messagebox.showwarning("Thiếu ảnh", "Vui lòng chụp bài tự luận trước!")
            return
        try:
            self.lbl_score_tl.configure(text="Đang xử lý...", text_color="gray")
            self.btn_cham_tl.configure(state="disabled")
            self.update_idletasks()

            result = self.processor.process_image(
                self._to_tempfile(self.tl_image_array),
                answer_key_db=self.answer_key_db,
            )
            self._cleanup_temp()

            self.tl_raw_result = result
            self.tl_annotated  = result["annotated_image"]
            self.score_result  = build_score_result(
                tn_result=self.tn_raw_result,
                tl_result=result,
            )

            self._refresh_score_ui()
            self._show(self.tl_annotated)
            self._set_info("✔ Đã chấm xong TN + TL. Nhấn LƯU để lưu kết quả.", "#2ecc71")
            self.btn_cham_tl.configure(state="normal")
            self.btn_save.configure(state="normal")

        except Exception as e:
            messagebox.showerror("Lỗi", "Lỗi chấm TL:\n" + str(e))
            self.lbl_score_tl.configure(text="LỖI", text_color="#e74c3c")
            self.btn_cham_tl.configure(state="normal")

    # ── Save ─────────────────────────────────────────────────────────────────

    def _save(self):
        if self.score_result is None:
            return
        append_student_score(self.score_result.to_record())
        messagebox.showinfo(
            "Đã Lưu",
            f"Đã lưu SBD {self.score_result.sbd} "
            f"— Tổng: {self.score_result.total:.2f}/10",
        )
        self.btn_save.configure(state="disabled")

    # ── Key editor ────────────────────────────────────────────────────────────

    def _open_key_editor(self):
        def on_save(ma_de: str, tn_dict: dict, tl_list: list):
            self.answer_key_db[ma_de] = tn_dict
            self.tl_key_db[ma_de]     = tl_list
            save_answer_keys(self.answer_key_db)
            save_tl_keys(self.tl_key_db)

        def on_bulk_save(db: dict):
            self.answer_key_db = db
            save_answer_keys(db)

        KeyEditorWindow(
            self,
            answer_key_db=self.answer_key_db,
            tl_key_db=self.tl_key_db,
            on_save=on_save,
            on_bulk_save=on_bulk_save,
        )

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _refresh_score_ui(self):
        r = self.score_result
        if r is None:
            return
        self.lbl_sbd.configure(text=f"SBD: {r.sbd}")
        self.lbl_made.configure(text=f"  MÃ ĐỀ: {r.ma_de}")
        self.lbl_score_tn.configure(text=f"{r.score_tn:.2f} / 10", text_color="#3498DB")
        self.lbl_score_tl.configure(
            text=f"{r.score_tl:.2f} / 10" if r.has_tl else "---",
            text_color="#e67e22",
        )
        self.lbl_score_total.configure(text=f"{r.total:.2f} / 10", text_color="#2ecc71")

    def _reset_display(self):
        self.lbl_sbd.configure(text="SBD: ---")
        self.lbl_made.configure(text="  MÃ ĐỀ: ---")
        self.lbl_score_tn.configure(text="---")
        self.lbl_score_tl.configure(text="---")
        self.lbl_score_total.configure(text="---", text_color="#2ecc71")
        self.lbl_info.configure(text="")
        self.btn_save.configure(state="disabled")

    def _set_info(self, text: str, color: str = "#95A5A6"):
        self.lbl_info.configure(text=text, text_color=color)

    def _show(self, source):
        """Hiển thị ảnh (numpy BGR array hoặc str path) lên panel phải."""
        self._display_source = source
        self._render_image(source)

    def _on_resize(self, _event):
        if self._display_source is not None:
            self._render_image(self._display_source)

    def _render_image(self, source):
        self.update_idletasks()
        if isinstance(source, str):
            img = Image.open(source)
        else:
            img = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))

        fw = self._right.winfo_width()
        fh = self._right.winfo_height()
        if fw < 10 or fh < 10:
            return

        ratio = min((fw - 40) / img.width, (fh - 40) / img.height)
        nw = int(img.width  * ratio)
        nh = int(img.height * ratio)
        if nw <= 0 or nh <= 0:
            return

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(nw, nh))
        self.image_label.configure(image=ctk_img, text="")

    # ── Temp file helpers ─────────────────────────────────────────────────────

    def _to_tempfile(self, arr: np.ndarray) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self._temp_path = tmp.name
        tmp.close()
        cv2.imwrite(self._temp_path, arr)
        return self._temp_path

    def _cleanup_temp(self):
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except Exception:
                pass
        self._temp_path = None