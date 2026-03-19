"""
app/ui/camera_window.py
───────────────────────
Popup camera độc lập: preview live, chụp ảnh, trả numpy BGR frame
về caller qua callback.  Không biết gì về OMR hay điểm số.
"""

import threading
import time
from typing import Callable

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image


class CameraWindow(ctk.CTkToplevel):
    """
    Mở webcam, preview ~30fps, cho phép chụp và xác nhận ảnh.

    Parameters
    ----------
    master              : cửa sổ cha
    on_capture_callback : hàm nhận 1 tham số (bgr_frame: np.ndarray)
                          được gọi khi người dùng nhấn "Sử dụng ảnh này"
    """

    def __init__(self, master, on_capture_callback: Callable[[np.ndarray], None]):
        super().__init__(master)
        self.on_capture_callback = on_capture_callback

        self.title("📷 Chụp Ảnh Bài Thi")
        self.geometry("860x620")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.focus_force()

        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 860) // 2
        y = (self.winfo_screenheight() - 620) // 2
        self.geometry(f"+{x}+{y}")

        # ── State ────────────────────────────────────────────────────────────
        self.cap: cv2.VideoCapture | None = None
        self.running      = False
        self.preview_mode = "live"      # "live" | "captured"
        self._latest_frame: np.ndarray | None = None   # buffer background thread
        self._captured_frame: np.ndarray | None = None
        self._ctk_img_ref = None        # GC guard

        self._build_ui()
        self._start_camera()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="#1a1a2e", height=55, corner_radius=0)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(
            header, text="📷  CHỤP ẢNH BÀI THI",
            font=ctk.CTkFont(size=18, weight="bold"), text_color="#00d4ff",
        ).pack(side="left", padx=20, pady=12)
        self.status_lbl = ctk.CTkLabel(
            header, text="● CAMERA ĐANG HOẠT ĐỘNG",
            font=ctk.CTkFont(size=12), text_color="#2ecc71",
        )
        self.status_lbl.pack(side="right", padx=20)

        # Button bar — pack TRƯỚC preview để không bị đẩy ra ngoài
        btn_bar = ctk.CTkFrame(self, fg_color="gray17", height=75)
        btn_bar.pack(side="bottom", fill="x")
        btn_bar.pack_propagate(False)

        # Preview
        preview = ctk.CTkFrame(self, fg_color="#0d0d1a", corner_radius=0)
        preview.pack(side="top", fill="both", expand=True, padx=20, pady=(10, 5))
        self.cam_label = ctk.CTkLabel(
            preview, text="Đang khởi động camera...",
            font=ctk.CTkFont(size=16), text_color="gray50",
        )
        self.cam_label.pack(expand=True, fill="both", padx=5, pady=5)

        # Buttons (căn giữa dọc trong btn_bar)
        btn_inner = ctk.CTkFrame(btn_bar, fg_color="transparent")
        btn_inner.place(relx=0, rely=0.5, relwidth=1, anchor="w")

        self.btn_capture = ctk.CTkButton(
            btn_inner, text="⬤  CHỤP ẢNH", command=self._capture,
            height=48, width=200, corner_radius=12,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#e74c3c", hover_color="#c0392b",
        )
        self.btn_capture.pack(side="left", padx=(20, 10))

        self.btn_retake = ctk.CTkButton(
            btn_inner, text="↺  CHỤP LẠI", command=self._retake,
            height=48, width=160, corner_radius=12,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#7f8c8d", hover_color="#636e72", state="disabled",
        )
        self.btn_retake.pack(side="left", padx=10)

        self.btn_use = ctk.CTkButton(
            btn_inner, text="✔  SỬ DỤNG ẢNH NÀY", command=self._use_photo,
            height=48, width=220, corner_radius=12,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#27ae60", hover_color="#229954", state="disabled",
        )
        self.btn_use.pack(side="right", padx=(10, 10))

        ctk.CTkButton(
            btn_inner, text="✖  HỦY", command=self._on_close,
            height=48, width=110, corner_radius=12,
            font=ctk.CTkFont(size=14),
            fg_color="#2c3e50", hover_color="#1a252f",
        ).pack(side="right", padx=5)

    # ── Camera loop ──────────────────────────────────────────────────────────

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_lbl.configure(text="● KHÔNG TÌM THẤY CAMERA", text_color="#e74c3c")
            self.cam_label.configure(text="Không thể mở camera.\nKiểm tra kết nối webcam.")
            return
        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        self._poll_frame()

    def _read_loop(self):
        """Background thread: chỉ đọc frame vào buffer, không đụng UI."""
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self._latest_frame = frame
            time.sleep(0.03)

    def _poll_frame(self):
        """Main thread: poll buffer và render (~30 fps)."""
        if not self.running:
            return
        if self.preview_mode == "live" and self._latest_frame is not None:
            self._render_frame(self._latest_frame)
        self.after(33, self._poll_frame)

    def _render_frame(self, frame: np.ndarray):
        """Chuyển BGR → CTkImage và cập nhật label. Luôn gọi trên main thread."""
        try:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            w = self.cam_label.winfo_width()  or 800
            h = self.cam_label.winfo_height() or 480
            if w < 10 or h < 10:
                return
            ratio = min(w / pil_img.width, h / pil_img.height)
            nw, nh = int(pil_img.width * ratio), int(pil_img.height * ratio)
            if nw <= 0 or nh <= 0:
                return
            self._ctk_img_ref = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(nw, nh))
            self.cam_label.configure(image=self._ctk_img_ref, text="")
        except Exception:
            pass

    # ── Actions ──────────────────────────────────────────────────────────────

    def _capture(self):
        frame = self._latest_frame
        if frame is None:
            return
        self._captured_frame = frame.copy()
        self.preview_mode    = "captured"
        self._render_frame(self._captured_frame)
        self.status_lbl.configure(text="● ẢNH ĐÃ CHỤP", text_color="#f39c12")
        self.btn_capture.configure(state="disabled")
        self.btn_retake.configure(state="normal")
        self.btn_use.configure(state="normal")

    def _retake(self):
        self._captured_frame = None
        self.preview_mode    = "live"
        self.status_lbl.configure(text="● CAMERA ĐANG HOẠT ĐỘNG", text_color="#2ecc71")
        self.btn_capture.configure(state="normal")
        self.btn_retake.configure(state="disabled")
        self.btn_use.configure(state="disabled")

    def _use_photo(self):
        if self._captured_frame is not None:
            self._stop()
            self.on_capture_callback(self._captured_frame)
            self.destroy()

    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _on_close(self):
        self._stop()
        self.destroy()