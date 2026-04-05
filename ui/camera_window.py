import os
import sys
from typing import Callable

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

# Thêm đường dẫn gốc để import từ core
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.camera_app import SmartCamera

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
        self.camera = SmartCamera(camera_index=1) # Mặc định dùng cam ngoài (index 1) thay vì cam tích hợp (index 0)
        self.preview_mode = "live"      # "live" | "captured"
        self._captured_frame: np.ndarray | None = None
        self._ctk_img_ref = None        # GC guard
        self._is_polling = False

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
            btn_inner, text="⬤  CHỤP ẢNH", command=self._start_capture_flow,
            height=48, width=200, corner_radius=12,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#e74c3c", hover_color="#c0392b",
        )
        self.btn_capture.pack(side="left", padx=(20, 10))

        # Dropdown hẹn giờ chụp
        self.timer_var = ctk.StringVar(value="0s (Tức thì)")
        self.timer_menu = ctk.CTkOptionMenu(
            btn_inner, variable=self.timer_var,
            values=["0s (Tức thì)", "Hẹn 3 Giây", "Hẹn 5 Giây"],
            width=140, height=48, corner_radius=12,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#34495e", button_color="#2c3e50", button_hover_color="#1a252f"
        )
        self.timer_menu.pack(side="left", padx=(0, 10))

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

    # ── Camera logic ─────────────────────────────────────────────────────────

    def _start_camera(self):
        success = self.camera.start()
        
        if not success:
            self.status_lbl.configure(text="● KHÔNG TÌM THẤY CAMERA", text_color="#e74c3c")
            self.cam_label.configure(text="Không thể mở camera.\nKiểm tra kết nối webcam.")
            return

        self._is_polling = True
        self._poll_frame()

    def _poll_frame(self):
        """Main thread: poll buffer và render (~30 fps)."""
        if not self._is_polling:
            return
            
        if self.preview_mode == "live":
            frame = self.camera.get_latest_frame()
            if frame is not None:
                # Xử lý NHẸ cho preview live (chỉ normalize)
                preview = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                self._render_frame(preview)
                
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

    def _start_capture_flow(self):
        val = self.timer_var.get()
        if "3" in val:
            sec = 3
        elif "5" in val:
            sec = 5
        else:
            sec = 0
            
        if sec > 0:
            self._countdown(sec)
        else:
            self._capture()

    def _countdown(self, sec):
        if not self.winfo_exists():
            return
            
        if sec > 0:
            self.btn_capture.configure(state="disabled")
            self.timer_menu.configure(state="disabled")
            self.btn_retake.configure(state="disabled")
            self.btn_use.configure(state="disabled")
            
            # Hiển thị bộ đếm ngược trên label
            self.status_lbl.configure(text=f"● CHỤP TỰ ĐỘNG SAU {sec}s...", text_color="#e74c3c")
            
            self.after(1000, self._countdown, sec - 1)
        else:
            self.timer_menu.configure(state="normal")
            self._capture()

    def _capture(self):
        frame = self.camera.get_latest_frame()
        if frame is None:
            return
        self._captured_frame = frame.copy()
        self.preview_mode    = "captured"
        
        # Áp dụng bộ lọc Enhance lên để người dùng xem luôn ảnh sau xử lý
        enhanced = SmartCamera.enhance_for_grading(self._captured_frame)
        self._render_frame(enhanced)
        
        self.status_lbl.configure(text="● ẢNH ĐÃ CHỤP", text_color="#f39c12")
        self.btn_capture.configure(state="disabled")
        self.timer_menu.configure(state="disabled")
        self.btn_retake.configure(state="normal")
        self.btn_use.configure(state="normal")

    def _retake(self):
        self._captured_frame = None
        self.preview_mode    = "live"
        self.status_lbl.configure(text="● CAMERA ĐANG HOẠT ĐỘNG", text_color="#2ecc71")
        self.btn_capture.configure(state="normal")
        self.timer_menu.configure(state="normal")
        self.btn_retake.configure(state="disabled")
        self.btn_use.configure(state="disabled")

    def _use_photo(self):
        if self._captured_frame is not None:
            # Truyền lại frame đã ĐƯỢC XỬ LÝ NẶNG về cho GUI chính chấm điểm
            enhanced = SmartCamera.enhance_for_grading(self._captured_frame)
            self._stop()
            self.on_capture_callback(enhanced)
            self.destroy()

    def _stop(self):
        self._is_polling = False
        self.camera.stop()

    def _on_close(self):
        self._stop()
        self.destroy()
