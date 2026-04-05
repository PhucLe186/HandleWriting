import os
import sys
import threading
import tempfile
import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.camera_app import SmartCamera

class AutoMCQWindow(ctk.CTkToplevel):
    def __init__(self, master, main_app):
        super().__init__(master)
        self.main_app = main_app  # Reference to SystemCamera for omr, exporter, answer_key_db
        
        self.title("⚡ Chấm Tự Động Nhiều Đề Trắc Nghiệm")
        self.geometry("1000x700")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.focus_force()

        # Căn giữa màn hình
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 1000) // 2
        y = (self.winfo_screenheight() - 700) // 2
        self.geometry(f"+{x}+{y}")

        # State
        self.camera = SmartCamera(camera_index=1)
        self._ctk_img_ref = None
        self._is_polling = False
        
        self.total_exams = 1
        self.current_exam = 0
        self.interval = 8
        self.current_countdown = 8
        self.is_running = False

        self._build_ui()
        self._start_camera()
        
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="#2c3e50", height=60, corner_radius=0)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(
            header, text="⚡ CHẤM TỰ ĐỘNG NHIỀU ĐỀ TRẮC NGHIỆM",
            font=ctk.CTkFont(size=18, weight="bold"), text_color="#f1c40f",
        ).pack(side="left", padx=20, pady=15)
        
        self.status_lbl = ctk.CTkLabel(
            header, text="● SẴN SÀNG",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#2ecc71",
        )
        self.status_lbl.pack(side="right", padx=20)

        # Body (Split: Left Config/Logs, Right Camera)
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=15, pady=15)

        # Left Panel
        left_panel = ctk.CTkFrame(body, width=320, fg_color="gray17", corner_radius=10)
        left_panel.pack(side="left", fill="y", padx=(0, 15))
        left_panel.pack_propagate(False)

        # Cấu hình
        config_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        config_frame.pack(fill="x", padx=15, pady=15)
        
        ctk.CTkLabel(config_frame, text="CẤU HÌNH", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(0, 10))
        
        # Số lượng đề
        ctk.CTkLabel(config_frame, text="Số lượng đề (1-100):").pack(anchor="w")
        self.entry_count = ctk.CTkEntry(config_frame, width=200, font=ctk.CTkFont(size=14))
        self.entry_count.insert(0, "10")
        self.entry_count.pack(anchor="w", pady=(2, 15))
        
        # Thời gian chờ
        ctk.CTkLabel(config_frame, text="Thời gian chụp mỗi đề (giây):").pack(anchor="w")
        self.entry_interval = ctk.CTkEntry(config_frame, width=200, font=ctk.CTkFont(size=14))
        self.entry_interval.insert(0, "8")
        self.entry_interval.pack(anchor="w", pady=(2, 15))

        # Buttons
        self.btn_start = ctk.CTkButton(
            config_frame, text="▶ BẮT ĐẦU CHẤM", command=self._start_auto,
            height=45, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#27ae60", hover_color="#229954"
        )
        self.btn_start.pack(fill="x", pady=(10, 5))

        self.btn_stop = ctk.CTkButton(
            config_frame, text="⏹ HỦY TRÌNH TỰ", command=self._stop_auto,
            height=45, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#c0392b", hover_color="#a53125",
            state="disabled"
        )
        self.btn_stop.pack(fill="x")

        # Logs
        ctk.CTkLabel(left_panel, text="Lịch sử xử lý:", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=15, pady=(20, 5))
        self.log_textbox = ctk.CTkTextbox(left_panel, font=ctk.CTkFont(size=12, family="Consolas"))
        self.log_textbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.log_textbox.configure(state="disabled")

        # Right Panel (Camera Preview + History)
        right_panel = ctk.CTkFrame(body, fg_color="transparent")
        right_panel.pack(side="right", fill="both", expand=True)
        
        # 1. Camera Frame (Top)
        cam_frame = ctk.CTkFrame(right_panel, fg_color="#0d0d1a", corner_radius=10)
        cam_frame.pack(side="top", fill="both", expand=True, pady=(0, 10))
        
        self.cam_label = ctk.CTkLabel(
            cam_frame, text="Đang khởi động camera...",
            font=ctk.CTkFont(size=16), text_color="gray50",
        )
        self.cam_label.pack(expand=True, fill="both", padx=5, pady=5)

        # Overlay Coundown Label
        self.countdown_lbl = ctk.CTkLabel(
            self.cam_label, text="",
            font=ctk.CTkFont(size=40, weight="bold"), text_color="#e74c3c",
            fg_color="#000000", corner_radius=10
        )
        
        # 2. History Frame (Bottom)
        hist_frame = ctk.CTkFrame(right_panel, fg_color="gray17", corner_radius=10, height=220)
        hist_frame.pack(side="bottom", fill="x")
        hist_frame.pack_propagate(False)

        ctk.CTkLabel(hist_frame, text="📸 Lịch sử ảnh đã chấm:", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.history_scroll = ctk.CTkScrollableFrame(hist_frame, orientation="horizontal", fg_color="gray20", height=160)
        self.history_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ── CAMERA LOGIC ─────────────────────────────────────────────────────────

    def _start_camera(self):
        success = self.camera.start()
        if not success:
            self.status_lbl.configure(text="● LỖI CAMERA", text_color="#e74c3c")
            self.cam_label.configure(text="Không thể mở camera.")
            return

        self._is_polling = True
        self._poll_frame()

    def _poll_frame(self):
        if not self._is_polling or not self.winfo_exists():
            return
            
        frame = self.camera.get_latest_frame()
        if frame is not None:
            preview = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            self._render_frame(preview)
                
        self.after(33, self._poll_frame)

    def _render_frame(self, frame: np.ndarray):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            w = self.cam_label.winfo_width() or 640
            h = self.cam_label.winfo_height() or 480
            if w < 10 or h < 10: return
            
            ratio = min(w / pil_img.width, h / pil_img.height)
            nw, nh = int(pil_img.width * ratio), int(pil_img.height * ratio)
            
            self._ctk_img_ref = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(nw, nh))
            self.cam_label.configure(image=self._ctk_img_ref, text="")
        except Exception:
            pass

    # ── LOGIC ────────────────────────────────────────────────────────────────

    def _append_log(self, text: str):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", text + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def _start_auto(self):
        try:
            self.total_exams = int(self.entry_count.get().strip())
            self.interval = int(self.entry_interval.get().strip())
            if self.total_exams <= 0 or self.interval <= 0:
                raise ValueError
        except:
            self._append_log("❌ Lỗi: Số lượng và thời gian phải là số nguyên dương.")
            return

        self.current_exam = 1
        self.is_running = True
        
        self.btn_start.configure(state="disabled")
        self.entry_count.configure(state="disabled")
        self.entry_interval.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        
        self._append_log("-----------------------------------------")
        self._append_log(f"▶ BẮT ĐẦU CHẤM TỔNG CỘNG {self.total_exams} ĐỀ")
        self.countdown_lbl.place(relx=0.5, rely=0.1, anchor="center")
        
        self._run_next_exam()

    def _stop_auto(self):
        self.is_running = False
        self.status_lbl.configure(text="● ĐÃ HỦY", text_color="#e67e22")
        self.countdown_lbl.place_forget()
        self._append_log("⏹ Đã hủy trình tự chấm tự động.")
        
        self.btn_start.configure(state="normal")
        self.entry_count.configure(state="normal")
        self.entry_interval.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def _run_next_exam(self):
        if not self.is_running:
            return
            
        if self.current_exam > self.total_exams:
            self._append_log("🎉 ĐÃ HOÀN THÀNH TẤT CẢ CÁC ĐỀ!")
            self.status_lbl.configure(text="● HOÀN THÀNH", text_color="#3498db")
            self._stop_auto()
            return

        self.current_countdown = self.interval
        self._countdown_step()

    def _countdown_step(self):
        if not self.is_running:
            return

        if self.current_countdown > 0:
            self.status_lbl.configure(
                text=f"● ĐỀ {self.current_exam}/{self.total_exams} - CHỤP SAU {self.current_countdown}s",
                text_color="#e74c3c"
            )
            self.countdown_lbl.configure(text=f" Chụp trong: {self.current_countdown}s ", fg_color="#e74c3c")
            
            self.current_countdown -= 1
            self.after(1000, self._countdown_step)
        else:
            self._do_capture_and_process()

    def _do_capture_and_process(self):
        if not self.is_running: return
        
        frame = self.camera.get_latest_frame()
        if frame is None:
            self._append_log("⚠️ Lỗi không lấy được khung hình camera. Bỏ qua đề này...")
        else:
            # Enhanced frame
            enhanced = SmartCamera.enhance_for_grading(frame)
            
            self.status_lbl.configure(text=f"● ĐÃ CHỤP XONG ĐỀ {self.current_exam}", text_color="#f1c40f")
            self.countdown_lbl.configure(text=f" ĐÃ CHỤP XONG ", fg_color="#27ae60")
            
            self._append_log(f"📷 Đã chụp đề {self.current_exam}. Đang chấm ngầm...")
            
            # Start background thread to process so UI doesn't block
            t = threading.Thread(
                target=self._process_background, 
                args=(enhanced.copy(), self.current_exam), 
                daemon=True
            )
            t.start()

        self.current_exam += 1
        
        # Đợi 1 giây để người dùng đọc chữ "ĐÃ CHỤP XONG" trước khi bắt đầu đếm ngược đề mới
        self.after(1500, self._run_next_exam)

    def _process_background(self, frame_bgr, exam_index):
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(tmp_fd)
        cv2.imwrite(tmp_path, frame_bgr)
        
        try:
            # Chạy OMR từ SystemCamera
            result = self.main_app.omr.process_image(tmp_path, answer_key_db=self.main_app.answer_key_db)
            
            sbd_str = result.get('sbd', '?')
            raw_score = result.get('score', 0.0)
            mcq_score = (raw_score / 10.0) * 4.0 # Tối đa trắc nghiệm là 4.0 điểm như main_window
            
            # Export luôn ra Excel
            self.main_app.exporter.export(
                mssv=sbd_str,
                mcq_score=mcq_score,
                essay_score=0.0,
                total_score=mcq_score,
                essay_details=[]
            )
            
            log_msg = f"✅ KP Đề {exam_index}: SBD [{sbd_str}] - TN: {mcq_score:.2f}đ - Đã Lưu!"
            # Gửi tín hiệu để UI update
            annotated_img = result.get('annotated_image', None)
            
            self.after(0, lambda: self._append_log(log_msg))
            if annotated_img is not None:
                self.after(0, lambda img=annotated_img.copy(), idx=exam_index, s=mcq_score: self._add_history_image(img, idx, s))

        except Exception as e:
            err_msg = f"❌ Lỗi Đề {exam_index}: {e}"
            self.after(0, lambda: self._append_log(err_msg))
            
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

    def _add_history_image(self, bgr_img, exam_idx, score):
        try:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            
            # Resize thumbnail (Ví dụ height cố định = 130px)
            h = 130
            ratio = h / pil_img.height
            w = int(pil_img.width * ratio)
            
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))
            
            item_frame = ctk.CTkFrame(self.history_scroll, fg_color="transparent")
            item_frame.pack(side="left", padx=5)
            
            lbl_img = ctk.CTkLabel(item_frame, text="", image=ctk_img, cursor="hand2")
            lbl_img.pack()
            # Giữ reference để không bị GC thu hồi
            lbl_img.image = ctk_img 
            
            # Click event để xem ảnh lớn
            lbl_img.bind("<Button-1>", lambda e, img_pass=pil_img.copy(), i=exam_idx, s=score: self._show_full_image(img_pass, i, s))
            
            lbl_txt = ctk.CTkLabel(item_frame, text=f"Đề {exam_idx}: {score:.2f}đ", font=ctk.CTkFont(size=11, weight="bold"), text_color="#2ecc71")
            lbl_txt.pack(pady=(4, 0))
            
            # Cuộn scroll view đến cuối (không có phương thức .see hay list nào, tk có xview_moveto)
            self.history_scroll._parent_canvas.xview_moveto(1.0)
        except Exception as e:
            self._append_log(f"⚠️ Không thể hiển thị ảnh lịch sử đề {exam_idx}: {e}")

    def _show_full_image(self, pil_img, exam_idx, score):
        viewer = ctk.CTkToplevel(self)
        viewer.title(f"🔍 Chi tiết Đề {exam_idx} - Điểm: {score:.2f}")
        viewer.geometry("860x680")
        viewer.transient(self)
        viewer.focus_force()
        # Không dùng grab_set() để người dùng vẫn có thể bấm nút Tạm dừng ở cửa sổ sau nếu muốn
        
        # Tính tỷ lệ thu phóng vừa cửa sổ
        viewer.update_idletasks()
        w, h = 840, 660 
        ratio = min(w / pil_img.width, h / pil_img.height)
        nw, nh = int(pil_img.width * ratio), int(pil_img.height * ratio)
        if nw <= 0 or nh <= 0: return

        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(nw, nh))
        lbl = ctk.CTkLabel(viewer, text="", image=ctk_img)
        lbl.pack(expand=True, fill="both", padx=10, pady=10)

    # ── CLEANUP ──────────────────────────────────────────────────────────────

    def _on_close(self):
        self.is_running = False
        self._is_polling = False
        self.camera.stop()
        self.destroy()
