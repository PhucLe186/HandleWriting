import cv2
import threading
import time
import numpy as np

class SmartCamera:
    """
    Core backend quản lý hoàn toàn các thao tác với camera bao gồm: 
    khởi động, tối ưu hóa phần cứng, đọc luồng hình ảnh bằng background thread
    và áp dụng bộ lọc tăng cường ảnh (enhancement).
    """
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.latest_frame = None
        self.thread = None

    def start(self) -> bool:
        """Khởi động camera và trả về True nếu thành công."""
        # Dùng DirectShow thay cho MSMF để tránh lỗi "can't grab frame" trên Windows
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print(f"⚠️ Không tìm thấy camera tại index {self.camera_index}.")
            return False

        # --- CẤU HÌNH TỐI ƯU CHO SONY 10 MARK 3 (12MP / 4K) ---
        # # Bật full độ phân giải tối đa 12MP (4000x3000) 4:3 để đọc chữ sắc nét trên giấy A4
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # # Bật HDR / Backlight Compensation của camera để nhận diện chữ tốt hơn môi trường độ tương phản cao
      
        
        # # Lấy nét tự động, phơi sáng và cân bằng trắng tự động
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)

        # ============================================================
        # CẤU HÌNH CHO MAXHUB UC W21 (4K) (MỞ COMMENT NẾU ĐỔI CAMERA)
        # ============================================================
        # Tận dụng cảm biến 4K (8.42MP) siêu nét để đọc kỹ tự/chấm OMR
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Maxhub UCW21 có hệ thống lấy nét tự động PDAF chính xác
        # self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        # Cho phép WDR và Color Balance tự động của UC W21 hoạt động tối đa
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 
        # self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self):
        """Background thread: liên tục đọc frame vào buffer."""
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Cắt ảnh (center crop) về tỷ lệ 4:3 để loại bỏ hiệu ứng góc quá rộng ở 2 bên
                    h, w = frame.shape[:2]
                    target_w = int(h * 4 / 3)
                    if w > target_w:
                        margin = (w - target_w) // 2
                        frame = frame[:, margin:margin + target_w]
                        
                    self.latest_frame = frame
            time.sleep(0.03)

    def get_latest_frame(self) -> np.ndarray | None:
        """Trả về frame mới nhất từ buffer (có thể là None nếu camera chưa sẵn sàng)."""
        return self.latest_frame

    def stop(self):
        """Dừng camera và giải phóng tài nguyên."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    @staticmethod
    def enhance_for_grading(frame: np.ndarray) -> np.ndarray:
        """
        Xử lý ảnh SAU KHI CHỤP để chấm bài.
        """
        # 1. Tự động dãn độ tương phản (Auto Normalize) - Giúp nền giấy trắng sáng tự nhiên, mực vừa đủ đậm
        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # 2. Tăng cường chi tiết thông minh (Detail Enhancement)
        # Thuật toán này của OpenCV khử mờ rìa chữ/ô OMR rất tốt mà KHÔNG sinh ra nhiễu hạt (noise)
        frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        
        return frame