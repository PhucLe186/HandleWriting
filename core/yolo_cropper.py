 # Load model YOLO, định vị và cắt (crop) các vùng chứa câu trả lời tự luận
import cv2
import os
from ultralytics import YOLO

class YoloCropper:
    def __init__(self, model_path, conf_thresh=0.2, imgsz=1280):
        print(f"[YOLO] Đang tải mô hình từ: {model_path}...")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        
        # Lấy đường dẫn gốc của toàn bộ project (Thư mục AutoGrader_Project)
        # __file__ là vị trí của file yolo_cropper.py hiện tại
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Trỏ thẳng vào thư mục data/temp_crops đã có sẵn của bạn
        self.temp_crop_dir = os.path.join(project_root, "data", "temp_crops")
        
        print("[YOLO] Tải mô hình thành công!")

    def detect_and_crop(self, frame, save_debug=True):
        # Chỉ xóa file ảnh bên trong, KHÔNG đụng chạm hay tạo mới thư mục
        if save_debug and os.path.exists(self.temp_crop_dir):
            for f in os.listdir(self.temp_crop_dir):
                file_path = os.path.join(self.temp_crop_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        print("[YOLO] Đang quét các vùng chứa câu trả lời...")
        
        results = self.model(
            source=frame, 
            conf=self.conf_thresh, 
            imgsz=self.imgsz, 
            iou=0.5,
            verbose=False
        )

        detected_regions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                detected_regions.append({"box": (x1, y1, x2, y2), "class_id": cls_id})

        if not detected_regions:
            print("[YOLO] Không tìm thấy vùng chữ nào trên ảnh!")
            return []

        # Sắp xếp các box từ trên xuống dưới theo tọa độ Y
        detected_regions = sorted(detected_regions, key=lambda item: item["box"][1])
        cropped_images = []

        for i, region in enumerate(detected_regions):
            x1, y1, x2, y2 = region["box"]
            cls_id = region["class_id"]
            
            crop_img = frame[y1:y2, x1:x2]
            cropped_images.append({
                "order": i,          
                "class_id": cls_id,  
                "image": crop_img    
            })

            # Lưu ảnh crop thẳng vào thư mục có sẵn
            if save_debug and os.path.exists(self.temp_crop_dir):
                filename = f"cau_{i+1}_cls_{cls_id}.jpg"
                cv2.imwrite(os.path.join(self.temp_crop_dir, filename), crop_img)

        print(f"[YOLO] Đã cắt thành công {len(cropped_images)} vùng câu trả lời.")
        return cropped_images


# =====================================================================
# PHẦN TEST ĐỘC LẬP MODULE (Chỉ chạy khi bạn run trực tiếp file này)
# =====================================================================
if __name__ == "__main__":
    # Đường dẫn file weights của bạn
    WEIGHTS = r"D:\Project\CheckPointLasted\models\yolo_weights\best.pt"
    cropper = YoloCropper(model_path=WEIGHTS)

    print("\n--- CHỌN CHẾ ĐỘ TEST ---")
    print("1. Đang chạy test bằng ảnh tĩnh có sẵn...")
    
    # Test bằng bức ảnh tĩnh của bạn
    TEST_IMG = r"D:\Project\chamdiemchosinhvien\dataset\images\test\z7497419511015_d448d2fa2ad1b0865ee202adc235299a.jpg"
    frame = cv2.imread(TEST_IMG)
    
    if frame is not None:
        results = cropper.detect_and_crop(frame, save_debug=True)
        print("=> Hãy kiểm tra thư mục 'data/temp_crops' xem ảnh cắt ra có chuẩn không nhé!\n")
    else:
        print("Không tìm thấy file ảnh test.")

    # -----------------------------------------------------------------
    # NẾU MUỐN TEST BẰNG WEBCAM: 
    # Bôi đen đoạn code dưới đây -> Nhấn Ctrl + / để mở comment
    # -----------------------------------------------------------------
    
    # print("2. Đang khởi động test bằng Webcam Maxhub...")
    # cap = cv2.VideoCapture(0) # Đổi thành 1 nếu máy bạn có 2 camera
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # while True:
    #     ret, cam_frame = cap.read()
    #     if not ret: break
        
    #     cv2.imshow("Test YOLO - Nhấn 'c' de chup, 'q' de thoat", cv2.resize(cam_frame, (1000, 750)))
    #     key = cv2.waitKey(1) & 0xFF
        
    #     if key == ord('c'):
    #         print("Đang chụp và crop...")
    #         cropper.detect_and_crop(cam_frame, save_debug=True)
    #         print("=> Đã lưu ảnh crop vào 'data/temp_crops'")
    #     elif key == ord('q'):
    #         break
            
    # cap.release()
    # cv2.destroyAllWindows()