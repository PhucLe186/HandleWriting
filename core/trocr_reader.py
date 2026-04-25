import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np

class TrOCRReader:
    def __init__(self, model_name='microsoft/trocr-large-handwritten'):
        print(f"[TrOCR] Đang tải mô hình từ HuggingFace: {model_name}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        
        print(f"[TrOCR] Tải mô hình thành công lên {self.device}!")

    def extract_text(self, img_array):
        if img_array is None or img_array.size == 0:
            return ""

        # 1. Chuyển đổi sang PIL Image
        if isinstance(img_array, np.ndarray):
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_rgb).convert("RGB")
        else:
            image = img_array.convert("RGB")

        # 2. Tiền xử lý
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        # 3. Sử dụng Model với cấu hình NGĂN CHẶN SÁNG TẠO
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=64,
                
                # --- THAY ĐỔI QUAN TRỌNG TẠI ĐÂY ---
                num_beams=1,           # Sử dụng Greedy Search (không ngồi suy nghĩ nhiều phương án)
                do_sample=False,       # Tắt tính năng lấy mẫu ngẫu nhiên
                temperature=1.0,       # Giữ nguyên mức nhiệt độ mặc định nhưng không cho phép sáng tạo vì do_sample=False
                # ----------------------------------
            )
            
        # 4. Giải mã kết quả
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text

# =====================================================================
# PHẦN TEST ĐỘC LẬP
# =====================================================================
if __name__ == "__main__":
    import os
    reader = TrOCRReader()

    test_path = r"D:\Project\CheckPointLasted\data\temp_crops\cau_4_cls_0.jpg"
    
    if os.path.exists(test_path):
        img = cv2.imread(test_path)
        result = reader.extract_text(img)
        
        print("\n" + "="*40)
        print(f"KẾT QUẢ TrOCR THỰC TẾ: '{result}'")
        print("="*40)
        
        cv2.imshow("Anh dau vao", img)
        cv2.waitKey(0)
    else:
        print("Không tìm thấy ảnh test!")