import pandas as pd
import os
from datetime import datetime

class ExcelExporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self, mssv, mcq_score, essay_score, total_score, essay_details):
        """
        Xuất file Excel. Xử lý an toàn tên file để tránh lỗi hệ điều hành Windows.
        """
        # 1. Tạo DataFrame cho thông tin tổng quát
        summary_data = [{
            "MSSV": mssv,
            "Điểm Trắc Nghiệm": round(mcq_score, 2),
            "Điểm Tự Luận": round(essay_score, 2),
            "TỔNG ĐIỂM": round(total_score, 2)
        }]
        df_summary = pd.DataFrame(summary_data)

        # 2. Tạo DataFrame cho chi tiết tự luận
        df_details = pd.DataFrame(essay_details)

        # 3. XỬ LÝ AN TOÀN TÊN FILE (Khắc phục lỗi Errno 22)
        # Thay thế dấu '?' thành chữ 'X'
        safe_mssv = str(mssv).replace("?", "X")
        
        # Nếu mã toàn là XXXXXX hoặc rỗng thì gán là KhongRoMSSV
        if safe_mssv == "XXXXXX" or safe_mssv.strip() == "":
            safe_mssv = "KhongRoMSSV"

        # Thêm timestamp (giờ phút giây) để tránh việc các bài thi trùng/không rõ MSSV bị ghi đè lên nhau
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{safe_mssv}_{timestamp}_DiemThi.xlsx"
        filepath = os.path.join(self.output_dir, filename)

        # 4. Ghi ra Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Tổng Điểm', index=False)
            if not df_details.empty:
                df_details.to_excel(writer, sheet_name='Chi Tiết Tự Luận', index=False)
        
        print(f"\n✅ [Excel] Đã lưu bảng điểm của SV {mssv} vào: {filepath}")
        return filepath