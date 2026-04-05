# Đưa text CRNN đọc được + Rubric (đáp án) vào Llama prompt để chấm điểm
import json
from groq import Groq

class LLMGrader:
    def __init__(self, api_key):
        print("[LLM Grader] Đang khởi tạo kết nối API tới Groq...")
        self.client = Groq(api_key=api_key)
        # Sử dụng Llama 3.1 8B (Nhanh, thông minh, hỗ trợ JSON cực tốt)
        self.model = "llama-3.1-8b-instant" 

    def grade(self, reference_ans, ocr_ans, question=""):
        """
        Gửi đáp án lên Llama để chấm điểm và trả về chuẩn JSON.
        Nếu có câu hỏi (question), AI sẽ dùng ngữ cảnh để đánh giá
        câu trả lời khác chữ nhưng cùng ý nghĩa.
        """
        # Xây dựng phần ngữ cảnh câu hỏi
        question_context = ""
        if question and question.strip():
            question_context = f"""
        Câu hỏi gốc: "{question}"
        QUAN TRỌNG: Hãy dùng câu hỏi gốc để hiểu ngữ cảnh. Nếu học sinh trả lời đúng ý của câu hỏi 
        dù cách diễn đạt khác với đáp án chuẩn, vẫn tính là ĐÚNG."""

        prompt = f"""
        Bạn là một giáo viên chấm thi tự luận công tâm và chi tiết.
        Nhiệm vụ: So sánh 'Câu trả lời của học sinh' (do máy quét OCR đọc được) với 'Đáp án chuẩn'.
        {question_context}
        Đáp án chuẩn: "{reference_ans}"
        Câu trả lời của học sinh: "{ocr_ans}"
        
        Quy tắc chấm:
        1. Máy quét OCR hay bị lỗi nhầm nét (VD: 'o' thành '0', 'l' thành 'I', thiếu dấu cách). Hãy lờ đi các lỗi đánh vần nhỏ này.
        2. Nếu ý nghĩa tương đương hoặc học sinh trả lời đúng trọng tâm: ĐÚNG.
        3. Nếu nội dung sai lệch hoàn toàn, ngược nghĩa, hoặc vô nghĩa: SAI.
        
        BẮT BUỘC chỉ trả về duy nhất định dạng JSON sau, không có thêm bất kỳ câu chữ nào khác:
        {{"result": "ĐÚNG" hoặc "SAI", "reason": "Nhận xét rõ ràng (20-40 chữ): nêu cụ thể điểm giống/khác giữa bài làm và đáp án, giải thích tại sao ĐÚNG hoặc SAI"}}
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"}, # Ép buộc Llama phải nhả ra JSON
                temperature=0.1 # Nhiệt độ thấp = Kết quả ổn định, không chém gió
            )
            
            # Lấy chuỗi JSON từ kết quả trả về và chuyển thành Dictionary
            response_text = chat_completion.choices[0].message.content
            return json.loads(response_text)
            
        except Exception as e:
            print(f"[LLM Grader] Lỗi gọi API: {e}")
            return {"result": "LỖI", "reason": "Không kết nối được với máy chủ chấm điểm."}

# =====================================================================
# PHẦN TEST ĐỘC LẬP (Không cần ảnh, chỉ test text)
# =====================================================================
if __name__ == "__main__":
    # Thay API Key của bạn vào đây (Bắt đầu bằng gsk_...)
    API_KEY = "" 
    
    grader = LLMGrader(api_key=API_KEY)
    
    # Giả lập đáp án và kết quả OCR (cố tình để OCR đọc sai chữ O thành số 0)
    ref_text = "hello world"
    ocr_text = "hell0 w0rld" 
    
    print(f"Đáp án chuẩn : {ref_text}")
    print(f"OCR đọc được: {ocr_text}")
    print("\n=> Đang gửi lên API để chấm điểm...")
    
    result = grader.grade(ref_text, ocr_text)
    
    print("\n--- KẾT QUẢ TỪ LLAMA ---")
    print(f"Trạng thái : {result.get('result')}")
    print(f"Lý do      : {result.get('reason')}")