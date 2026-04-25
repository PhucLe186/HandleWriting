# Đưa text CRNN đọc được + Rubric (đáp án) vào Llama prompt để chấm điểm
import json
from groq import Groq

# Điền API Key của GROQ vào đây (Bắt đầu bằng gsk_...):
API_KEY = "YOUR_KEY"

class LLMGrader:
    def __init__(self):
        print("[LLM Grader] Đang khởi tạo kết nối API tới Groq...")
        self.client = Groq(api_key=API_KEY)
        # Sử dụng Llama 3.1 8B (Nhanh, thông minh, hỗ trợ JSON cực tốt)
        self.model = "llama-3.1-8b-instant" 

    def grade(self, reference_ans, ocr_ans, question=""):
        """
        Gửi đáp án lên Llama để chấm điểm và trả về chuẩn JSON.
        Bao gồm ngữ cảnh từ câu hỏi, đáp án chuẩn, và tiến hành chấm cực kỳ khắt khe.
        """
        # Xây dựng phần ngữ cảnh câu hỏi
        question_context = ""
        if question and question.strip():
            question_context = f"""
        Câu hỏi gốc: "{question}"
        YÊU CẦU: Hãy phân tích kỹ câu hỏi gốc để làm cơ sở đối chiếu khắt khe đáp án chuẩn và câu trả lời của học sinh."""

        prompt = f"""
        Bạn là một giám khảo chấm thi tự luận cực kỳ khắt khe và KHÔNG NHÂN NHƯỢNG.
        Nhiệm vụ: So sánh 'Câu trả lời của học sinh' (do máy quét OCR đọc được) với 'Đáp án chuẩn' dựa trên 'Câu hỏi gốc' (nếu có).
        {question_context}
        Đáp án chuẩn: "{reference_ans}"
        Câu trả lời của học sinh: "{ocr_ans}"
        
        Quy tắc chấm (KHẮT KHE):
        1. Xem xét thật kỹ câu hỏi, đáp án chuẩn, và câu trả lời của học sinh. Học sinh phải trả lời ĐẦY ĐỦ Ý, ĐÚNG TRỌNG TÂM như đáp án chuẩn thì mới được đánh là ĐÚNG.
        2. Nếu câu trả lời thiếu ý quan trọng, trả lời nửa vời, chung chung, hoặc diễn đạt sai lệch: Bắt buộc đánh SAI. Không châm chước kiểu "gần đúng".
        3. Chỉ bỏ qua các lỗi nhận diện OCR hiển nhiên (VD: 'o' thành '0', 'l' thành 'I', lỗi khoảng trắng). Tuyệt đối không bỏ qua lỗi sai kiến thức.
        4. Bất kỳ sự mâu thuẫn hay sai kiến thức nào đều dẫn đến kết quả là SAI.
        
        BẮT BUỘC chỉ trả về duy nhất định dạng JSON sau, không có thêm bất kỳ câu chữ nào khác:
        {{"result": "ĐÚNG" hoặc "SAI", "reason": "Nhận xét khắt khe (20-40 chữ): giải thích chi tiết tại sao bài làm thiếu sót dẫn đến SAI, hoặc nếu hoàn toàn đúng thì tại sao ĐÚNG"}}
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
            error_msg = str(e)
            print(f"[LLM Grader] Lỗi gọi API: {error_msg}")
            return {"result": "LỖI", "reason": f"Lỗi chi tiết: {error_msg}"}

# =====================================================================
# PHẦN TEST ĐỘC LẬP (Không cần ảnh, chỉ test text)
# =====================================================================
if __name__ == "__main__":
    
    grader = LLMGrader()
    
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