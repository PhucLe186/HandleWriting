class ScoreCalculator:
    def __init__(self, mcq_weight=0.1, essay_weight=1.0):
        # Ví dụ: Trắc nghiệm 40 câu (mỗi câu 0.1đ) = 4 điểm
        # Tự luận 6 câu (mỗi câu 1đ) = 6 điểm
        self.mcq_weight = mcq_weight
        self.essay_weight = essay_weight

    def calculate_total(self, mcq_details, essay_results):
        total_mcq_score = 0.0
        total_essay_score = 0.0

        # Tính điểm trắc nghiệm
        if mcq_details:
            correct_mcq = sum(1 for detail in mcq_details.values() if detail['correct'])
            total_mcq_score = correct_mcq * self.mcq_weight

        # Tính điểm tự luận
        if essay_results:
            correct_essay = sum(1 for res in essay_results if res['Trạng thái'] == "ĐÚNG")
            total_essay_score = correct_essay * self.essay_weight

        final_score = total_mcq_score + total_essay_score

        return {
            "mcq_score": total_mcq_score,
            "essay_score": total_essay_score,
            "final_score": final_score
        }