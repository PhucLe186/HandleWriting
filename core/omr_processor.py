# Thuật toán xử lý ảnh, đếm bóng (bubble) cho bài trắc nghiệm
import cv2
import numpy as np


class OMRProcessor:
    def __init__(self):
        self.PAPER_WIDTH_MM = 210.0
        self.PAPER_HEIGHT_MM = 297.0
        # 10 pixels per mm -> 2100 x 2970 image. High precision.
        self.PPM = 10 
        self.warp_width = int(self.PAPER_WIDTH_MM * self.PPM)
        self.warp_height = int(self.PAPER_HEIGHT_MM * self.PPM)

        # Expected marker coordinates in mm
        # 6.5mm x 6.5mm squares at the corners
        self.MARKER_TL_MM = (11.0, 12.0)
        self.MARKER_TR_MM = (199.0, 12.0)
        self.MARKER_BL_MM = (11.0, 285.0)
        self.MARKER_BR_MM = (199.0, 285.0)

        self.MARKER_TL_PX = (int(self.MARKER_TL_MM[0] * self.PPM), int(self.MARKER_TL_MM[1] * self.PPM))
        self.MARKER_TR_PX = (int(self.MARKER_TR_MM[0] * self.PPM), int(self.MARKER_TR_MM[1] * self.PPM))
        self.MARKER_BL_PX = (int(self.MARKER_BL_MM[0] * self.PPM), int(self.MARKER_BL_MM[1] * self.PPM))
        self.MARKER_BR_PX = (int(self.MARKER_BR_MM[0] * self.PPM), int(self.MARKER_BR_MM[1] * self.PPM))

        # 9 Inner Alignment Markers (4.5x4.5 mm) for local distortion correction
        ans_1_10_y = 77.0
        bottom_y = ans_1_10_y + 6.5 + 9 * 8.0 + 9.5 # 165
        self.INNER_MARKERS_MM = [
            # Top row
            (28.5, 75.5), (86.5, 75.5), (144.5, 75.5),
            # Mid row
            (28.5, 163.5), (86.5, 163.5), (144.5, 163.5),
            # Bot row
            (28.5, bottom_y + 86.5), (86.5, bottom_y + 86.5), (144.5, bottom_y + 86.5) # 251.5
        ]

    def detect_markers(self, image):
        """Find the 4 corner alignment markers."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold to isolate dark features
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 51, 15)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            # A 6.5x6.5 mm marker - lower threshold for webcams with lower resolution
            if area < 30: 
                continue
                
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            
            # The markers are solid squares
            if 0.65 <= aspect_ratio <= 1.35:
                extent = area / (w * h)
                if extent > 0.65: # Solid shape
                    # Add center of mass
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        # Tính điểm: càng vuông vắn càng tốt (gần 1.0)
                        score = abs(aspect_ratio - 1.0) + abs(extent - 1.0)
                        candidates.append((score, cx, cy))
        
        if len(candidates) < 4:
            return None # Not enough markers found
            
        # Chỉ lấy 35 ứng viên vuông vức nhất để tăng tốc duyệt tổ hợp
        candidates.sort(key=lambda x: x[0])
        best_cands = [(c[1], c[2]) for c in candidates[:35]]
        
        import itertools
        best_corners = None
        best_area = 0
        
        # Hàm tính khoảng cách giữa 2 điểm
        def dist(p1, p2): 
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        # Quét mọi cụm 4 điểm để tìm ra tứ giác giống khung phiếu A4 nhất
        for quad in itertools.combinations(best_cands, 4):
            quad = np.array(quad)
            sums = quad.sum(axis=1)
            diffs = np.diff(quad, axis=1) # y - x
            
            tl = quad[np.argmin(sums)]
            br = quad[np.argmax(sums)]
            tr = quad[np.argmin(diffs)]
            bl = quad[np.argmax(diffs)]
            
            w_top = dist(tl, tr)
            w_bot = dist(br, bl)
            h_left = dist(tl, bl)
            h_right = dist(tr, br)
            
            if w_top == 0 or w_bot == 0 or h_left == 0 or h_right == 0: continue
            
            # Hai cạnh đối diện không được lệch nhau quá 35% (để chống méo perspective biến dạng nặng)
            if max(w_top, w_bot) / min(w_top, w_bot) > 1.35: continue
            if max(h_left, h_right) / min(h_left, h_right) > 1.35: continue
            
            avg_w = (w_top + w_bot) / 2.0
            avg_h = (h_left + h_right) / 2.0
            ratio = avg_w / avg_h
            
            # Tỷ lệ 4 góc chuẩn của OMR là 188mm / 273mm = 0.688. 
            # Cho phép từ 0.35 đến 1.1 để bù trừ góc máy nghiêng.
            if 0.35 < ratio < 1.1:
                area = avg_w * avg_h
                # Lấy khung hình chữ nhật có diện tích LỚN NHẤT
                if area > best_area:
                    best_area = area
                    best_corners = np.array([tl, tr, br, bl], dtype="float32")
                    
        return best_corners

    def align_document(self, image):
        src_pts = self.detect_markers(image)
        if src_pts is None:
            return None
            
        dst_pts = np.array([
            self.MARKER_TL_PX,
            self.MARKER_TR_PX,
            self.MARKER_BR_PX,
            self.MARKER_BL_PX
        ], dtype="float32")
        
        # Perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (self.warp_width, self.warp_height))
        return warped

    def refine_inner_markers(self, thresh_warped):
        """Find the exact center of the 9 inner markers to build local offsets."""
        actual_markers = {}
        for (mx_mm, my_mm) in self.INNER_MARKERS_MM:
            mx = int(mx_mm * self.PPM)
            my = int(my_mm * self.PPM)
            
            # Search window of +/- 45 pixels (4.5 mm) around expected center
            search_radius = 45
            x1 = max(0, mx - search_radius)
            y1 = max(0, my - search_radius)
            x2 = min(thresh_warped.shape[1], mx + search_radius)
            y2 = min(thresh_warped.shape[0], my + search_radius)
            
            roi = thresh_warped[y1:y2, x1:x2]
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_c = None
            best_area = -1
            # Inner markers are 4.5mmx4.5mm => ~45x45 pixels = ~2000 area
            for c in contours:
                area = cv2.contourArea(c)
                # Filter by roughly matching the area 
                if 500 < area < 4000:
                    best_c = c
                    best_area = area
                    
            if best_c is not None:
                M = cv2.moments(best_c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    actual_markers[(mx_mm, my_mm)] = (cx, cy)
                else:
                    actual_markers[(mx_mm, my_mm)] = (mx, my) # fallback theoretical
            else:
                actual_markers[(mx_mm, my_mm)] = (mx, my) # fallback theoretical
                
        return actual_markers

    def get_interpolated_center(self, x_mm, y_mm, actual_markers):
        """Interpolates precise pixel coords using the 9 inner bounding anchors."""
        anchor_xs = [28.5, 86.5, 144.5]
        best_x = min(anchor_xs, key=lambda xs: abs(xs - x_mm))
        
        anchor_ys = [75.5, 163.5, 251.5]
        
        # Calculate localized offsets dx, dy
        if y_mm <= anchor_ys[0]:
            am = actual_markers[(best_x, anchor_ys[0])]
            dx = am[0] - best_x * self.PPM
            dy = am[1] - anchor_ys[0] * self.PPM
        elif y_mm >= anchor_ys[2]:
            am = actual_markers[(best_x, anchor_ys[2])]
            dx = am[0] - best_x * self.PPM
            dy = am[1] - anchor_ys[2] * self.PPM
        else:
            if y_mm <= anchor_ys[1]:
                y_top = anchor_ys[0]
                y_bot = anchor_ys[1]
            else:
                y_top = anchor_ys[1]
                y_bot = anchor_ys[2]
                
            am_top = actual_markers[(best_x, y_top)]
            dx_top = am_top[0] - best_x * self.PPM
            dy_top = am_top[1] - y_top * self.PPM
            
            am_bot = actual_markers[(best_x, y_bot)]
            dx_bot = am_bot[0] - best_x * self.PPM
            dy_bot = am_bot[1] - y_bot * self.PPM
            
            # Linear interpolation based on Y distance
            t = (y_mm - y_top) / (y_bot - y_top)
            dx = dx_top + t * (dx_bot - dx_top)
            dy = dy_top + t * (dy_bot - dy_top)
            
        cx_px = int(x_mm * self.PPM + dx)
        cy_px = int(y_mm * self.PPM + dy)
        return cx_px, cy_px

    def get_bubble_density(self, thresh_img, cx_px, cy_px, radius_mm):
        r = int(radius_mm * self.PPM)
        
        # Create a circular mask
        mask = np.zeros(thresh_img.shape, dtype="uint8")
        cv2.circle(mask, (cx_px, cy_px), r, 255, -1)
        
        # Bitwise AND to get pixels inside bubble
        bubble = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
        
        # Count non-zero pixels area
        total = cv2.countNonZero(mask)
        filled = cv2.countNonZero(bubble)
        return filled / float(total) if total > 0 else 0

    def read_grid(self, thresh, actual_markers, x_start_mm, y_grid_mm, cols, rows=10, col_w=7.5, row_gap=8.0, r=2.7):
        """Read a grid of bubbles (0-9 vertical, cols horizontal)"""
        result = []
        for c in range(cols):
            filled_digits = []
            cx_mm = x_start_mm + c * col_w + col_w / 2
            
            for r_idx in range(rows):
                cy_mm = y_grid_mm + 3.5 + r_idx * row_gap
                
                # Get precisely interpolated pixel coordinates
                cx_px, cy_px = self.get_interpolated_center(cx_mm, cy_mm, actual_markers)
                
                density = self.get_bubble_density(thresh, cx_px, cy_px, r)
                
                # Minimum fill threshold (e.g. 40%)
                if density > 0.4:
                    filled_digits.append(r_idx)
                    
            if len(filled_digits) == 1:
                result.append(str(filled_digits[0]))
            elif len(filled_digits) > 1:
                result.append("?") # Báo lỗi do tô nhiều ô
            else:
                result.append("?") # Unread/blank
        return "".join(result)

    def read_answers(self, thresh, actual_markers):
        answers = {}
        # Configuration from generate_answer_sheet.py
        q1_10_x = 142.0
        ans_1_10_y = 77.0
        
        bottom_y = ans_1_10_y + 6.5 + 9 * 8.0 + 9.5
        col1_x = 26.0
        col3_x = q1_10_x 
        col2_x = (col1_x + col3_x) / 2.0
        
        cols_config = [
            (q1_10_x, ans_1_10_y, 1, 10),
            (col1_x, bottom_y, 11, 10),
            (col2_x, bottom_y, 21, 10),
            (col3_x, bottom_y, 31, 10)
        ]
        
        choice_gap = 7.8
        row_gap_ans = 8.0
        bubble_r = 2.9
        choice_letters = ['A', 'B', 'C', 'D']
        
        for base_x, y_start, start_q, num_q in cols_config:
            bubble_start_x = base_x + 10.0
            
            for i in range(num_q):
                q_num = start_q + i
                row_y = y_start + 6.5 + i * row_gap_ans
                
                filled_choices = []
                
                for j, letter in enumerate(choice_letters):
                    cx_mm = bubble_start_x + j * choice_gap
                    cy_mm = row_y
                    
                    cx_px, cy_px = self.get_interpolated_center(cx_mm, cy_mm, actual_markers)
                    density = self.get_bubble_density(thresh, cx_px, cy_px, bubble_r)
                    
                    if density > 0.35:
                        filled_choices.append(letter)
                
                if len(filled_choices) == 1:
                    answers[q_num] = filled_choices[0]
                elif len(filled_choices) > 1:
                    answers[q_num] = "ERR" # Tô >1 đáp án
                else:
                    answers[q_num] = None
                
        return answers

    def grade_answers(self, answers, answer_key):
        correct = 0
        total = len(answer_key)
        details = {}
        for q, expected in answer_key.items():
            user_ans = answers.get(q)
            is_correct = (user_ans == expected)
            if is_correct:
                correct += 1
            details[q] = {
                'user': user_ans,
                'expected': expected,
                'correct': is_correct
            }
        
        score = (correct / total) * 10.0 if total > 0 else 0
        return score, details

    def draw_feedback(self, warped, sbd, ma_de, answers, details, answer_key, actual_markers):
        output = warped.copy()
        
        # Helper to draw circles
        def highlight_bubble(cx_mm, cy_mm, r_mm, color, thickness=2):
            cx_px, cy_px = self.get_interpolated_center(cx_mm, cy_mm, actual_markers)
            r = int(r_mm * self.PPM)
            cv2.circle(output, (cx_px, cy_px), r + 2, color, thickness)
            return cx_px, cy_px, r

        # Configs
        q1_10_x = 142.0
        ans_1_10_y = 77.0
        bottom_y = ans_1_10_y + 6.5 + 9 * 8.0 + 9.5
        col1_x = 26.0
        col3_x = q1_10_x 
        col2_x = (col1_x + col3_x) / 2.0
        cols_config = [
            (q1_10_x, ans_1_10_y, 1, 10),
            (col1_x, bottom_y, 11, 10),
            (col2_x, bottom_y, 21, 10),
            (col3_x, bottom_y, 31, 10)
        ]
        choice_gap = 7.8
        row_gap_ans = 8.0
        bubble_r = 2.9
        choice_letters = ['A', 'B', 'C', 'D']

        # Draw answers feedback
        if answer_key:
            for base_x, y_start, start_q, num_q in cols_config:
                bubble_start_x = base_x + 10.0
                for i in range(num_q):
                    q_num = start_q + i
                    row_y = y_start + 6.5 + i * row_gap_ans
                    
                    detail = details.get(q_num)
                    if not detail:
                        continue
                        
                    user = detail['user']
                    exp = detail['expected']
                    
                    # Draw user's mark
                    if user and user in choice_letters:
                        idx = choice_letters.index(user)
                        cx = bubble_start_x + idx * choice_gap
                        if detail['correct']:
                            # Green for correct
                            highlight_bubble(cx, row_y, bubble_r, (0, 200, 0), 4)
                        else:
                            # Red for wrong
                            cx_p, cy_p, r_p = highlight_bubble(cx, row_y, bubble_r, (0, 0, 255), 4)
                            # Cross out wrong
                            cv2.line(output, (cx_p-r_p, cy_p-r_p), (cx_p+r_p, cy_p+r_p), (0,0,255), 3)
                            cv2.line(output, (cx_p-r_p, cy_p+r_p), (cx_p+r_p, cy_p-r_p), (0,0,255), 3)
                    elif user == "ERR":
                        # Cảnh báo tô 2 ô
                        cx_p, cy_p = self.get_interpolated_center(bubble_start_x - 3, row_y, actual_markers)
                        cv2.putText(output, "ERR", (cx_p - 15, cy_p + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    
                    # Highlight expected answer if wrong or missed
                    if not detail['correct'] or user not in choice_letters:
                        if exp:
                            idx = choice_letters.index(exp)
                            cx = bubble_start_x + idx * choice_gap
                            highlight_bubble(cx, row_y, bubble_r, (255, 0, 0), 3) # Blue for expected

        # ── VẼ VÒNG TRÒN XANH LÁ CHO SBD & MÃ ĐỀ ĐÃ DETECT ──────────────
        sbd_x_start = 38.25
        sbd_y_grid  = 80.0
        sbd_cols    = 6
        sbd_col_w   = 7.5
        sbd_row_gap = 8.0
        sbd_r       = 2.7

        made_x_start = 89.75
        made_cols    = 3

        def draw_grid_detection(grid_x_start, grid_cols, detected_str, label):
            """Vẽ vòng tròn xanh lá lên ô bubble đã detect cho SBD hoặc Mã Đề."""
            for c in range(grid_cols):
                cx_mm = grid_x_start + c * sbd_col_w + sbd_col_w / 2
                char = detected_str[c] if c < len(detected_str) else "?"

                if char != "?":
                    digit = int(char)
                    cy_mm = sbd_y_grid + 3.5 + digit * sbd_row_gap
                    cx_px, cy_px = self.get_interpolated_center(cx_mm, cy_mm, actual_markers)
                    r_px = int(sbd_r * self.PPM)
                    # Vòng tròn XANH LÁ đậm — nghĩa là ĐÃ DETECT THÀNH CÔNG
                    cv2.circle(output, (cx_px, cy_px), r_px + 4, (0, 220, 0), 3)
                else:
                    # Vẽ dấu "?" đỏ ở đầu cột nếu không detect được
                    cy_mm = sbd_y_grid + 3.5
                    cx_px, cy_px = self.get_interpolated_center(cx_mm, cy_mm, actual_markers)
                    cv2.putText(output, "?", (cx_px - 10, cy_px - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        draw_grid_detection(sbd_x_start, sbd_cols, sbd, "SBD")
        draw_grid_detection(made_x_start, made_cols, ma_de, "MaDe")

        # ── HIỂN THỊ TEXT KẾT QUẢ Ở TRÊN ẢNH ────────────────────────────
        sbd_color  = (0, 220, 0) if "?" not in sbd else (0, 165, 255)
        made_color = (0, 220, 0) if "?" not in ma_de else (0, 165, 255)

        cv2.putText(output, f"SBD: {sbd}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, sbd_color, 3)
        cv2.putText(output, f"Ma de: {ma_de}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, made_color, 3)

        # Điểm số (nếu có đáp án)
        if answer_key and details:
            correct_count = sum(1 for d in details.values() if d['correct'])
            total_count   = len(answer_key)
            score_val     = (correct_count / total_count) * 10.0 if total_count > 0 else 0
            cv2.putText(output, f"Diem: {score_val:.1f}/10  ({correct_count}/{total_count})", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 200, 0), 3)

        return output

    def process_image(self, image_path, answer_key=None, answer_key_db=None):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load {image_path}")

        warped = self.align_document(img)
        if warped is None:
            raise ValueError("Không nhận diện được 4 góc đen trên phiếu.\nHãy để phiếu thẳng, thấy rõ 4 góc vuông và đủ sáng!")

        # Threshold for bubble density check and inner marker detection
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # Binarize with adaptive threshold. Inverse so bubbles and markers are White.
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 51, 15)

        # Refine coords using 9 inner markers
        actual_markers = self.refine_inner_markers(thresh)

        sbd = self.read_grid(thresh, actual_markers, x_start_mm=38.25, y_grid_mm=80.0, cols=6)
        ma_de = self.read_grid(thresh, actual_markers, x_start_mm=89.75, y_grid_mm=80.0, cols=3)
        answers = self.read_answers(thresh, actual_markers)

        score = 0
        details = {}
        
        active_key = answer_key
        if answer_key_db is not None:
            if ma_de in answer_key_db:
                active_key = answer_key_db[ma_de]
            else:
                active_key = None

        if active_key:
            score, details = self.grade_answers(answers, active_key)

        annotated = self.draw_feedback(warped, sbd, ma_de, answers, details, active_key, actual_markers)

        return {
            'sbd': sbd,
            'ma_de': ma_de,
            'answers': answers,
            'score': score,
            'details': details,
            'annotated_image': annotated
        }

if __name__ == "__main__":
    pass