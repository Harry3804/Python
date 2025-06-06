import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from fractions import Fraction
import io
from contextlib import redirect_stdout
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')
from typing import List, Tuple, Optional, Dict
import copy
from decimal import Decimal, getcontext

from fractions import Fraction

class SimplexDictionary:
    def __init__(self, c, A, b, problem_type='max'):
        self.n = len(c)
        self.m = len(b)
        self.vars = ['x' + str(i + 1) for i in range(self.n)]
        self.problem_type = problem_type.lower()
        self.steps_output = []

        # Lưu trữ hệ số gốc của hàm mục tiêu
        self.original_c = [Fraction(ci) for ci in c]

        # Kiểm tra xem có b_i < 0 không
        self.has_negative_b = any(bi < 0 for bi in b)
        
        # Kiểm tra xem có b_i = 0 không (degeneracy)
        self.has_zero_b = any(bi == 0 for bi in b)
        
        if self.has_negative_b:
            self.steps_output.append("### Vì tồn tại b_i < 0 nên không sử dụng được thuật toán đơn hình")
            self.steps_output.append("Cần sử dụng phương pháp hai pha hoặc Big M để giải quyết bài toán này.")
            self.A = [[Fraction(aij) for aij in row] for row in A]
            self.b = [Fraction(bi) for bi in b]
            self.z = []
            self.slack = []
            self.basic = []
            self.nonbasic = []
            return
        
        if self.has_zero_b:
            self.steps_output.append("### Vì tồn tại b_i = 0 nên hãy sử dụng thuật toán Bland")
            self.steps_output.append("Thuật toán Bland sẽ tránh được hiện tượng cycling (lặp vô hạn) do degeneracy.")

        # Xử lý bình thường khi tất cả b_i >= 0
        self.A = [[Fraction(aij) for aij in row] for row in A]
        self.b = [Fraction(bi) for bi in b]

        # Tạo biến slack
        self.slack = ['w' + str(i + 1) for i in range(self.m)]
        self.basic = self.slack.copy()
        self.nonbasic = self.vars.copy()

        # Thiết lập hàm mục tiêu
        if self.problem_type == 'max':
            self.z = [-Fraction(ci) for ci in c]
        else:
            self.z = [Fraction(ci) for ci in c]

        # Thêm biến slack vào ma trận A
        for i in range(self.m):
            slack_col = [Fraction(1 if j == i else 0) for j in range(self.m)]
            self.A[i] = self.A[i] + slack_col
        
        # Thêm hệ số 0 cho biến slack trong hàm mục tiêu
        self.z += [Fraction(0)] * self.m

    def get_all_variables(self):
        """Trả về danh sách tất cả biến theo thứ tự: biến gốc + biến slack"""
        return self.vars + self.slack

    def vars_and_slack(self):
        """Trả về danh sách tất cả biến (alias cho get_all_variables)"""
        return self.get_all_variables()

    def render_latex(self):
        latex = "$$\\begin{aligned}\n"
        latex += "z &= " + self.expr_to_latex(self.z, self.get_all_variables()) + " \\\\\n"
        for i in range(self.m):
            latex += self.basic[i] + " &= " + str(self.b[i])
            for j, coeff in enumerate(self.A[i]):
                if coeff == 0:
                    continue
                sign = ' + ' if coeff > 0 else ' - '
                latex += f"{sign} {abs(coeff)}{self.get_all_variables()[j]}"
            latex += " \\\\\n"
        latex += "\\end{aligned}$$"
        return latex

    def expr_to_latex(self, coeffs, vars):
        terms = []
        for coeff, var in zip(coeffs, vars):
            if coeff == 0:
                continue
            if coeff < 0:
                terms.append(f"- {abs(coeff)}{var}")
            else:
                terms.append(f"+ {abs(coeff)}{var}")
        
        if not terms:
            return "0"
            
        expr = " ".join(terms)
        if expr.startswith('+ '):
            expr = expr[2:]
        return expr

    def calculate_optimal_value(self):
        if self.has_negative_b:
            return None
            
        solution = self.get_optimal_solution()
        optimal_value = sum(self.original_c[i] * solution[self.vars[i]] for i in range(self.n))
        return optimal_value

    def get_optimal_solution(self):
        if self.has_negative_b:
            return {}
            
        solution = {}
        
        # Khởi tạo tất cả biến với giá trị 0
        all_vars = self.get_all_variables()
        for var in all_vars:
            solution[var] = Fraction(0)
            
        # Biến cơ sở có giá trị từ vector b
        for i, var in enumerate(self.basic):
            solution[var] = self.b[i]
            
        return solution

    def pivot(self, entering_idx, leaving_idx):
        pivot_element = self.A[leaving_idx][entering_idx]
        
        if pivot_element == 0:
            raise ValueError("Phần tử trục không thể bằng 0")
        
        # Lấy tên biến vào và biến ra
        all_vars = self.get_all_variables()
        entering_var = all_vars[entering_idx]
        leaving_var = self.basic[leaving_idx]
        
        # Chuẩn hóa hàng trục
        for j in range(len(self.A[leaving_idx])):
            self.A[leaving_idx][j] /= pivot_element
        self.b[leaving_idx] /= pivot_element
        
        # Khử biến vào khỏi các hàng khác
        for i in range(self.m):
            if i != leaving_idx:
                factor = self.A[i][entering_idx]
                if factor != 0:
                    for j in range(len(self.A[i])):
                        self.A[i][j] -= factor * self.A[leaving_idx][j]
                    self.b[i] -= factor * self.b[leaving_idx]
        
        # Khử biến vào khỏi hàm mục tiêu
        factor = self.z[entering_idx]
        if factor != 0:
            for j in range(len(self.z)):
                self.z[j] -= factor * self.A[leaving_idx][j]
        
        # Cập nhật danh sách biến cơ sở và phi cơ sở
        self.basic[leaving_idx] = entering_var
        
        # Cập nhật danh sách nonbasic
        if entering_var in self.nonbasic:
            nonbasic_idx = self.nonbasic.index(entering_var)
            self.nonbasic[nonbasic_idx] = leaving_var

    def solve(self):
        # Kiểm tra nếu có b_i < 0
        if self.has_negative_b:
            return None

        step = 1
        self.steps_output.append(f"### Bảng Simplex ban đầu:")
        self.steps_output.append(self.render_latex())
        
        max_iterations = 100  # Tránh vòng lặp vô hạn
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Kiểm tra điều kiện tối ưu
            if self.problem_type == 'max':
                if all(c >= 0 for c in self.z):
                    break
            else:
                if all(c <= 0 for c in self.z):
                    break
            
            self.steps_output.append(f"\n### Bước {step}:")
            
            # Chọn biến vào cơ sở
            if self.problem_type == 'max':
                entering_idx = None
                min_coeff = 0
                for i, coeff in enumerate(self.z):
                    if coeff < min_coeff:
                        min_coeff = coeff
                        entering_idx = i
            else:
                entering_idx = None
                min_coeff = 0
                for i, coeff in enumerate(self.z):
                    if coeff < min_coeff:
                        min_coeff = coeff
                        entering_idx = i
            
            if entering_idx is None:
                break
                
            all_vars = self.get_all_variables()
            entering_var = all_vars[entering_idx]
            
            self.steps_output.append(f"Chọn biến vào cơ sở: **{entering_var}** (chỉ số {entering_idx}, hệ số {self.z[entering_idx]})")

            # Tính tỷ số cho kiểm tra tỷ số tối thiểu
            ratios = []
            for i in range(self.m):
                a = self.A[i][entering_idx]
                if a > 0:
                    ratio = self.b[i] / a
                    ratios.append((i, ratio, self.basic[i]))
                    self.steps_output.append(f"  Tỷ số tại hàng {i} ({self.basic[i]}): {self.b[i]} / {a} = {ratio}")
            
            if not ratios:
                self.steps_output.append("Bài toán vô hạn (không bị chặn)")
                return None
            
            # Chọn biến ra khỏi cơ sở (tỷ số nhỏ nhất)
            min_ratio_tuple = min(ratios, key=lambda x: x[1])
            leaving_idx = min_ratio_tuple[0]
            leaving_var = self.basic[leaving_idx]

            self.steps_output.append(f"Chọn biến ra khỏi cơ sở: **{leaving_var}** (chỉ số hàng {leaving_idx}, tỷ số nhỏ nhất {min_ratio_tuple[1]})")
            self.steps_output.append(f"\nThực hiện phép xoay với phần tử trục [{leaving_idx}, {entering_idx}] = {self.A[leaving_idx][entering_idx]}")
            
            # Thực hiện phép xoay
            self.pivot(entering_idx, leaving_idx)
            
            self.steps_output.append("\nSau khi xoay:")
            self.steps_output.append(self.render_latex())

            step += 1
        
        if iteration >= max_iterations:
            self.steps_output.append(f"\n### Cảnh báo: Đã đạt giới hạn {max_iterations} vòng lặp!")
                
        self.steps_output.append("\n### Bảng Simplex tối ưu:")
        self.steps_output.append(self.render_latex())
        
        optimal_value = self.calculate_optimal_value()
        optimal_solution = self.get_optimal_solution()
        
        self.steps_output.append(f"\n### Giá trị tối ưu: {optimal_value}")
        self.steps_output.append("### Nghiệm tối ưu:")
        for var, val in optimal_solution.items():
            if var.startswith('x'):
                self.steps_output.append(f"{var} = {val}")
            
        return optimal_value, optimal_solution

    def get_steps_output(self):
        return "\n".join(self.steps_output)

# Đặt độ chính xác cao cho các phép tính thập phân
getcontext().prec = 50

class GeometricMethod:
    def __init__(self, c, A, b, problem_type='max'):
        self.c = np.array(c, dtype=np.float64)
        self.A = np.array(A, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.problem_type = problem_type.lower()
        self.steps_output = []
        self.n = len(c)
        
        # Cải thiện ngưỡng cho các số lớn
        self.tolerance = max(1e-10, np.max(np.abs(self.b)) * 1e-12)
        
        # Chỉ xử lý các bài toán 2 chiều cho phương pháp hình học
        if self.n != 2:
            raise ValueError("Phương pháp hình học chỉ hỗ trợ 2 biến")
        
        # Phát hiện tỷ lệ để xử lý tốt hơn với các số lớn
        self.scale_factor = self._detect_scale()
        self.use_scientific_notation = np.max(np.abs(self.b)) > 1e6

    def _detect_scale(self):
        """Phát hiện tỷ lệ của bài toán để xử lý số học tốt hơn"""
        max_coeff = max(np.max(np.abs(self.A)), np.max(np.abs(self.b)), np.max(np.abs(self.c)))
        if max_coeff > 1e6:
            return max_coeff / 1e6
        return 1.0

    def solve(self):
        self.steps_output.append("### Giải bài toán bằng Phương pháp Hình học Nâng cao")
        self.steps_output.append(f"### Tỷ lệ bài toán: {self._format_number(self.scale_factor)}")
        
        # Kiểm tra trước xem bài toán có khả thi không
        if not self.has_feasible_region():
            self.steps_output.append("### Kết quả: KHÔNG KHẢ THI")
            self.steps_output.append("Hệ thống ràng buộc không có nghiệm khả thi.")
            return None, None, "KHÔNG KHẢ THI"
        
        # Kiểm tra xem bài toán có vô giới không
        unbounded_result = self.check_unbounded()
        if unbounded_result:
            self.steps_output.append("### Kết quả: VÔ GIỚI")
            self.steps_output.append("Bài toán không bị giới hạn theo hướng tối ưu hóa.")
            self.steps_output.append(f"Hướng vô giới: {self._format_point(unbounded_result['direction'])}")
            self.steps_output.append(f"Bắt đầu từ điểm khả thi: {self._format_point(unbounded_result['start_point'])}")
            
            # Tạo hình ảnh minh họa cho trường hợp vô giới
            fig = self.plot_unbounded_case(unbounded_result)
            return fig, None, "VÔ GIỚI"
        
        # Nếu bị giới hạn, tiến hành phân tích điểm góc thông thường
        return self.solve_bounded()

    def _format_number(self, num):
        """Định dạng số phù hợp để hiển thị"""
        if abs(num) < 1e-10:
            return "0"
        elif self.use_scientific_notation and abs(num) >= 1e6:
            return f"{num:.2e}"
        elif abs(num) >= 1000:
            return f"{num:,.0f}"
        else:
            return f"{num:.3f}"

    def _format_point(self, point):
        """Định dạng tọa độ điểm để hiển thị"""
        return f"({self._format_number(point[0])}, {self._format_number(point[1])})"

    def has_feasible_region(self):
        """Kiểm tra tính khả thi nâng cao cho các bài toán quy mô lớn"""
        # Kiểm tra một loạt các điểm cho các bài toán quy mô lớn
        test_points = [(0, 0)]
        
        # Thêm các điểm dựa trên giới hạn của ràng buộc
        for i in range(len(self.A)):
            if self.A[i][0] != 0 and self.b[i] > 0:
                x_intercept = self.b[i] / self.A[i][0]
                if x_intercept > 0:
                    test_points.extend([(x_intercept, 0), (x_intercept/2, 0)])
            
            if self.A[i][1] != 0 and self.b[i] > 0:
                y_intercept = self.b[i] / self.A[i][1]
                if y_intercept > 0:
                    test_points.extend([(0, y_intercept), (0, y_intercept/2)])
        
        # Kiểm tra các điểm chiến lược bổ sung
        max_bound = np.max(self.b[self.b > 0]) if np.any(self.b > 0) else 1000
        test_values = [1, 10, 100, max_bound/4, max_bound/2] if max_bound > 100 else [1, 10]
        
        for val in test_values:
            test_points.extend([(val, 0), (0, val), (val, val)])
        
        for point in test_points:
            if self.is_feasible(point):
                return True
        
        # Kiểm tra kỹ lưỡng hơn bằng các điểm giao nhau có độ chính xác cao
        try:
            corner_points = self.find_all_intersection_points()
            for point in corner_points:
                if point[0] >= -self.tolerance and point[1] >= -self.tolerance and self.is_feasible(point):
                    return True
        except Exception:
            pass
        
        return False

    def check_unbounded(self):
        """Kiểm tra vô giới nâng cao với độ ổn định số học tốt hơn"""
        start_point = self.find_feasible_point()
        if start_point is None:
            return None
        
        c1, c2 = self.c[0], self.c[1]
        
        # Hướng tối ưu hóa
        if self.problem_type == 'max':
            direction = (c1, c2)
        else:
            direction = (-c1, -c2)
        
        # Xử lý trường hợp các hệ số hàm mục tiêu bằng 0
        direction_norm = np.sqrt(direction[0]**2 + direction[1]**2)
        if direction_norm < self.tolerance:
            return None  # Hàm mục tiêu gần như không đổi
        
        unit_direction = (direction[0]/direction_norm, direction[1]/direction_norm)
        
        # Kiểm tra các ràng buộc với độ ổn định số học cải thiện
        max_step = float('inf')
        
        for i in range(len(self.A)):
            a1, a2 = self.A[i][0], self.A[i][1]
            constraint_dot_direction = a1 * unit_direction[0] + a2 * unit_direction[1]
            
            if constraint_dot_direction > self.tolerance:
                current_constraint_value = a1 * start_point[0] + a2 * start_point[1]
                max_step_for_constraint = (self.b[i] - current_constraint_value) / constraint_dot_direction
                max_step = min(max_step, max_step_for_constraint)
        
        # Ngưỡng điều chỉnh cho các bài toán quy mô lớn
        unbounded_threshold = max(1e6, np.max(self.b) * 1000) if np.max(self.b) > 0 else 1e6
        
        if max_step > unbounded_threshold:
            return {
                'direction': unit_direction,
                'start_point': start_point,
                'objective_direction': direction,
                'threshold': unbounded_threshold
            }
        
        return None

    def find_feasible_point(self):
        """Tìm điểm khả thi nâng cao với các phương pháp số học tốt hơn"""
        # Thử các điểm đơn giản trước
        test_points = [(0, 0)]
        
        # Thêm các điểm dựa trên từng ràng buộc riêng lẻ
        for i in range(len(self.A)):
            if self.A[i][0] != 0 and self.b[i] > 0:
                point = (self.b[i] / self.A[i][0], 0)
                if point[0] >= -self.tolerance:
                    test_points.append(point)
            
            if self.A[i][1] != 0 and self.b[i] > 0:
                point = (0, self.b[i] / self.A[i][1])
                if point[1] >= -self.tolerance:
                    test_points.append(point)
        
        for point in test_points:
            if self.is_feasible(point):
                return point
        
        # Thử các điểm góc
        try:
            corner_points = self.find_corner_points()
            if corner_points:
                return corner_points[0]
        except Exception:
            pass
        
        # Giải hệ phương trình nâng cao với điều kiện tốt hơn
        try:
            for i in range(len(self.A)):
                for j in range(i+1, len(self.A)):
                    A_matrix = np.array([self.A[i], self.A[j]])
                    b_vector = np.array([self.b[i], self.b[j]])
                    
                    # Kiểm tra số điều kiện
                    cond_num = np.linalg.cond(A_matrix)
                    if cond_num < 1e12:  # Hệ phương trình có điều kiện tốt
                        try:
                            point = np.linalg.solve(A_matrix, b_vector)
                            if (point[0] >= -self.tolerance and point[1] >= -self.tolerance and 
                                self.is_feasible(point)):
                                return tuple(point)
                        except np.linalg.LinAlgError:
                            continue
        except Exception:
            pass
        
        return None

    def plot_unbounded_case(self, unbounded_info):
        """Vẽ đồ thị nâng cao cho các trường hợp vô giới với tỷ lệ thông minh"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Xác định phạm vi vẽ dựa trên tỷ lệ bài toán
        start = unbounded_info['start_point']
        max_coord = max(abs(start[0]), abs(start[1]), 1)
        
        # Tính toán phạm vi thông minh
        if max_coord > 1e6:
            plot_range = max_coord * 2
            x_range = np.linspace(0, plot_range, 1000)
        else:
            plot_range = max(max_coord * 10, 100)
            x_range = np.linspace(0, plot_range, 1000)
        
        # Ràng buộc không âm
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='x₂ ≥ 0')
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, label='x₁ ≥ 0')
        
        # Vẽ các đường ràng buộc với xử lý cải thiện
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
        for i, (constraint, rhs) in enumerate(zip(self.A, self.b)):
            a, b = constraint
            color = colors[i % len(colors)]
            
            if abs(b) > self.tolerance:
                y_vals = [(rhs - a * x) / b for x in x_range]
                # Lọc các giá trị cực đại để hiển thị tốt hơn
                y_vals = [y if abs(y) < plot_range * 2 else np.nan for y in y_vals]
                ax.plot(x_range, y_vals, color=color, linewidth=2,
                       label=f"{self._format_number(a)}x₁ + {self._format_number(b)}x₂ ≤ {self._format_number(rhs)}")
            else:
                if abs(a) > self.tolerance:
                    x_val = rhs / a
                    if 0 <= x_val <= plot_range:
                        ax.axvline(x=x_val, color=color, linewidth=2,
                                  label=f"{self._format_number(a)}x₁ ≤ {self._format_number(rhs)}")
        
        # Đánh dấu điểm bắt đầu
        ax.scatter(start[0], start[1], color='green', s=100, 
                  label=f"Điểm khả thi: {self._format_point(start)}", zorder=5)
        
        # Hiển thị hướng vô giới bằng mũi tên
        direction = unbounded_info['direction']
        arrow_length = plot_range / 5
        ax.arrow(start[0], start[1], 
                direction[0] * arrow_length, direction[1] * arrow_length,
                head_width=plot_range/50, head_length=plot_range/30, 
                fc='red', ec='red', linewidth=2,
                label=f"Hướng vô giới: {self._format_point(direction)}")
        
        # Hiển thị các điểm dọc theo tia vô giới với giá trị hàm mục tiêu
        step_size = plot_range / 10
        for mult in [1, 2, 3, 4]:
            t = mult * step_size
            point = (start[0] + t * direction[0], start[1] + t * direction[1])
            if point[0] >= -self.tolerance and point[1] >= -self.tolerance:
                obj_val = self.c[0] * point[0] + self.c[1] * point[1]
                ax.scatter(point[0], point[1], color='orange', s=40, alpha=0.8)
                ax.annotate(f'z={self._format_number(obj_val)}', point, 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlim(-plot_range*0.1, plot_range)
        ax.set_ylim(-plot_range*0.1, plot_range)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f"Bài toán {'Tối đa hóa' if self.problem_type == 'max' else 'Tối thiểu hóa'} Vô giới\n"
                    f"Tỷ lệ: {self._format_number(self.scale_factor)}")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig

    def solve_bounded(self):
        """Giải bài toán bị giới hạn nâng cao với số học cải tiến"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Vẽ vùng khả thi
        self.plot_feasible_region(ax)
        
        # Tìm các điểm góc với độ chính xác nâng cao
        corner_points = self.find_corner_points()
        
        if not corner_points:
            self.steps_output.append("### Lỗi: Không tìm thấy điểm góc")
            return fig, None, None
        
        self.steps_output.append("### Các Điểm Góc:")
        for i, point in enumerate(corner_points):
            self.steps_output.append(f"Điểm {i+1}: {self._format_point(point)}")
            ax.scatter(point[0], point[1], color='red', s=100, zorder=5)
            ax.annotate(f"P{i+1}", (point[0], point[1]), xytext=(5, 5), 
                       textcoords='offset points', fontweight='bold', fontsize=10)
        
        # Đánh giá hàm mục tiêu với độ chính xác cao
        self.steps_output.append("\n### Giá trị Hàm Mục tiêu tại Các Điểm Góc:")
        best_point = None
        best_value = float('-inf') if self.problem_type == 'max' else float('inf')
        
        for i, point in enumerate(corner_points):
            # Sử dụng độ chính xác cao cho các số lớn
            if self.use_scientific_notation:
                obj_value = float(Decimal(str(self.c[0])) * Decimal(str(point[0])) + 
                                Decimal(str(self.c[1])) * Decimal(str(point[1])))
            else:
                obj_value = self.c[0] * point[0] + self.c[1] * point[1]
            
            self.steps_output.append(f"Điểm {i+1} {self._format_point(point)}: z = {self._format_number(obj_value)}")
            
            if (self.problem_type == 'max' and obj_value > best_value) or \
               (self.problem_type == 'min' and obj_value < best_value):
                best_value = obj_value
                best_point = point
        
        # Vẽ các đường mức của hàm mục tiêu
        if best_point is not None:
            self.plot_objective_function(ax, best_point, best_value)
            
            # Đánh dấu điểm tối ưu
            ax.scatter(best_point[0], best_point[1], color='green', s=200, 
                      label=f"Tối ưu: {self._format_point(best_point)}", 
                      zorder=6, marker='*')
            
            self.steps_output.append(f"\n### Nghiệm Tối ưu:")
            self.steps_output.append(f"x₁ = {self._format_number(best_point[0])}")
            self.steps_output.append(f"x₂ = {self._format_number(best_point[1])}")
            self.steps_output.append(f"Giá trị tối ưu: {self._format_number(best_value)}")
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig, best_point, best_value

    def plot_feasible_region(self, ax):
        """Vẽ vùng khả thi nâng cao cho các bài toán quy mô lớn"""
        # Xác định giới hạn vẽ phù hợp
        corner_points = self.find_corner_points()
        
        if corner_points:
            max_x = max(point[0] for point in corner_points)
            max_y = max(point[1] for point in corner_points)
            x_max = max_x * 1.2 + 1
            y_max = max_y * 1.2 + 1
        else:
            # Giới hạn dự phòng dựa trên các ràng buộc
            x_max = y_max = 100
            for i in range(len(self.A)):
                if self.A[i][0] > self.tolerance:
                    x_max = max(x_max, self.b[i] / self.A[i][0])
                if self.A[i][1] > self.tolerance:
                    y_max = max(y_max, self.b[i] / self.A[i][1])
            x_max = min(x_max * 1.2, np.max(self.b) * 2) if np.max(self.b) > 0 else 100
            y_max = min(y_max * 1.2, np.max(self.b) * 2) if np.max(self.b) > 0 else 100
        
        # Giới hạn các giá trị cực đại để hiển thị
        x_max = min(x_max, 1e9)
        y_max = min(y_max, 1e9)
        
        # Ràng buộc không âm
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Vẽ các ràng buộc với xử lý số học cải thiện
        x_range = np.linspace(0, x_max, 1000)
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
        
        for i, (constraint, rhs) in enumerate(zip(self.A, self.b)):
            a, b = constraint
            color = colors[i % len(colors)]
            
            if abs(b) > self.tolerance:
                y_vals = []
                for x in x_range:
                    y = (rhs - a * x) / b
                    if -y_max * 0.1 <= y <= y_max * 1.1:  # Giữ các giá trị hợp lý
                        y_vals.append(y)
                    else:
                        y_vals.append(np.nan)
                
                ax.plot(x_range, y_vals, color=color, linewidth=2,
                       label=f"{self._format_number(a)}x₁ + {self._format_number(b)}x₂ ≤ {self._format_number(rhs)}")
            else:
                if abs(a) > self.tolerance:
                    x_val = rhs / a
                    if 0 <= x_val <= x_max:
                        ax.axvline(x=x_val, color=color, linewidth=2,
                                  label=f"{self._format_number(a)}x₁ ≤ {self._format_number(rhs)}")
        
        # Tô màu vùng khả thi nếu có thể
        try:
            if len(corner_points) >= 3:
                # Sắp xếp các điểm để tạo đa giác đúng
                center = np.mean(corner_points, axis=0)
                angles = [np.arctan2(p[1] - center[1], p[0] - center[0]) for p in corner_points]
                sorted_points = [p for _, p in sorted(zip(angles, corner_points))]
                
                # Lọc các điểm cực đại để hiển thị
                filtered_points = [(min(p[0], x_max), min(p[1], y_max)) for p in sorted_points]
                
                polygon = Polygon(filtered_points, alpha=0.2, color='lightblue', 
                                label='Vùng Khả thi')
                ax.add_patch(polygon)
        except Exception:
            pass  # Bỏ qua việc tô màu nếu thất bại
        
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f"Bài toán {'Tối đa hóa' if self.problem_type == 'max' else 'Tối thiểu hóa'}\n"
                    f"Tỷ lệ: {self._format_number(self.scale_factor)}")
        ax.grid(True, alpha=0.3)

    def plot_objective_function(self, ax, best_point, best_value):
        """Vẽ hàm mục tiêu nâng cao"""
        if abs(self.c[1]) > self.tolerance:
            x_range = np.linspace(0, ax.get_xlim()[1], 100)
            
            # Vẽ đường mức tối ưu
            y_optimal = []
            for x in x_range:
                y = (best_value - self.c[0] * x) / self.c[1]
                if ax.get_ylim()[0] <= y <= ax.get_ylim()[1]:
                    y_optimal.append(y)
                else:
                    y_optimal.append(np.nan)
            
            ax.plot(x_range, y_optimal, 'g--', linewidth=3,
                   label=f"Tối ưu: {self._format_number(self.c[0])}x₁ + {self._format_number(self.c[1])}x₂ = {self._format_number(best_value)}")
            
            # Vẽ các đường mức với khoảng cách thích nghi
            value_range = max(abs(best_value) * 0.1, 1)
            offsets = [-2 * value_range, -value_range, value_range, 2 * value_range]
            
            for offset in offsets:
                level_value = best_value + offset * (1 if self.problem_type == 'max' else -1)
                y_level = []
                for x in x_range:
                    y = (level_value - self.c[0] * x) / self.c[1]
                    if ax.get_ylim()[0] <= y <= ax.get_ylim()[1]:
                        y_level.append(y)
                    else:
                        y_level.append(np.nan)
                ax.plot(x_range, y_level, '--', alpha=0.4, color='gray')
            
            # Mũi tên hướng tối ưu với tỷ lệ phù hợp
            mid_x, mid_y = ax.get_xlim()[1]/3, ax.get_ylim()[1]/3
            direction = (self.c[0], self.c[1]) if self.problem_type == 'max' else (-self.c[0], -self.c[1])
            
            # Chuẩn hóa mũi tên để hiển thị tốt hơn
            arrow_scale = min(ax.get_xlim()[1], ax.get_ylim()[1]) / 10
            norm = np.sqrt(direction[0]**2 + direction[1]**2)
            if norm > self.tolerance:
                direction = (direction[0] * arrow_scale / norm, direction[1] * arrow_scale / norm)
                ax.quiver(mid_x, mid_y, direction[0], direction[1], 
                         angles='xy', scale_units='xy', scale=1, color='blue', width=0.005,
                         label=f"Hướng tối ưu hóa")

    def find_corner_points(self):
        """Tìm điểm góc nâng cao với độ ổn định số học tốt hơn"""
        corner_points = []
        
        # Kiểm tra gốc tọa độ
        origin = (0, 0)
        if self.is_feasible(origin):
            corner_points.append(origin)
        
        # Giao điểm với các trục - xử lý số học cải thiện
        for i in range(len(self.A)):
            if abs(self.A[i][0]) > self.tolerance:
                x_val = self.b[i] / self.A[i][0]
                point = (x_val, 0)
                if x_val >= -self.tolerance and self.is_feasible(point):
                    point = (max(0, x_val), 0)  # Đảm bảo không âm
                    if not self.point_in_list(point, corner_points):
                        corner_points.append(point)
            
            if abs(self.A[i][1]) > self.tolerance:
                y_val = self.b[i] / self.A[i][1]
                point = (0, y_val)
                if y_val >= -self.tolerance and self.is_feasible(point):
                    point = (0, max(0, y_val))  # Đảm bảo không âm
                    if not self.point_in_list(point, corner_points):
                        corner_points.append(point)
        
        # Giao điểm của các đường ràng buộc - phương pháp số học nâng cao
        for i in range(len(self.A)):
            for j in range(i+1, len(self.A)):
                point = self.line_intersection_robust(self.A[i], self.b[i], self.A[j], self.b[j])
                if (point and point[0] >= -self.tolerance and point[1] >= -self.tolerance and 
                    self.is_feasible(point) and not self.point_in_list(point, corner_points)):
                    # Đảm bảo tọa độ không âm
                    point = (max(0, point[0]), max(0, point[1]))
                    corner_points.append(point)
        
        return corner_points

    def line_intersection_robust(self, a1, b1, a2, b2):
        """Tìm giao điểm đường thẳng mạnh mẽ với xử lý số học tốt hơn"""
        A = np.array([a1, a2], dtype=np.float64)
        b = np.array([b1, b2], dtype=np.float64)
        
        try:
            # Kiểm tra số điều kiện
            cond_num = np.linalg.cond(A)
            if cond_num > 1e12:  # Điều kiện kém
                return None
            
            x = np.linalg.solve(A, b)
            
            # Xác minh nghiệm
            residual = np.linalg.norm(A @ x - b)
            if residual > self.tolerance * 100:
                return None
            
            return tuple(x)
        except np.linalg.LinAlgError:
            return None

    def find_all_intersection_points(self):
        """Tìm tất cả các điểm giao nhau với các phương pháp số học mạnh mẽ"""
        points = []
        
        for i in range(len(self.A)):
            for j in range(i+1, len(self.A)):
                point = self.line_intersection_robust(self.A[i], self.b[i], self.A[j], self.b[j])
                if point:
                    points.append(point)
        
        return points

    def is_feasible(self, point):
        """Kiểm tra tính khả thi nâng cao với ngưỡng phù hợp"""
        for i in range(len(self.A)):
            constraint_value = sum(self.A[i][j] * point[j] for j in range(len(point)))
            if constraint_value > self.b[i] + self.tolerance:
                return False
        return True

    def point_in_list(self, point, point_list):
        """So sánh điểm nâng cao với ngưỡng thích nghi"""
        tolerance = max(self.tolerance, np.max(np.abs(point)) * 1e-12)
        for existing_point in point_list:
            if (abs(point[0] - existing_point[0]) < tolerance and 
                abs(point[1] - existing_point[1]) < tolerance):
                return True
        return False

    def get_steps_output(self):
        return "\n".join(self.steps_output)


class BlandRule(SimplexDictionary):
    def __init__(self, c, A, b, problem_type='max'):
        super().__init__(c, A, b, problem_type)
        self.EPS = Fraction(1, 1000000000)  # 1e-9 dạng Fraction

    def get_variable_index(self, var_name):
        """Lấy chỉ số theo Luật Bland để sắp xếp biến"""
        if var_name.startswith('x'):
            return int(var_name[1:])
        elif var_name.startswith('w'):
            return 1000 + int(var_name[1:])
        elif var_name.startswith('s'):
            return 1000 + int(var_name[1:])
        return 9999

    def pivot(self, entering_idx, leaving_idx_row):
        """Thực hiện thao tác xoay với quản lý danh sách biến phù hợp"""
        pivot_element = self.A[leaving_idx_row][entering_idx]
        if abs(pivot_element) < self.EPS:
            raise ValueError("Phần tử xoay (gần bằng) 0")

        entering_var_name = self.vars_and_slack()[entering_idx]
        leaving_var_name = self.basic[leaving_idx_row]

        # Chuẩn hóa hàng xoay
        for j in range(len(self.A[leaving_idx_row])):
            self.A[leaving_idx_row][j] /= pivot_element
        self.b[leaving_idx_row] /= pivot_element

        # Khử biến vào khỏi các hàng ràng buộc khác
        for i in range(self.m):
            if i != leaving_idx_row:
                factor = self.A[i][entering_idx]
                if factor != 0:
                    for j in range(len(self.A[i])):
                        self.A[i][j] -= factor * self.A[leaving_idx_row][j]
                    self.b[i] -= factor * self.b[leaving_idx_row]

        # Khử biến vào khỏi hàm mục tiêu
        factor = self.z[entering_idx]
        if factor != 0:
            for j in range(len(self.z)):
                self.z[j] -= factor * self.A[leaving_idx_row][j]
        
        # Cập nhật danh sách biến cơ bản và không cơ bản
        self.basic[leaving_idx_row] = entering_var_name
        
        # Tìm và thay thế trong danh sách nonbasic
        if entering_var_name in self.nonbasic:
            nonbasic_idx = self.nonbasic.index(entering_var_name)
            self.nonbasic[nonbasic_idx] = leaving_var_name
            self.nonbasic.sort(key=lambda v: self.get_variable_index(v))
        else:
            # Nếu không tìm thấy, thêm vào danh sách nonbasic
            self.nonbasic.append(leaving_var_name)
            self.nonbasic.sort(key=lambda v: self.get_variable_index(v))

    def check_initial_feasibility(self):
        """Kiểm tra tính khả thi ban đầu"""
        for i, bi in enumerate(self.b):
            if bi < -self.EPS:
                self.steps_output.append(f"Ràng buộc {i+1} có b[{i}] = {bi} < 0, không thể sử dụng thuật toán đơn hình tiêu chuẩn.")
                self.steps_output.append("Vui lòng sử dụng thuật toán hai pha hoặc đơn hình đối ngẫu.")
                return False
        return True

    def solve(self):
        """Giải bài toán bằng Luật Bland chống lặp chu kỳ"""
        # Kiểm tra nếu có b_i < 0
        if self.has_negative_b:
            return None, None
            
        step = 1
        self.steps_output.append(f"### Từ điển Khởi tạo (Bland Rule):")
        self.steps_output.append(self.render_latex())

        max_iterations = 100  # Tránh vòng lặp vô hạn
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self.steps_output.append(f"\n### Bước {step}:")

            # Tìm ứng viên biến vào (tuân theo hướng tối ưu hóa)
            entering_candidates = []
            for i, coeff in enumerate(self.z):
                var_name = self.vars_and_slack()[i]
                if var_name in self.nonbasic:
                    if (self.problem_type == 'max' and coeff < -self.EPS) or \
                        (self.problem_type == 'min' and coeff < -self.EPS):  # Cả hai đều là < -EPS
                        var_idx_bland = self.get_variable_index(var_name)
                        entering_candidates.append({
                            'orig_idx': i, 
                            'bland_idx': var_idx_bland, 
                            'name': var_name, 
                            'coeff': coeff
                        })
            
            if not entering_candidates:
                # Kiểm tra điều kiện tối ưu đã đạt
                break  # Nghiệm tối ưu
            
            # Chọn biến vào theo luật Bland (chỉ số nhỏ nhất)
            entering_candidates.sort(key=lambda x: x['bland_idx'])
            entering_info = entering_candidates[0]
            entering_idx = entering_info['orig_idx']
            entering_var = entering_info['name']
            
            self.steps_output.append(f"Chọn biến vào (Luật Bland): **{entering_var}** (chỉ số {entering_idx}, hệ số {self.z[entering_idx]})")

            # Tính tỷ số tối thiểu
            ratios = []
            for i in range(self.m):
                a_val = self.A[i][entering_idx]
                if a_val > self.EPS:
                    ratio = self.b[i] / a_val
                    if ratio >= -self.EPS:  # Cho phép b_i = 0
                        basic_var_in_row = self.basic[i]
                        var_idx_bland = self.get_variable_index(basic_var_in_row)
                        ratios.append({
                            'row_idx': i, 
                            'ratio': ratio, 
                            'name': basic_var_in_row, 
                            'bland_idx': var_idx_bland
                        })
                        self.steps_output.append(f"  Tỷ số tại hàng {i} ({basic_var_in_row}): {self.b[i]} / {a_val} = {ratio}")
            
            if not ratios:
                self.steps_output.append("Bài toán không giới nội (unbounded)")
                return None, None
            
            # Tìm tỷ số tối thiểu
            min_ratio_val = min(r['ratio'] for r in ratios)
            leaving_candidates = [r for r in ratios if abs(r['ratio'] - min_ratio_val) < self.EPS]
            
            # Phá thế bế tắc bằng luật Bland (chọn chỉ số nhỏ nhất)
            leaving_candidates.sort(key=lambda x: x['bland_idx'])
            leaving_info = leaving_candidates[0]
            leaving_idx_row = leaving_info['row_idx']
            leaving_var = leaving_info['name']
            
            if len(leaving_candidates) > 1:
                self.steps_output.append(f"Thế bế tắc tại tỷ số tối thiểu {min_ratio_val}. Các ứng viên: {[cand['name'] for cand in leaving_candidates]}")
                self.steps_output.append(f"Luật Bland: chọn chỉ số nhỏ nhất: {leaving_var}")
            
            self.steps_output.append(f"Chọn biến ra: **{leaving_var}** (chỉ số hàng {leaving_idx_row}, tỷ số tối thiểu {min_ratio_val})")
            self.steps_output.append(f"\nThực hiện xoay với phần tử A[{leaving_idx_row}, {entering_idx}] = {self.A[leaving_idx_row][entering_idx]}")
            
            self.pivot(entering_idx, leaving_idx_row)
            
            self.steps_output.append("\nSau khi xoay:")
            self.steps_output.append(self.render_latex())
            step += 1

        if iteration >= max_iterations:
            self.steps_output.append(f"\n### Cảnh báo: Đã đạt giới hạn {max_iterations} vòng lặp!")

        # Kiểm tra vô số nghiệm
        has_multiple_solutions = False
        for i, coeff in enumerate(self.z):
            var_name = self.vars_and_slack()[i]
            if var_name in self.nonbasic and abs(coeff) < self.EPS:
                has_multiple_solutions = True
                break

        self.steps_output.append("\n### Từ vựng Tối ưu:")
        self.steps_output.append(self.render_latex())
        
        optimal_value = self.calculate_optimal_value()
        optimal_solution = self.get_optimal_solution()
        
        if has_multiple_solutions:
            self.steps_output.append("Bài toán có vô số nghiệm")
        
        self.steps_output.append(f"\n### Giá trị Tối ưu: {optimal_value}")
        self.steps_output.append("### Nghiệm Tối ưu:")
        
        decision_vars_sorted = sorted(
            [var for var in optimal_solution.keys() if var.startswith('x')],
            key=lambda v: self.get_variable_index(v)
        )
        for var_name in decision_vars_sorted:
            self.steps_output.append(f"{var_name} = {optimal_solution[var_name]}")
            
        return optimal_value, optimal_solution

class TwoPhaseMethod:
    def __init__(self, c, A, b, problem_type='max', constraint_types=None):
        self.original_c = [Fraction(ci) for ci in c]
        self.original_A = [[Fraction(aij) for aij in row] for row in A]
        self.original_b = [Fraction(bi) for bi in b]
        self.problem_type = problem_type.lower()
        self.num_original_vars = len(c)
        self.num_constraints = len(b)

        if constraint_types is None:
            self.constraint_types = ['<='] * self.num_constraints
        else:
            self.constraint_types = constraint_types

        self.steps_output = []
        self.artificial_vars = []
        self.slack_vars = []
        self.surplus_vars = []
        self.has_artificial = False

    def _ensure_positive_rhs(self, A, b, types):
        """Bước 1a: Đảm bảo tất cả các b_i đều không âm."""
        self.steps_output.append("**Bước 1a: Đảm bảo tất cả các giá trị bên phải (RHS) đều không âm**")
        
        new_A = [[x for x in row] for row in A]
        new_b = list(b)
        new_types = list(types)
        
        changes_made = False
        for i in range(len(new_b)):
            if new_b[i] < 0:
                self.steps_output.append(f"  Ràng buộc {i+1}: b[{i}] = {b[i]} < 0, nhân ràng buộc với -1")
                new_b[i] *= -1
                new_A[i] = [-x for x in new_A[i]]
                if new_types[i] == '<=':
                    new_types[i] = '>='
                elif new_types[i] == '>=':
                    new_types[i] = '<='
                # '=' giữ nguyên
                changes_made = True
                constraint_str = " + ".join([f"{new_A[i][j]}x{j+1}" if new_A[i][j] > 0 else f"{new_A[i][j]}x{j+1}" 
                                           for j in range(len(new_A[i])) if new_A[i][j] != 0])
                self.steps_output.append(f"    Ràng buộc mới: {constraint_str} {new_types[i]} {new_b[i]}")
        
        if not changes_made:
            self.steps_output.append("  Tất cả các giá trị RHS đã không âm.")
            
        return new_A, new_b, new_types

    def _convert_to_standard_form(self, A, b, types):
        """Bước 1b-2: Chuyển về dạng chuẩn và xác định các biến nhân tạo cần thiết."""
        self.steps_output.append("\n**Bước 1b: Chuyển các ràng buộc về dạng phương trình**")
        
        standard_A = []
        standard_b = list(b)
        artificial_needed = []
        
        # Theo dõi số lượng biến
        current_var_count = self.num_original_vars
        slack_count = 0
        surplus_count = 0
        artificial_count = 0
        
        for i in range(self.num_constraints):
            constraint_row = list(A[i])
            
            if types[i] == '<=':
                # Thêm biến chùng
                for j in range(len(standard_A)):
                    standard_A[j].append(Fraction(0))
                constraint_row.append(Fraction(1))
                self.slack_vars.append(current_var_count)
                current_var_count += 1
                slack_count += 1
                artificial_needed.append(False)
                self.steps_output.append(f"  Ràng buộc {i+1} (<=): Thêm biến chùng s{slack_count}")
                
            elif types[i] == '>=':
                # Thêm biến dư (trừ ở phía bên trái)
                for j in range(len(standard_A)):
                    standard_A[j].append(Fraction(0))
                constraint_row.append(Fraction(-1))
                self.surplus_vars.append(current_var_count)
                current_var_count += 1
                surplus_count += 1
                artificial_needed.append(True)
                self.steps_output.append(f"  Ràng buộc {i+1} (>=): Thêm biến dư e{surplus_count}, cần biến nhân tạo")
                
            else:  # '='
                artificial_needed.append(True)
                self.steps_output.append(f"  Ràng buộc {i+1} (=): Cần biến nhân tạo")
            
            standard_A.append(constraint_row)
        
        # Thêm các biến nhân tạo khi cần
        self.steps_output.append("\n**Bước 2: Thêm các biến nhân tạo**")
        for i in range(self.num_constraints):
            if artificial_needed[i]:
                # Thêm cột biến nhân tạo
                for j in range(len(standard_A)):
                    if j == i:
                        standard_A[j].append(Fraction(1))
                    else:
                        standard_A[j].append(Fraction(0))
                
                self.artificial_vars.append(current_var_count)
                current_var_count += 1
                artificial_count += 1
                self.has_artificial = True
                self.steps_output.append(f"  Thêm biến nhân tạo a{artificial_count} cho ràng buộc {i+1}")
        
        # Đảm bảo tất cả các hàng có cùng số biến
        max_vars = max(len(row) for row in standard_A) if standard_A else self.num_original_vars
        for row in standard_A:
            while len(row) < max_vars:
                row.append(Fraction(0))
        
        self.steps_output.append(f"  Tổng số biến: {max_vars} (gốc: {self.num_original_vars}, chùng: {slack_count}, dư: {surplus_count}, nhân tạo: {artificial_count})")
        
        return standard_A, standard_b, max_vars

    def _create_phase1_objective(self, total_vars):
        """Bước 3: Tạo hàm mục tiêu Giai đoạn 1 để tối thiểu hóa tổng các biến nhân tạo."""
        self.steps_output.append("\n**Bước 3: Tạo hàm mục tiêu Giai đoạn 1**")
        
        phase1_c = [Fraction(0)] * total_vars
        for art_var in self.artificial_vars:
            phase1_c[art_var] = Fraction(1)
        
        self.steps_output.append(f"  Hàm mục tiêu Giai đoạn 1: tối thiểu w = " + 
                               " + ".join([f"a{i+1}" for i in range(len(self.artificial_vars))]))
        self.steps_output.append(f"  Vector hệ số: {phase1_c}")
        
        return phase1_c

    def _solve_phase1(self, phase1_c, standard_A, standard_b):
        """Bước 4-5: Giải bài toán Giai đoạn 1 bằng SimplexDictionary."""
        self.steps_output.append("\n**Bước 4-5: Giải bài toán Giai đoạn 1**")
        
        try:
            # Chuyển về định dạng mà SimplexDictionary mong đợi (tất cả là ràng buộc <=)
            # Vì đã ở dạng chuẩn (Ax = b), cần chuyển lại thành <= cho SimplexDictionary
            phase1_A_ineq = []
            phase1_b_ineq = []
            
            for i in range(len(standard_A)):
                # Chuyển phương trình về ràng buộc <= cho SimplexDictionary
                # Sử dụng các biến nhân tạo/chùng làm cơ sở
                row = list(standard_A[i])
                
                # Tìm biến cơ bản cho ràng buộc này (chùng hoặc nhân tạo)
                basic_var_idx = -1
                if i < len(self.slack_vars) and self.slack_vars and len(self.slack_vars) > i:
                    # Ràng buộc này ban đầu là <= và có biến chùng
                    basic_var_idx = self.slack_vars[i] if i < len(self.slack_vars) else -1
                
                if basic_var_idx == -1 and self.artificial_vars:
                    # Tìm biến nhân tạo trong hàng này
                    for art_idx in self.artificial_vars:
                        if art_idx < len(row) and row[art_idx] == Fraction(1):
                            basic_var_idx = art_idx
                            break
                
                # Tạo ràng buộc cho SimplexDictionary (mong đợi dạng <=)
                phase1_A_ineq.append(row[:self.num_original_vars])  # Chỉ các biến gốc cho bất đẳng thức
                phase1_b_ineq.append(standard_b[i])
            
            # Tạo SimplexDictionary cho Giai đoạn 1
            phase1_solver = SimplexDictionary(
                phase1_c[:self.num_original_vars],  # Chỉ các biến gốc trong hàm mục tiêu
                phase1_A_ineq, 
                phase1_b_ineq, 
                problem_type='min'
            )
            
            self.steps_output.append("**Từ điển Khởi tạo Giai đoạn 1:**")
            self.steps_output.append(phase1_solver.render_latex())
            
            # Giải Giai đoạn 1
            phase1_optimal_value, phase1_solution = phase1_solver.solve()
            
            # Lấy các bước từ giải pháp Giai đoạn 1
            phase1_steps = phase1_solver.get_steps_output()
            self.steps_output.append("\n**Các Bước Giải Giai đoạn 1:**")
            self.steps_output.append(phase1_steps)
            
            return phase1_optimal_value, phase1_solution, phase1_solver
            
        except Exception as e:
            self.steps_output.append(f"Lỗi trong Giai đoạn 1: {str(e)}")
            return None, None, None

    def _analyze_phase1_result(self, phase1_optimal_value, phase1_solution):
        """Bước 6: Phân tích kết quả Giai đoạn 1."""
        self.steps_output.append(f"\n**Bước 6: Phân tích Kết quả Giai đoạn 1**")
        self.steps_output.append(f"Giá trị tối ưu Giai đoạn 1 w* = {phase1_optimal_value}")
        
        if phase1_optimal_value is None:
            self.steps_output.append("Giai đoạn 1 không tìm được nghiệm")
            return False
        
        tolerance = Fraction(1, 1000000)
        if phase1_optimal_value > tolerance:
            self.steps_output.append("w* > 0: Bài toán gốc không có nghiệm khả thi (không khả thi)")
            return False
        else:
            self.steps_output.append("w* = 0: Tìm thấy nghiệm khả thi cho bài toán gốc")
            self.steps_output.append("Tất cả các biến nhân tạo có giá trị 0 trong nghiệm tối ưu")
            return True

    def _setup_phase2(self, phase1_solver, phase1_solution):
        """Bước 7: Thiết lập Giai đoạn 2 với hàm mục tiêu gốc."""
        self.steps_output.append("\n**Bước 7: Thiết lập Giai đoạn 2 với hàm mục tiêu gốc**")
        
        try:
            # Chuyển bài toán gốc sang dạng chuẩn cho Giai đoạn 2
            # Sử dụng cùng cấu trúc ràng buộc nhưng với hàm mục tiêu gốc
            phase2_A = []
            phase2_b = list(self.original_b)
            
            # Áp dụng các biến đổi tương tự Bước 1a
            for i in range(self.num_constraints):
                if self.constraint_types[i] == '<=':
                    phase2_A.append(list(self.original_A[i]))
                elif self.constraint_types[i] == '>=':
                    # Chuyển >= thành <= bằng cách nhân với -1
                    phase2_A.append([-x for x in self.original_A[i]])
                    phase2_b[i] = -phase2_b[i]
                else:  # '='
                    # Đối với đẳng thức, có thể xử lý như <= cho SimplexDictionary
                    # Đây là một cách đơn giản hóa - trong thực tế, ràng buộc đẳng thức cần xử lý đặc biệt
                    phase2_A.append(list(self.original_A[i]))
            
            # Tạo bộ giải Giai đoạn 2 với hàm mục tiêu gốc
            phase2_solver = SimplexDictionary(
                self.original_c, 
                phase2_A, 
                phase2_b, 
                problem_type=self.problem_type
            )
            
            self.steps_output.append("**Từ điển Khởi tạo Giai đoạn 2 (với hàm mục tiêu gốc):**")
            self.steps_output.append(phase2_solver.render_latex())
            
            return phase2_solver
            
        except Exception as e:
            self.steps_output.append(f"Lỗi khi thiết lập Giai đoạn 2: {str(e)}")
            return None

    def _solve_phase2(self, phase2_solver):
        """Bước 8: Giải Giai đoạn 2 với hàm mục tiêu gốc."""
        self.steps_output.append("\n**Bước 8: Giải bài toán Giai đoạn 2**")
        
        try:
            # Giải Giai đoạn 2
            optimal_value, solution = phase2_solver.solve()
            
            # Lấy các bước giải
            phase2_steps = phase2_solver.get_steps_output()
            self.steps_output.append("\n**Các Bước Giải Giai đoạn 2:**")
            self.steps_output.append(phase2_steps)
            
            return optimal_value, solution
            
        except Exception as e:
            self.steps_output.append(f"Lỗi trong Giai đoạn 2: {str(e)}")
            return None, None

    def solve(self):
        """Phương thức giải chính thực hiện thuật toán hai giai đoạn."""
        self.steps_output.append("# Phương pháp Simplex Hai Giai đoạn")
        self.steps_output.append("\n## GIAI ĐOẠN 1: Tìm Nghiệm Khả thi Ban đầu")
        
        try:
            # Bước 1a: Đảm bảo RHS dương
            current_A, current_b, current_types = self._ensure_positive_rhs(
                self.original_A, self.original_b, self.constraint_types
            )
            
            # Kiểm tra xem có cần biến nhân tạo không
            needs_artificial = any(ct in ['>=', '='] for ct in current_types)
            
            if not needs_artificial:
                self.steps_output.append("\n**Không cần biến nhân tạo - giải trực tiếp**")
                return self._solve_directly(current_A, current_b, current_types)
            
            # Bước 1b-2: Chuyển về dạng chuẩn
            standard_A, standard_b, total_vars = self._convert_to_standard_form(current_A, current_b, current_types)
            
            # Bước 3: Tạo hàm mục tiêu Giai đoạn 1
            phase1_c = self._create_phase1_objective(total_vars)
            
            # Bước 4-5: Giải Giai đoạn 1
            phase1_optimal_value, phase1_solution, phase1_solver = self._solve_phase1(
                phase1_c, standard_A, standard_b
            )
            
            # Bước 6: Phân tích kết quả Giai đoạn 1
            if not self._analyze_phase1_result(phase1_optimal_value, phase1_solution):
                return None, None
            
            # Giai đoạn 2
            self.steps_output.append("\n## GIAI ĐOẠN 2: Giải Bài toán Gốc")
            
            # Bước 7: Thiết lập Giai đoạn 2
            phase2_solver = self._setup_phase2(phase1_solver, phase1_solution)
            if phase2_solver is None:
                return None, None
            
            # Bước 8: Giải Giai đoạn 2
            optimal_value, solution = self._solve_phase2(phase2_solver)
            
            # Bước 9: Kết quả cuối cùng
            if optimal_value is not None:
                self.steps_output.append(f"\n## Bước 9: Kết quả Cuối cùng")
                self.steps_output.append(f"**Giá trị Tối ưu:** {optimal_value}")
                self.steps_output.append("**Nghiệm Tối ưu:**")
                for var, val in solution.items():
                    if var.startswith('x'):
                        self.steps_output.append(f"  {var} = {val}")
            
            return optimal_value, solution
                
        except Exception as e:
            self.steps_output.append(f"Lỗi trong phương pháp hai giai đoạn: {str(e)}")
            return None, None

    def _solve_directly(self, A, b, types):
        """Giải trực tiếp bằng SimplexDictionary khi không cần biến nhân tạo."""
        self.steps_output.append("\n## Giải Trực tiếp (Không cần biến nhân tạo)")
        
        try:
            # Chuyển tất cả các ràng buộc thành <= cho SimplexDictionary
            std_A = []
            std_b = []
            
            for i in range(self.num_constraints):
                if types[i] == '<=':
                    std_A.append(A[i])
                    std_b.append(b[i])
                elif types[i] == '>=':
                    # Chuyển >= thành <= bằng cách nhân với -1
                    std_A.append([-x for x in A[i]])
                    std_b.append(-b[i])
                else:  # '=' - xử lý như <= cho SimplexDictionary
                    std_A.append(A[i])
                    std_b.append(b[i])
            
            solver = SimplexDictionary(self.original_c, std_A, std_b, problem_type=self.problem_type)
            
            self.steps_output.append("**Từ điển Giải Trực tiếp:**")
            self.steps_output.append(solver.render_latex())
            
            optimal_value, solution = solver.solve()
            
            # Lấy các bước giải
            solution_steps = solver.get_steps_output()
            self.steps_output.append("\n**Các Bước Giải:**")
            self.steps_output.append(solution_steps)
            
            if optimal_value is not None:
                self.steps_output.append(f"\n## Kết quả Cuối cùng")
                self.steps_output.append(f"**Giá trị Tối ưu:** {optimal_value}")
                self.steps_output.append("**Nghiệm Tối ưu:**")
                for var, val in solution.items():
                    if var.startswith('x'):
                        self.steps_output.append(f"  {var} = {val}")
            
            return optimal_value, solution
            
        except Exception as e:
            self.steps_output.append(f"Lỗi trong giải trực tiếp: {str(e)}")
            return None, None

    def get_steps_output(self):
        """Trả về quá trình giải từng bước."""
        return "\n".join(self.steps_output)

class DualSimplexSolver:
    def __init__(self, c, A, b, problem_type='min'):
        """
        Khởi tạo bộ giải Simplex Đối Ngẫu.

        Parameters:
        - c: Danh sách hệ số của hàm mục tiêu.
        - A: Ma trận ràng buộc.
        - b: Danh sách giá trị bên phải của ràng buộc.
        - problem_type: 'min' hoặc 'max'.
        """
        self.original_c = [Fraction(ci) for ci in c]
        self.original_A = [[Fraction(aij) for aij in row] for row in A]
        self.original_b = [Fraction(bi) for bi in b]
        self.problem_type = problem_type.lower()
        self.num_original_vars = len(c)
        self.num_constraints = len(b)
        self.steps_output = []
        self.slack_vars = []
        self.has_negative_b = False

    def _check_initial_feasibility(self):
        """Kiểm tra tính khả thi ban đầu và quyết định phương pháp giải"""
        self.steps_output.append("**Kiểm tra điều kiện ban đầu:**")
        
        negative_b_indices = []
        for i, bi in enumerate(self.original_b):
            if bi < 0:
                negative_b_indices.append(i)
        
        if negative_b_indices:
            self.steps_output.append(f"  Phát hiện b_i < 0 tại các chỉ số: {negative_b_indices}")
            self.steps_output.append("  → Sử dụng **Thuật toán Dual Simplex**")
            self.has_negative_b = True
            return "dual_simplex"
        else:
            self.steps_output.append("  Tất cả b_i >= 0")
            self.steps_output.append("  → Chuyển sang **Thuật toán Bland (Primal Simplex)**")
            return "bland_rule"

    def _solve_with_bland_rule(self):
        """Giải bài toán bằng thuật toán Bland khi tất cả b_i >= 0"""
        self.steps_output.append("\n" + "="*50)
        self.steps_output.append("CHUYỂN SANG THUẬT TOÁN BLAND")
        self.steps_output.append("="*50)
        
        # Tạo đối tượng BlandRule
        bland_solver = BlandRule(
            c=self.original_c,
            A=self.original_A, 
            b=self.original_b,
            problem_type=self.problem_type
        )
        
        # Giải bằng Bland Rule
        optimal_value, optimal_solution = bland_solver.solve()
        
        # Thêm output từ Bland Rule vào steps_output
        self.steps_output.extend(bland_solver.get_steps_output().split('\n'))
        
        return optimal_value, optimal_solution

    def _ensure_standard_form(self):
        """Bước 1: Đảm bảo bài toán ở dạng chuẩn (min, Ax <= b, x >= 0, b >= 0)."""
        self.steps_output.append("\n**Bước 1: Đảm bảo bài toán ở dạng chuẩn**")
        
        new_A = [[x for x in row] for row in self.original_A]
        new_b = list(self.original_b)
        
        changes_made = False
        for i in range(self.num_constraints):
            if new_b[i] < 0:
                self.steps_output.append(f"  Ràng buộc {i+1}: b[{i}] = {new_b[i]} < 0, nhân ràng buộc với -1")
                new_b[i] *= -1
                new_A[i] = [-x for x in new_A[i]]
                changes_made = True
                constraint_str = " + ".join([f"{new_A[i][j]}x{j+1}" if new_A[i][j] > 0 else f"{new_A[i][j]}x{j+1}" 
                                           for j in range(len(new_A[i])) if new_A[i][j] != 0])
                self.steps_output.append(f"    Ràng buộc mới: {constraint_str} <= {new_b[i]}")
        
        if not changes_made:
            self.steps_output.append("  Tất cả các giá trị RHS đã không âm.")
        
        return new_A, new_b

    def _build_initial_tableau(self, A, b):
        """Bước 2: Xây dựng bảng đơn hình ban đầu với các biến chùng."""
        self.steps_output.append("\n**Bước 2: Xây dựng bảng đơn hình ban đầu**")
        
        # Thêm các biến chùng
        identity = np.eye(self.num_constraints, dtype=object)
        for i in range(self.num_constraints):
            identity[i] = [Fraction(x) for x in identity[i]]
            self.slack_vars.append(self.num_original_vars + i)
        tableau = np.hstack((A, identity))
        tableau = np.vstack((tableau, [-ci for ci in self.original_c] + [0] * self.num_constraints))
        tableau = np.hstack((tableau, np.vstack((np.array(b).reshape(-1, 1), [[0]]))))
        
        self.tableau = tableau
        self.basic_vars = list(range(self.num_original_vars, self.num_original_vars + self.num_constraints))
        self.nonbasic_vars = list(range(self.num_original_vars))
        
        self.steps_output.append(f"  Thêm {self.num_constraints} biến chùng: " + 
                               ", ".join([f"s{i+1} (x{j+1})" for i, j in enumerate(self.slack_vars)]))
        self.steps_output.append("  Bảng đơn hình ban đầu:")
        self.steps_output.append(self._render_tableau())

    def _check_optimality(self):
        """Bước 3: Kiểm tra điều kiện dừng (tất cả b_i >= 0)."""
        self.steps_output.append("\n**Bước 3: Kiểm tra điều kiện dừng**")
        
        rhs = self.tableau[:-1, -1]
        all_non_negative = all(bi >= 0 for bi in rhs)
        self.steps_output.append(f"  Giá trị RHS: {[str(x) for x in rhs]}")
        if all_non_negative:
            self.steps_output.append("  Tất cả b_i >= 0, bài toán đã khả thi.")
        else:
            self.steps_output.append("  Có b_i < 0, tiếp tục thuật toán đơn hình đối ngẫu.")
        return all_non_negative

    def _select_leaving_variable(self):
        """Bước 4: Chọn biến ra dựa trên b_i âm nhất."""
        self.steps_output.append("\n**Bước 4: Chọn biến ra**")
        
        min_b = min(self.tableau[:-1, -1])
        if min_b >= 0:
            self.steps_output.append("  Không có b_i < 0, không cần chọn biến ra.")
            return None
        leaving_row = np.argmin(self.tableau[:-1, -1])
        self.steps_output.append(f"  Chọn hàng {leaving_row + 1} với b[{leaving_row}] = {min_b}")
        return leaving_row

    def _select_entering_variable(self, leaving_row):
        """Bước 5: Chọn biến vào dựa trên tỉ lệ min {c_j / -a_{i0 j}} với a_{i0 j} < 0."""
        self.steps_output.append("\n**Bước 5: Chọn biến vào**")
        
        ratios = []
        for j in self.nonbasic_vars:
            if self.tableau[leaving_row, j] < 0:
                ratio = -self.tableau[-1, j] / self.tableau[leaving_row, j]
                ratios.append((ratio, j))
                self.steps_output.append(f"    Biến x{j+1}: {-self.tableau[-1, j]} / {-self.tableau[leaving_row, j]} = {ratio}")
        if not ratios:
            self.steps_output.append("  Không có biến vào hợp lệ, bài toán không khả thi.")
            return None
        min_ratio, entering_col = min(ratios)
        self.steps_output.append(f"  Chọn biến vào: x{entering_col + 1} với tỉ lệ {min_ratio}")
        return entering_col

    def _pivot(self, entering_col, leaving_row):
        """Bước 6: Thực hiện phép xoay."""
        self.steps_output.append("\n**Bước 6: Thực hiện phép xoay**")
        
        pivot_element = self.tableau[leaving_row, entering_col]
        self.steps_output.append(f"  Phần tử xoay tại [{leaving_row + 1}, {entering_col + 1}]: {pivot_element}")
        self.tableau[leaving_row] /= pivot_element
        for i in range(self.tableau.shape[0]):
            if i != leaving_row:
                factor = self.tableau[i, entering_col]
                self.tableau[i] -= factor * self.tableau[leaving_row]
        # Cập nhật biến cơ bản và không cơ bản
        old_basic = self.basic_vars[leaving_row]
        self.basic_vars[leaving_row] = entering_col
        self.nonbasic_vars.remove(entering_col)
        self.nonbasic_vars.append(old_basic)
        self.steps_output.append(f"  Biến ra: x{old_basic + 1}, Biến vào: x{entering_col + 1}")
        self.steps_output.append("  Bảng đơn hình sau khi xoay:")
        self.steps_output.append(self._render_tableau())

    def _render_tableau(self):
        """Hiển thị bảng đơn hình dưới dạng chuỗi."""
        headers = [f"x{i+1}" for i in range(self.num_original_vars + self.num_constraints)] + ["RHS"]
        rows = []
        for i in range(self.tableau.shape[0] - 1):
            row = [str(self.tableau[i, j]) for j in range(self.tableau.shape[1])]
            rows.append(f"Hàng {i+1}:\t" + "\t".join(row))
        z_row = [str(self.tableau[-1, j]) for j in range(self.tableau.shape[1])]
        rows.append("z:\t" + "\t".join(z_row))
        return "\n".join(rows)

    def _get_solution(self):
        """Bước 7: Trích xuất nghiệm từ bảng đơn hình."""
        self.steps_output.append("\n**Bước 7: Trích xuất nghiệm**")
        solution = {f'x{i+1}': Fraction(0) for i in range(self.num_original_vars)}
        for i, var in enumerate(self.basic_vars):
            if var < self.num_original_vars:
                solution[f'x{var + 1}'] = self.tableau[i, -1]
                self.steps_output.append(f"  x{var + 1} = {self.tableau[i, -1]}")
        for var in self.nonbasic_vars:
            if var < self.num_original_vars:
                self.steps_output.append(f"  x{var + 1} = 0")
        optimal_value = -self.tableau[-1, -1] if self.problem_type == 'max' else self.tableau[-1, -1]
        self.steps_output.append(f"  Giá trị tối ưu: {optimal_value}")
        return solution, optimal_value

    def solve(self):
        """Phương thức chính để giải bài toán."""
        self.steps_output.append("# Bộ Giải Tối Ưu Hóa Thông Minh")
        self.steps_output.append("Tự động chọn thuật toán phù hợp dựa trên điều kiện bài toán")
        
        # Kiểm tra điều kiện và chọn phương pháp
        method = self._check_initial_feasibility()
        
        if method == "bland_rule":
            # Sử dụng thuật toán Bland
            return self._solve_with_bland_rule()
        
        elif method == "dual_simplex":
            # Sử dụng thuật toán Dual Simplex
            self.steps_output.append("\n" + "="*50)
            self.steps_output.append("SỬ DỤNG THUẬT TOÁN DUAL SIMPLEX")
            self.steps_output.append("="*50)
            
            try:
                # Bước 1: Đảm bảo dạng chuẩn
                A, b = self._ensure_standard_form()
                
                # Bước 2: Xây dựng bảng đơn hình ban đầu
                self._build_initial_tableau(A, b)
                
                # Lặp thuật toán đối ngẫu
                max_iterations = 100
                iteration = 0
                while not self._check_optimality() and iteration < max_iterations:
                    iteration += 1
                    
                    # Bước 4: Chọn biến ra
                    leaving_row = self._select_leaving_variable()
                    if leaving_row is None:
                        break
                    
                    # Bước 5: Chọn biến vào
                    entering_col = self._select_entering_variable(leaving_row)
                    if entering_col is None:
                        self.steps_output.append("Bài toán không khả thi.")
                        return None, None
                    
                    # Bước 6: Thực hiện phép xoay
                    self._pivot(entering_col, leaving_row)
                
                if iteration >= max_iterations:
                    self.steps_output.append(f"Cảnh báo: Đạt giới hạn {max_iterations} vòng lặp!")
                
                # Bước 7: Trích xuất nghiệm
                solution, optimal_value = self._get_solution()
                
                # Kết quả cuối cùng
                self.steps_output.append("\n## Kết quả Cuối cùng")
                self.steps_output.append(f"**Giá trị Tối ưu:** {optimal_value}")
                self.steps_output.append("**Nghiệm Tối ưu:**")
                for var, val in solution.items():
                    self.steps_output.append(f"  {var} = {val}")
                
                return optimal_value, solution
                    
            except Exception as e:
                self.steps_output.append(f"Lỗi trong phương pháp đơn hình đối ngẫu: {str(e)}")
                return None, None

    def get_steps_output(self):
        """Trả về quá trình giải từng bước."""
        return "\n".join(self.steps_output)


class LinearProgrammingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Programming Solver")
        self.root.geometry("1200x800")
        
        # Create main frames
        self.input_frame = ttk.Frame(root, padding="10")
        self.input_frame.grid(row=0, column=0, sticky="nsew")
        
        self.output_frame = ttk.Frame(root, padding="10")
        self.output_frame.grid(row=0, column=1, sticky="nsew")
        
        # Configure weights
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)
        
        self.create_input_widgets()
        self.create_output_widgets()
    
    def create_input_widgets(self):
        # Problem type
        ttk.Label(self.input_frame, text="Kiểu bài toán:").grid(row=0, column=0, sticky="w", pady=5)
        self.problem_type = tk.StringVar(value="MAX")
        ttk.Radiobutton(self.input_frame, text="Maximization", variable=self.problem_type, value="MAX").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(self.input_frame, text="Minimization", variable=self.problem_type, value="MIN").grid(row=0, column=2, sticky="w")
        
        # Solving method
        ttk.Label(self.input_frame, text="Phương pháp:").grid(row=1, column=0, sticky="w", pady=5)
        self.solving_method = tk.StringVar(value="simplex")
        methods = [
            ("Đơn hình từ vựng", "simplex"),
            ("Hình học", "geometric"),
            ("Bland", "bland"),
            ("Hai pha", "two_phase"),
            ("Đơn hình đối ngẫu", "dual")
        ]
        
        for i, (text, value) in enumerate(methods):
            ttk.Radiobutton(self.input_frame, text=text, variable=self.solving_method, value=value).grid(row=1, column=i+1, sticky="w")
        
        # Number of variables and constraints
        ttk.Label(self.input_frame, text="Số biến:").grid(row=2, column=0, sticky="w", pady=5)
        self.num_vars = tk.StringVar(value="2")
        ttk.Entry(self.input_frame, textvariable=self.num_vars, width=5).grid(row=2, column=1, sticky="w")
        
        ttk.Label(self.input_frame, text="Số ràng buộc:").grid(row=3, column=0, sticky="w", pady=5)
        self.num_constraints = tk.StringVar(value="2")
        ttk.Entry(self.input_frame, textvariable=self.num_constraints, width=5).grid(row=3, column=1, sticky="w")
        
        ttk.Button(self.input_frame, text="Xác nhận", command=self.generate_input_fields).grid(row=4, column=0, columnspan=2, sticky="w", pady=10)
        
        # Frame for coefficients
        self.coef_frame = ttk.Frame(self.input_frame)
        self.coef_frame.grid(row=5, column=0, columnspan=6, sticky="nsew", pady=10)
        
        # Buttons
        btn_frame = ttk.Frame(self.input_frame)
        btn_frame.grid(row=6, column=0, columnspan=6, sticky="w", pady=10)
        
        ttk.Button(btn_frame, text="Giải", command=self.solve_problem).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Xoá hết", command=self.clear_fields).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Ví dụ mẫu", command=self.load_example).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Xuất kết quả", command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Xuất LaTeX", command=self.export_latex_advanced).pack(side=tk.LEFT, padx=5)
        
        # Initialize variables
        self.c_entries = []
        self.A_entries = []
        self.b_entries = []
        self.figure = None
        self.canvas = None
    
    def create_output_widgets(self):
        # Notebook for output tabs
        self.notebook = ttk.Notebook(self.output_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Steps tab
        self.steps_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.steps_frame, text="Các bước giải")
        
        self.steps_text = scrolledtext.ScrolledText(self.steps_frame, wrap=tk.WORD, width=60, height=30)
        self.steps_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Graph tab for geometric method
        self.graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_frame, text="Minh hoạ hình học")
    
    def generate_input_fields(self):
        # Clear previous fields
        for widget in self.coef_frame.winfo_children():
            widget.destroy()
        
        try:
            n = int(self.num_vars.get())
            m = int(self.num_constraints.get())
            
            if n <= 0 or m <= 0:
                messagebox.showerror("Invalid Input", "Number of variables and constraints must be positive integers")
                return
            
            # Objective function coefficients
            ttk.Label(self.coef_frame, text="Objective Function Coefficients:").grid(row=0, column=0, sticky="w", pady=5, columnspan=n+1)
            
            self.c_entries = []
            for j in range(n):
                ttk.Label(self.coef_frame, text=f"c{j+1}:").grid(row=1, column=j, padx=5)
                entry = ttk.Entry(self.coef_frame, width=5)
                entry.grid(row=2, column=j, padx=5, pady=2)
                self.c_entries.append(entry)
            
            # Constraint coefficients
            ttk.Label(self.coef_frame, text="Constraint Coefficients:").grid(row=3, column=0, sticky="w", pady=5, columnspan=n+2)
            
            self.A_entries = []
            self.b_entries = []
            
            # Column headers
            for j in range(n):
                ttk.Label(self.coef_frame, text=f"x{j+1}").grid(row=4, column=j, padx=5)
            ttk.Label(self.coef_frame, text="b").grid(row=4, column=n, padx=5)
            
            # Create entries for each constraint
            for i in range(m):
                row_entries = []
                for j in range(n):
                    entry = ttk.Entry(self.coef_frame, width=5)
                    entry.grid(row=5+i, column=j, padx=5, pady=2)
                    row_entries.append(entry)
                
                self.A_entries.append(row_entries)
                
                # b value
                b_entry = ttk.Entry(self.coef_frame, width=5)
                b_entry.grid(row=5+i, column=n, padx=5, pady=2)
                self.b_entries.append(b_entry)
                
                # Constraint type (always <= for standard form)
                ttk.Label(self.coef_frame, text="≤").grid(row=5+i, column=n-1, padx=0, pady=2)
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of variables and constraints must be integers")
    
    def clear_fields(self):
        self.num_vars.set("2")
        self.num_constraints.set("2")
        self.steps_text.delete(1.0, tk.END)
        self.generate_input_fields()
        
        # Clear graph if it exists
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
    
    def load_example(self):
        # Set up the example problem
        if self.problem_type.get() == "MAX":
            # Max example: Maximize 3x1 + 5x2 subject to x1 + 2x2 <= 8, 2x1 + x2 <= 10, x1,x2 >= 0
            self.num_vars.set("2")
            self.num_constraints.set("2")
            self.generate_input_fields()
            
            self.c_entries[0].insert(0, "3")
            self.c_entries[1].insert(0, "5")
            
            self.A_entries[0][0].insert(0, "1")
            self.A_entries[0][1].insert(0, "2")
            self.b_entries[0].insert(0, "8")
            
            self.A_entries[1][0].insert(0, "2")
            self.A_entries[1][1].insert(0, "1")
            self.b_entries[1].insert(0, "10")
        else:
            # Min example: Minimize 2x1 + 3x2 subject to x1 + x2 >= 5, 2x1 + x2 >= 6, x1,x2 >= 0
            # Converted to standard form: Minimize 2x1 + 3x2 subject to -x1 - x2 <= -5, -2x1 - x2 <= -6, x1,x2 >= 0
            self.num_vars.set("2")
            self.num_constraints.set("2")
            self.generate_input_fields()
            
            self.c_entries[0].insert(0, "2")
            self.c_entries[1].insert(0, "3")
            
            self.A_entries[0][0].insert(0, "-1")
            self.A_entries[0][1].insert(0, "-1")
            self.b_entries[0].insert(0, "-5")
            
            self.A_entries[1][0].insert(0, "-2")
            self.A_entries[1][1].insert(0, "-1")
            self.b_entries[1].insert(0, "-6")
    
    def get_problem_data(self):
        try:
            n = int(self.num_vars.get())
            m = int(self.num_constraints.get())
            
            c = [float(entry.get()) for entry in self.c_entries]
            
            A = []
            for i in range(m):
                row = [float(entry.get()) for entry in self.A_entries[i]]
                A.append(row)
            
            b = [float(entry.get()) for entry in self.b_entries]
            
            return c, A, b
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid numeric values: {str(e)}")
            return None, None, None
    
    def solve_problem(self):
        self.steps_text.delete(1.0, tk.END)
        
        c, A, b = self.get_problem_data()
        if c is None:
            return
            
        problem_type = self.problem_type.get().lower()
        method = self.solving_method.get()
        
        try:
            if method == "simplex":
                solver = SimplexDictionary(c, A, b, problem_type)
                result = solver.solve()
                self.steps_text.insert(tk.END, solver.get_steps_output())
                
            elif method == "geometric":
                if len(c) != 2:
                    messagebox.showinfo("Information", "Geometric method only works with 2 variables. Using 2 variables.")
                    
                solver = GeometricMethod(c, A, b, problem_type)
                fig, best_point, best_value = solver.solve()
                self.steps_text.insert(tk.END, solver.get_steps_output())
                
                # Display the graph
                if self.canvas:
                    self.canvas.get_tk_widget().destroy()
                
                self.canvas = FigureCanvasTkAgg(fig, self.graph_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Switch to graph tab
                self.notebook.select(1)
                
            elif method == "bland":
                solver = BlandRule(c, A, b, problem_type)
                result = solver.solve()
                self.steps_text.insert(tk.END, solver.get_steps_output())
                
            elif method == "two_phase":
                solver = TwoPhaseMethod(c, A, b, problem_type)
                result = solver.solve()
                self.steps_text.insert(tk.END, solver.get_steps_output())
                
            elif method == "dual":
                solver = DualSimplexSolver(c, A, b, problem_type)
                result = solver.solve()
                self.steps_text.insert(tk.END, solver.get_steps_output())
            
            # Switch to steps tab if not using geometric method
            if method != "geometric":
                self.notebook.select(0)
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            import traceback
            self.steps_text.insert(tk.END, traceback.format_exc())
    
    def export_results(self):
        """Export the results to a text file"""
        results = self.steps_text.get(1.0, tk.END)
        if not results.strip():
            messagebox.showinfo("Information", "No results to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(results)
            messagebox.showinfo("Success", f"Results exported to {file_path}")
    
    def export_latex(self):
        """Export the results in LaTeX format with proper formatting"""
        results = self.steps_text.get(1.0, tk.END).strip()
        if not results:
            messagebox.showinfo("Information", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".tex",
            filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")]
        )
    
        if file_path:
            try:
                # Process the text to make it LaTeX-friendly
                latex_content = self.format_text_for_latex(results)
            
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("\\documentclass[12pt]{article}\n")
                    f.write("\\usepackage[utf8]{inputenc}\n")
                    f.write("\\usepackage[T1]{fontenc}\n")
                    f.write("\\usepackage{amsmath}\n")
                    f.write("\\usepackage{amssymb}\n")
                    f.write("\\usepackage{array}\n")
                    f.write("\\usepackage{booktabs}\n")
                    f.write("\\usepackage[margin=1in]{geometry}\n")
                    f.write("\\usepackage{fancyhdr}\n")
                    f.write("\\pagestyle{fancy}\n")
                    f.write("\\fancyhf{}\n")
                    f.write("\\rhead{Linear Programming Solution}\n")
                    f.write("\\cfoot{\\thepage}\n\n")
                    f.write("\\title{Linear Programming Solution}\n")
                    f.write("\\date{\\today}\n\n")
                    f.write("\\begin{document}\n")
                    f.write("\\maketitle\n\n")
                    f.write(latex_content)
                    f.write("\n\\end{document}")
            
                messagebox.showinfo("Success", f"LaTeX exported to {file_path}")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export LaTeX: {str(e)}")

    def format_text_for_latex(self, text):
        """Convert plain text to LaTeX-friendly format"""
        # Escape special LaTeX characters
        latex_text = text.replace('\\', '\\textbackslash{}')
        latex_text = latex_text.replace('&', '\\&')
        latex_text = latex_text.replace('%', '\\%')
        latex_text = latex_text.replace('$', '\\$')
        latex_text = latex_text.replace('#', '\\#')
        latex_text = latex_text.replace('^', '\\textasciicircum{}')
        latex_text = latex_text.replace('_', '\\_')
        latex_text = latex_text.replace('{', '\\{')
        latex_text = latex_text.replace('}', '\\}')
        latex_text = latex_text.replace('~', '\\textasciitilde{}')
        
        # Convert common mathematical expressions
        latex_text = latex_text.replace('<=', '$\\leq$')
        latex_text = latex_text.replace('>=', '$\\geq$')
        latex_text = latex_text.replace('≤', '$\\leq$')
        latex_text = latex_text.replace('≥', '$\\geq$')
        latex_text = latex_text.replace('∞', '$\\infty$')
        
        # Format sections and headers
        lines = latex_text.split('\n')
        formatted_lines = []
        in_table = False
        table_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if in_table:
                    formatted_lines.extend(self.format_table_for_latex(table_lines))
                    table_lines = []
                    in_table = False
                continue
            
            # Check for section headers
            if line.endswith(':') and len(line) > 3:
                formatted_lines.append(f"\\section*{{{line[:-1]}}}")
            elif line.startswith(('Step', 'Iteration', 'Phase', 'Bước')):
                formatted_lines.append(f"\\subsection*{{{line}}}")
            # Check for table-like content
            elif '|' in line and line.count('|') >= 2:
                table_lines.append(line)
                in_table = True
            else:
                # Regular text
                formatted_lines.append(line)
        
        if in_table:
            formatted_lines.extend(self.format_table_for_latex(table_lines))
        
        return '\n'.join(formatted_lines)

    def format_table_for_latex(self, table_lines):
        """Format table lines for LaTeX"""
        if not table_lines:
            return []
        
        result = []
        result.append('\\begin{center}')
        # Tính số cột dựa trên số dấu '|'
        num_cols = max(line.count('|') for line in table_lines) - 1
        result.append('\\begin{tabular}{' + 'c|' * num_cols + '}')
        result.append('\\hline')
        
        for line in table_lines:
            # Loại bỏ khoảng trắng thừa và thay thế '|' bằng '&'
            clean_line = line.strip().strip('|')
            cells = [cell.strip() for cell in clean_line.split('|')]
            # Đảm bảo số cột đúng
            while len(cells) < num_cols:
                cells.append('')
            formatted_line = ' & '.join(cells) + ' \\\\'
            result.append(formatted_line)
            result.append('\\hline')
        
        result.append('\\end{tabular}')
        result.append('\\end{center}')
        
        return result

    def export_latex_advanced(self):
        """Advanced LaTeX export with better table formatting"""
        results = self.steps_text.get(1.0, tk.END).strip()
        if not results:
            messagebox.showinfo("Information", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".tex",
            filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("\\documentclass[12pt]{article}\n")
                    f.write("\\usepackage[utf8]{inputenc}\n")
                    f.write("\\usepackage[T1]{fontenc}\n")
                    f.write("\\usepackage{amsmath}\n")
                    f.write("\\usepackage{amssymb}\n")
                    f.write("\\usepackage{array}\n")
                    f.write("\\usepackage{booktabs}\n")
                    f.write("\\usepackage{longtable}\n")
                    f.write("\\usepackage[margin=1in]{geometry}\n")
                    f.write("\\usepackage{fancyhdr}\n")
                    f.write("\\usepackage{listings}\n")
                    f.write("\\lstset{basicstyle=\\ttfamily\\small, breaklines=true}\n")
                    f.write("\\pagestyle{fancy}\n")
                    f.write("\\fancyhf{}\n")
                    f.write("\\rhead{Linear Programming Solution}\n")
                    f.write("\\cfoot{\\thepage}\n\n")
                    f.write("\\title{Linear Programming Solution}\n")
                    f.write("\\author{Linear Programming Solver}\n")
                    f.write("\\date{\\today}\n\n")
                    f.write("\\begin{document}\n")
                    f.write("\\maketitle\n\n")
                    
                    latex_content = self.process_content_for_latex(results)
                    f.write(latex_content)
                    
                    f.write("\n\\end{document}")
                
                messagebox.showinfo("Success", f"LaTeX exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export LaTeX: {str(e)}")

    def process_content_for_latex(self, content):
        """Process content with better structure recognition"""
        lines = content.split('\n')
        result = []
        in_table = False
        table_lines = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if in_table:
                    result.extend(self.format_table_for_latex(table_lines))
                    table_lines = []
                    in_table = False
                continue
            
            if '|' in line and line.count('|') >= 2:
                if not in_table:
                    in_table = True
                table_lines.append(line)
            else:
                if in_table:
                    result.extend(self.format_table_for_latex(table_lines))
                    table_lines = []
                    in_table = False
                
                formatted_line = self.format_line_for_latex(line)
                result.append(formatted_line)
        
        if in_table:
            result.extend(self.format_table_for_latex(table_lines))
        
        return '\n'.join(result)

    def format_line_for_latex(self, line):
        """Format a single line for LaTeX"""
        line = line.replace('\\', '\\textbackslash{}')
        line = line.replace('&', '\\&')
        line = line.replace('%', '\\%')
        line = line.replace('$', '\\$')
        line = line.replace('#', '\\#')
        line = line.replace('^', '\\textasciicircum{}')
        line = line.replace('_', '\\_')
        line = line.replace('{', '\\{')
        line = line.replace('}', '\\}')
        line = line.replace('~', '\\textasciitilde{}')
        
        line = line.replace('<=', '$\\leq$')
        line = line.replace('>=', '$\\geq$')
        line = line.replace('≤', '$\\leq$')
        line = line.replace('≥', '$\\geq$')
        line = line.replace('∞', '$\\infty$')
        
        if line.endswith(':') and len(line) > 3:
            return f"\\section*{{{line[:-1]}}}"
        elif line.startswith(('Step', 'Iteration', 'Phase', 'Bước')):
            return f"\\subsection*{{{line}}}"
        else:
            return line

def main():
    root = tk.Tk()
    app = LinearProgrammingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()