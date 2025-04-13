from manim import *
from manim.utils.tex import TexTemplate
import numpy as np

# Template hỗ trợ tiếng Việt với pdflatex
def get_vietnamese_template():
    preamble = r"""
    \usepackage[utf8]{vietnam}
    \usepackage{amsmath, amssymb}
    \usepackage{tikz}
    \usepackage{fancyhdr}
    \usepackage{graphicx}
    \mathversion{bold}
    \usepackage{color}
    """
    return TexTemplate(preamble=preamble, tex_compiler="pdflatex")

class Tikz(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        # Dùng template tiếng Việt
        template = get_vietnamese_template()

        # Đề bài
        question = Tex(
            r"\textbf{Câu 1.} Cho khối chóp $S.ABCD$ có đáy $ABCD$ là hình thoi cạnh bằng $2$, "
            r"$\widehat{ABC} = 120^\circ$, $SD = SA$. Mặt phẳng $(SAD)$ vuông góc với đáy và "
            r"cạnh bên $SA$ tạo với mặt phẳng đáy một góc $60^\circ$. Thể tích khối chóp $S.ABCD$ bằng bao nhiêu?",
            tex_template=template
        ).set_color(GREEN)

        question.scale(0.5)
        question.to_edge(UP, buff=0.5)
        question.shift(RIGHT * 0.8)  # Dịch đề bài sang phải

        # Hình chữ nhật bao quanh đề bài
        question_box = Rectangle(
            width=question.width + 0.6, height=question.height + 0.4, color=YELLOW
        ).surround(question)

        self.play(Write(question))
        self.play(Create(question_box))
        self.wait(2)

        # Tạo hình và di chuyển xuống + sang phải
        figure_group = self.create_geometry()
        figure_group.scale(0.9)
        figure_group.to_corner(UL)
        figure_group.shift(DOWN * 1.5 + RIGHT * 0.4)  # Dịch hình vẽ

        self.play(Create(figure_group))
        self.wait(1)

        # Tiêu đề "Lời giải:"
        solution_title = Tex(r"\textbf{\textit{\underline{Lời giải.}}}", tex_template=template)
        solution_title.scale(0.4)
        solution_title.next_to(question_box, DOWN, buff=0.2)
        self.play(Write(solution_title))
        self.wait(1)

        # Các bước giải (step1 đến step4)
        step1 = Tex(
            r"$SA$ tạo với mặt phẳng đáy một góc $60^{\circ}$, nghĩa là $\widehat{SAH} = 60^{\circ}$.",
            tex_template=template
        )
        step2 = Tex(
            r"Ta suy ra $SH = AH \cdot \tan 60^{\circ} = \sqrt{3}$.",
            tex_template=template
        )
        step3 = Tex(
            r"Vì $\widehat{ABC} = 120^{\circ}$ nên $AC = 2\sqrt{3}$ và $DB = 2$,",
            tex_template=template
        )
        step4 = Tex(
            r"nên diện tích đáy $S_{ABCD} = \dfrac{1}{2} \cdot 2\sqrt{3} \cdot 2 = 2\sqrt{3}$.",
            tex_template=template
        )
        step5 = Tex(
            r"Thể tích khối chóp $S.ABCD$ là: $V = \dfrac{1}{3} \cdot \sqrt{3} \cdot 2\sqrt{3} = 2$.",
            tex_template=template
        )

        # Scale và sắp xếp từng bước
        for step in [step1, step2, step3, step4]:
            step.scale(0.5)

        steps = VGroup(step1, step2, step3, step4).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        steps.to_edge(RIGHT)

        for step in steps:
            self.play(Write(step))
            self.wait(1)

        # Xử lý riêng step5
        step5.scale(0.5)
        step5.next_to(steps, DOWN, aligned_edge=LEFT, buff=0.4)

        # Hình chữ nhật bao quanh step5
        step5_box = Rectangle(
            width=step5.width + 0.2,
            height=step5.height + 0.2,
            color=RED
        ).surround(step5)

        self.play(Write(step5))
        self.play(Create(step5_box))
        self.wait(2)

    def create_geometry(self):
        A = np.array([0, 0, 0])
        B = np.array([3, 0, 0])
        C = np.array([4, 2, 0])
        S = np.array([0.5, 4, 0])
        H = np.array([0.5, 1, 0])
        D = np.array([1, 2, 0])

        edges = [
            Line(A, B),
            Line(B, C),
            Line(S, A),
            Line(S, B),
            Line(S, C),
        ]

        dashed_edges = [
            DashedLine(A, H),
            DashedLine(H, D),
            DashedLine(S, D),
            DashedLine(D, C),
            DashedLine(S, H),
            DashedLine(D, B),
            DashedLine(B, H),
        ]

        dots = [Dot(p) for p in [A, B, C, S, H, D]]

        template = get_vietnamese_template()
        labels = [
            MathTex("A", tex_template=template).next_to(A, DOWN),
            MathTex("B", tex_template=template).next_to(B, DOWN),
            MathTex("C", tex_template=template).next_to(C, RIGHT),
            MathTex("S", tex_template=template).next_to(S, UL),
            MathTex("H", tex_template=template).next_to(H, DR),
            MathTex("D", tex_template=template).next_to(D, UR),
        ]

        return VGroup(*edges, *dashed_edges, *dots, *labels)

