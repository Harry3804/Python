from manim import *
from manim.utils.tex import TexTemplate
import numpy as np

def get_vietnamese_template():
    preamble = r"""
    \usepackage[utf8]{vietnam}
    \usepackage{xcolor}  
    \usepackage{amsmath, amssymb}
    \usepackage{tikz}
    \usepackage{fancyhdr}
    \usepackage{graphicx}
    \mathversion{bold}
    \newcommand{\vt}[1]{\overrightarrow{#1}}

    % Màu Manim
    \makeatletter
    \@ifundefined{manimred}{
        \definecolor{manimred}{rgb}{0.988, 0.384, 0.333}
    }{}
    \makeatother
    """
    return TexTemplate(preamble=preamble, tex_compiler="pdflatex")

class Tikz(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        template = get_vietnamese_template()

        # Đề bài
        problem = Tex(
            r"""\begin{tabular}[t]{@{}l@{}}
            \textbf{Bài toán.} Chứng minh rằng trong một hình bình hành, tổng bình phương hai đường chéo \\
            \phantom{\textbf{Bài toán.}} bằng tổng bình phương các cạnh.
            \end{tabular}""",
            tex_template=template
        ).set_color(GREEN).scale(0.5)
        problem.to_corner(UL)


        problem_box = Rectangle(
            width=problem.width + 0.6,
            height=problem.height + 0.4,
            color=YELLOW
        ).surround(problem)

        self.play(Write(problem), Create(problem_box))
        self.wait(1.5)

        # Tiêu đề lời giải
        solution_title = Tex(
            r"\textbf{\textit{\underline{Chứng minh.}}}",
            tex_template=template
        ).scale(0.5).to_edge(UP, buff=1.5)
        self.play(Write(solution_title))
        self.wait(1)

        # Câu dẫn trước khi dựng hình
        step1 = Tex(r"Xét hình bình hành $ABCD$ như sau:", tex_template=template).scale(0.5)
        step1.next_to(solution_title, DOWN, buff=0.4)
        self.play(Write(step1))
        self.wait(0.5)

        # Hình vẽ
        points = {
            "A": np.array([0, 0, 0]),
            "B": np.array([3, 0, 0]),
            "C": np.array([2, -1, 0]),
            "D": np.array([-1, -1, 0]),
        }

        segments = [
            Line(points["A"], points["B"]),
            Line(points["B"], points["C"]),
            Line(points["C"], points["D"]),
            Line(points["D"], points["A"]),
        ]
        diagonals = [
            Line(points["A"], points["C"], color=RED),
            Line(points["B"], points["D"], color=RED)
        ]

        dots = [Dot(points[name]) for name in "ABCD"]
        labels = [
            MathTex(name, tex_template=template).scale(0.5).next_to(points[name], direction)
            for name, direction in zip("ABCD", [UP, UP, DOWN, DOWN])
        ]

        figure = VGroup(*segments, *diagonals, *dots, *labels)
        figure.scale(0.7)
        figure.next_to(step1, DOWN, buff=0.4)

        self.play(Create(figure))
        self.wait(1)

        # Các bước chứng minh
        step2 = Tex(r"Ta có: $\textcolor{manimred}{AC^2 + BD^2} = (\vt{AC})^2 + (\vt{BD})^2 = (\vt{AB} + \vt{BC})^2 + (\vt{BC} + \vt{CD})^2$", tex_template=template)
        step3 = Tex(r"$= (\vt{AB})^2 + (\vt{BC})^2 + 2\vt{AB} \cdot \vt{BC} + (\vt{BC})^2 + (\vt{CD})^2 + 2\vt{BC} \cdot \vt{CD}$", tex_template=template)
        step4 = Tex(r"$= AB^2 + BC^2 + CD^2 + AD^2 + 2\vt{BC} \cdot (\vt{AB} + \vt{CD})$", tex_template=template)
        step5 = Tex(r"$= AB^2 + BC^2 + CD^2 + AD^2 + 2\vt{BC} \cdot \vt{0}$", tex_template=template)
        step6 = Tex(r"Vậy $\textcolor{manimred}{AC^2 + BD^2} = AB^2 + BC^2 + CD^2 + AD^2$ (đpcm).", tex_template=template)

        steps = VGroup(step2, step3, step4, step5, step6).arrange(
            DOWN, center=False, aligned_edge=LEFT, buff=0.4
        )
        steps.scale(0.5)
        steps.next_to(figure, DOWN, buff=0.4)

        for step in [step2, step3, step4, step5, step6]:
            self.play(Write(step))
            self.wait(0.5)

        final_box = Rectangle(
            width=step6.width + 0.3,
            height=step6.height + 0.3,
            color=RED
        ).surround(step6)
        self.play(Create(final_box))
        self.wait(2)


