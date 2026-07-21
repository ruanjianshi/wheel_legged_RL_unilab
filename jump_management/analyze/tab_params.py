#!/usr/bin/env python3
"""机器人参数表 — LaTeX 格式."""
from __future__ import annotations


def generate_params_table(out_path: str = "paper/table_robot_params.tex"):
    latex = r"""\begin{table}[htbp]
    \centering
    \caption{xqrobotwl机器人主要参数}
    \label{tab:robot_params}
    \begin{tabular}{lcc}
        \toprule
        参数 & 值 & 说明 \\
        \midrule
        总质量 & 18.65\,\text{kg} & CAD装配总计 \\
        机身质量 & 5.41\,\text{kg} & 含电池/电路/外壳 \\
        车轮质量（单侧） & 2.32\,\text{kg} & 含轮毂/轮胎/电机转子 \\
        车轮半径 & 0.065\,\text{m} & 轮胎外径 \\
        大腿长 & 0.224\,\text{m} & 髋—膝关节轴线距 \\
        小腿长 & 0.200\,\text{m} & 膝—轮心轴线距 \\
        单腿自由度 & 3+1 & 3旋转关节+1主动轮 \\
        基座惯量$(I_{xx},I_{yy},I_{zz})$ & $(0.014,0.025,0.025)\,\text{kg\,m}^2$ & \\
        \bottomrule
    \end{tabular}
\end{table}
"""

    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Saved: {out_path}")
    print(latex)


if __name__ == "__main__":
    generate_params_table()
