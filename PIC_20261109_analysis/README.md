# PIC 研究分析与图表

先打开PIC_研究分析结果报告.html。该文件内嵌图表，可单文件离线阅读。
figures下每幅图有PNG、矢量PDF及SVG；tables为可复核汇总数据；code为复现程序。
执行说明.md记录建模前确定的实施细化；run_summary.json保存种子、模型参数和输入摘要。
private目录为患者划分和预测明细，仅保留本地，不进入分享包；不要将原始数据或private目录放入公开仓库。

复现：
python code/analyze_pic.py --input "/path/to/PIC"
python code/build_results_report.py

所需环境：Python numpy pandas scipy scikit-learn statsmodels matplotlib markdown threadpoolctl。
本阶段尚未完成变量官方单位核验，报告按编号解释。当前为内部验证研究探索，不是临床工具。
