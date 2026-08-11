# 论文图表生成说明

## 目录结构

```
yolo-go-detector/
├── test/                    # 测试程序
│   └── charts/             # 图表生成脚本（含 run_charts.bat）
├── results/                 # 测试结果存储
│   └── charts/            # 图表保存位置（生成后的 PNG/PDF 图表）
└── paper/                  # 论文相关文件
    └── paper_final.tex  # 论文源文件（\includegraphics 引用 results/charts/figX.png）
```

## 一键生成全部图表

执行 `run_charts.bat`，将按序调用以下 **8 个脚本**（任一失败不影响其余脚本继续执行，脚本结束会汇总 Passed/Failed 数量）：

| 序号 | 脚本 | 生成内容 |
|------|------|----------|
| 1 | `generate_session_pool_arch.py` | 图1：Session Pool 架构图 |
| 2 | `generate_all_charts.py` | 图2–图7：吞吐量/内存/批处理/模型规模/CPU利用率/稳定性主图（含 fig5 模型规模对比） |
| 3 | `generate_journal_charts.py` | 期刊补充图（journal 版） |
| 4 | `generate_reinforced_charts.py` | 强化实验对比图 |
| 5 | `generate_memory_scalability.py` | 内存可扩展性图 |
| 6 | `generate_latency_boxplot.py` | 延迟箱线图 |
| 7 | `plot_rss_curve.py` | RSS/PM 内存曲线图 |
| 8 | `generate_reference_flowchart.py` | 参考流程图 |

> 注：模型规模对比图（fig5）由 `generate_all_charts.py` 统一生成，无需单独的 `generate_model_size_comparison.py`（该脚本已移除）。

> 注：仓库实际共有 **9 个**图表生成脚本（详见《测试程序与图表生成程序完整清单》第 6 节）；上表 `run_charts.bat` 自动调用其中 8 个，`generate_arena_charts.py`（Arena 消融相关图表）未接入批处理，需单独运行以生成 Arena 图。

也可单独运行某个脚本，例如：

```bash
cd d:\mlz\trae_projects\1\yolo-go-detector\test\charts
python generate_session_pool_arch.py
```

## 论文图表列表

| 序号 | 图表名称 | 文件名（位于 results/charts/） | 说明 |
|------|----------|-----------------------------------|------|
| 1 | Session Pool架构 | fig1_session_pool_architecture.png | 图1 |
| 2 | 吞吐量对比 | fig2_throughput_comparison.png | 已退出正文（改用表 tab:arch_scalability / tab:fairness_comparison） |
| 3 | 内存对比 | fig3_memory_comparison.png | 图2（文件名fig3，实际图号2） |
| 4 | 批量处理效果 | fig4_batch_effect.png | 图3 |
| 5 | 模型大小对比 | fig5_model_size_comparison.png | 图4 |
| 6 | CPU利用率 | fig6_cpu_utilization.png | 图5 |
| 7 | 稳定性分析 | fig7_stability.png | 图6 |

> 注：论文 `paper_final.tex` 通过 `\includegraphics{../results/charts/figX.png}` 直接引用 **PNG**（600dpi）格式图表；脚本同时输出同名 `.pdf` 矢量副本备用。`journal` 前缀的变体用于期刊投稿版本，文件名不同。文件名编号（fig3起）比 LaTeX 自动图号大 1，属历史遗留（fig2 退出正文所致），不影响编译。

## 编译论文

```bash
cd d:\mlz\trae_projects\1\yolo-go-detector\paper
xelatex paper_final.tex
```

## 注意事项

1. **所有图表都保存到 `results/charts/` 目录（PNG 格式，600dpi；同时生成同名 PDF 矢量副本）**，论文直接引用该目录下的 `figX.png`。
2. **内存口径**：论文核心内存指标统一使用 PM（Private Memory，对应 `PrivateMemorySize64`），与 `RSS` 标签的历史脚本名（如 `plot_rss_curve.py`）无关，数据含义以论文 §3.3 声明为准。
3. **字体配置**：中文使用华文中宋，英文使用 Times New Roman。
4. **数据精度**：符合核心期刊数值处理规范。

## 更新记录

### 2026年7月10日更新
- ✅ 修正脚本清单：实际由 `run_charts.bat` 调用 **8 个**生成脚本（原文档仅列 2 个，已过时）。
- ✅ 修正图表引用路径：论文图表统一保存至 `results/charts/`，非原文档所述 `final_charts/`。格式以 2026-07-24 直读 `paper_final.tex` 核实为准——正文 7 处 `\includegraphics` 均为 `figX.png`（600dpi），故文档主体（注意事项第 1、3 条）已统一为 PNG 措辞，PDF 仅作同名矢量副本备用。

### 2026年7月24日更新
- ✅ 直读 `paper_final.tex` 核实：正文 7 处图表引用（`\includegraphics`）全部为 `results/charts/figX.png`（600dpi），无一引用 PDF。据此将文档主体（注意事项第 1、3 条）由"PDF"修正回"PNG"措辞，并修正 2026-07-10 记录中"实际引用 PDF"的不准确表述。
- ✅ 字体口径核实：图表中文统一使用华文中宋（STZhongsong，回退 SimSun/SimHei），英文使用 Times New Roman；与 `font_utils.py` 实际注册一致（`font_utils.py` L44）。
