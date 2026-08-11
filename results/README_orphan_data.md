# 遗留/孤儿数据文件说明

本目录下的以下文件**未被论文 `paper/paper_final.tex` 引用**，仅作为历史全量基准的留存，供溯源与复核参考。

## 1. `paper_full_benchmark_ablation.json`

- **性质**：ONNX Runtime 全量基准的"生命周期"消融数据。
- **与论文表8（`go_session_pool_ablation.json`）的关键区别**：本文件的 `start_pm_mb` 在 **Session 池创建之前**采样（如 pool=4/intra=1 起始约 247 MB），因此其 `pm_drift_mb` 包含一次性 **Session 创建内存开销**（约 2 GB 量级），漂移值很大（如 2064 MB）。
- **论文表8 的真实数据源**是 `go_session_pool_ablation.json`，其 `start_pm_mb` 在 **Session 池创建之后**采样，故 `PM漂移` 仅含推理阶段增量（接近零或随池容量增长），不含创建开销。
- 二者差异源于**测量起点不同**，均为 `PrivateMemorySize64`（PM）口径，并非 RSS vs PM 之别。若需复现论文表8，请使用 `go_session_pool_ablation.json`。

## 2. `paper_full_benchmark_summary.txt`

- **性质**：上述全量基准的文本汇总报告。
- **已知问题**：`[6/6] 批处理` 部分 Batch 2/4 存在测量 bug（吞吐显示 `+Inf`、延迟 `0.000 ms`），数据无效。**规范的批处理数据见 `go_batch_inference_result.json`**（论文 §4.2 实际引用此文件）。
- 本文件未被论文引用，仅供历史留存。
