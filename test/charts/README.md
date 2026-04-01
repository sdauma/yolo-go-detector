# 论文图表生成说明

## 目录结构

```
yolo-go-detector/
├── test/                    # 测试程序
│   └── charts/             # 图表生成脚本
│       ├── generate_all_charts.py           # 生成图2-图7
│       └── generate_session_pool_arch.py    # 生成图1
├── results/                 # 测试结果存储
│   └── charts/            # 图表保存位置
│       └── final_charts/   # 最终论文图表
└── paper/                  # 论文相关文件
    └── paper_final.tex  # 论文源文件
```

## 生成论文图表

### 生成图1（Session Pool架构）

```bash
cd d:\mlz\trae_projects\1\yolo-go-detector\test\charts
python generate_session_pool_arch.py
```

### 生成图2-图7

```bash
cd d:\mlz\trae_projects\1\yolo-go-detector\test\charts
python generate_all_charts.py
```

## 论文图表列表

| 序号 | 图表名称 | PNG文件名 | 说明 |
|------|----------|-----------|------|
| 1 | Session Pool架构 | final_charts/fig1_session_pool_architecture.png | 图1 |
| 2 | 吞吐量对比 | final_charts/fig2_throughput_comparison.png | 图2 |
| 3 | 内存对比 | final_charts/fig3_memory_comparison.png | 图3 |
| 4 | 批量处理效果 | final_charts/fig4_batch_effect.png | 图4 |
| 5 | 模型大小对比 | final_charts/fig5_model_size_comparison.png | 图5 |
| 6 | CPU利用率 | final_charts/fig6_cpu_utilization.png | 图6 |
| 7 | 稳定性分析 | final_charts/fig7_stability.png | 图7 |

## 编译论文

```bash
cd d:\mlz\trae_projects\1\yolo-go-detector\paper
pdflatex paper_final.tex
```

## 注意事项

1. **所有图表都保存到 `results/charts/` 目录**
2. **论文直接引用 `results/charts/final_charts/` 下的PNG图表**
3. **字体配置**：中文使用华文中宋，英文使用Times New Roman
4. **数据精度**：符合核心期刊数值处理规范

## 更新记录

### 2026年4月1日更新（v2.3）
- ✅ 修复图2（吞吐量对比）的图表类型问题，确保生成柱状图而非折线图
- ✅ 优化图2布局，调整y轴范围，解决箭头盖住数据的问题
- ✅ 统一所有图表生成脚本的字体配置，确保中文使用华文中宋，英文使用Times New Roman
- ✅ 重新生成所有7个论文图表，保持风格一致
- ✅ 修复数据读取脚本，确保正确解析架构对比测试结果

**版本**：v2.3  
**更新日期**：2026-04-01
