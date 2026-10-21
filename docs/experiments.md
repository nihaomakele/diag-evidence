# DiagEvidence 可复跑验证

## 合成契约场景

```bash
diag-evidence evaluate --output-dir artifacts/evaluation
```

12个场景的输入和工具响应来自`diag_evidence.demo`。每个场景冻结预期终态、支持等级和结论上限，评分程序逐项比对；同时检查实际`effective_strength`不超过上限，报告工具字段、Markdown与顶层结果一致。无效信号还要求transport调用数为0。所有结果保存为JSON和Markdown，实际运行编号可追溯到报告与日志。字段与正文的一致性检查不判断自然语言推论是否正确。

该数据没有真实设备故障标签，`matched_expectations/total`仅是工程行为符合预期的计数，不计算准确率、召回率或诊断收益。场景选择不是目标总体抽样，因此不从12个样例估计生产错误率。

## 代码与HTTP边界

```bash
python -m unittest discover -s diag_evidence/tests -t . -v
```

验证非法输入、分类契约、必需检查未知、强度限制、提示词检查四态、嵌套脱敏、重复/并发产物、写入失败和CLI行为。HTTP测试实际通过本机随机端口，检查非JSON、4xx/5xx、超时、重试预算与畸形结构；清理测试服务。

## 安装与发布包

```bash
python -m build
python scripts/check_release.py dist
```

在新环境安装wheel后，切换到项目以外的目录运行`diag-evidence --version`与`diag-evidence demo --scenario all --no-write`。同时检查sdist可重建，wheel含所需SOP资源，分发档案无未核实权重、密钥、缓存和运行产物。真实验证记录见根目录`VALIDATION.md`。

## 后续真实任务评价

先明确输入对象、任务标签和记录来源，再按设备或独立原始记录划分训练/验证/测试；相邻窗口不能随机散入不同集合后宣称泛化。冻结上游工具输出，对比直接输出、简单置信度阈值、旧规则、新规则及消融。

同时报告错误高强度输出占全部样本和占高强度输出的比例，以及覆盖率、拒绝率、错误拒绝、原因分布和成本。必须预先按实际`effective_strength`定义输出阈值，不能用`max_allowed_strength`代替实际输出。当前合法分类适配器通常只请求moderate；若只计算字面strong结果可能没有分母，此时记N/A。区间估计以独立设备/记录或时间块为单位。缺乏真实标签时，只做契约与失效处理验证。阶段目标见[研究路线](roadmap.md)。
