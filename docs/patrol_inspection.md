# 诊断告警触发巡检复检

告警计划将设备坐标、任务优先级、复检时限和服务时间与诊断结果关联。调度器到告警送达时刻才创建复检任务。复检默认优先级为 9，参与空闲机器人的下一轮任务匹配。正在执行的任务继续执行。

```bash
diag-evidence patrol-demo --scenario clean --seed 7 --execution confirmed --traffic yield --inspection examples/patrol_inspection.json --output-dir artifacts/inspection
```

使用返回的 `replay_path` 查看告警到达、派工和设备状态变化。“下一条异常”可以跳到这些事件。示例中的 pump-A 在第 4 步生成复检，第 20 步完成确认并更新为“复检发现异常”。

告警计划 JSON 包含 `assets` 和 `alerts`。每条告警的 `result` 可直接使用 `DiagnosisAgent.run` 返回的结构化结果，设备编号和观测、送达时刻由调用方提供。Python 调用方也可用 `diag_evidence.patrol.inspection.alert_from_result` 构造 `Alert`，再通过 `InspectionPlan` 传给 `simulate`。回放记录诊断运行编号、派工决策和状态变化。

- 默认 `--alarm-policy verified` 要求分类成功、诊断流程完成、支持等级高且结论强度至少为中等。使用 `--alarm-policy all` 可比较采用全部可用故障分类的策略。
- 同一设备的重复结果、超过 12 步的告警、已存在待确认任务、健康分类和不可用分类都会记下原因。设备同一时刻只保留一个待确认复检任务。
- 复检任务通过执行确认回执更新设备状态。示例中的复检发现由 `simulated_finding` 指定，可为正常、异常或无结论。设备在收到确认前保持待复检状态。常规巡检消息无法结束复检任务。

```bash
python scripts/patrol_inspection_compare.py --inspection examples/patrol_inspection.json --output-dir artifacts/inspection-study
```

在种子 80–99 的 20 个布局上配对运行，共 40 次仿真。两种策略都启用执行确认和单方绕行，使用同一批告警和模拟发现。数据同时包含证据较弱的真实异常和正常设备，也包含证据较强但复检正常的样例。

| 每个布局的平均值 | 采用全部故障分类 | 核验后派工 |
|---|---:|---:|
| 生成复检任务 | 4 | 2 |
| 复检确认异常 | 1.95 | 1 |
| 复检确认正常 | 2 | 1 |
| 按时完成常规巡检 | 8 | 8 |
| 能耗 | 164.50 | 153.10 |

核验减少了正常设备的复检和能耗，也筛掉了低支持的真实异常。测试通过既有诊断模块的合成服务验证了结果适配；设备复检发现来自示例预设。[完整对照结果](results/patrol_inspection_20261110.json)
