# 历史检索案例的声明适用性检查

本功能属于0.2.0.dev2本地开发版，尚未发布。它把已有的部分元信息用于规则判定，不验证设备真值、元信息真实性、统计独立性或诊断效果。

## 范围

只处理振动案例检索接口中的历史参考。跨设备的历史案例可以作为参考，不能据此认定当前设备故障。当前观测、维修手册、未来计划与预测结果需要各自的用途规则，本阶段不将它们接入这条历史参考路径。

默认不启用，旧调用仍按已有的标签和候选数量规则运行。新增reference_applicability检查标记为not_applicable且required=false，不代表适用性通过。

## 直接运行演示

```bash
python -m diag_evidence demo-references --no-write
```

安装本开发副本后，也可以使用 `diag-evidence demo-references --no-write`。8个自主构造场景覆盖兼容历史、单位错配、采样率错配、预处理错配、截止时点后可用、可用时间缺失、不支持的证据角色，以及正常候选中混入不兼容案例。

保存演示结果：

```bash
python -m diag_evidence demo-references --output-dir artifacts/reference-demo
```

它使用数学合成信号和固定响应，不启动服务或加载模型。匹配预期计数不代表真实诊断准确率。

## 启用方式

在Python的DiagnosisAgent.run、run_agent或CLI输入JSON的顶层传入evidence_context，例如：

```json
{
  "decision_at": "2026-11-01T10:00:00+08:00",
  "units": "g",
  "sampling_rate": 50000,
  "preprocessing_id": "example-preprocessing-v1"
}
```

不要放入sample_info。decision_at、units、sampling_rate必填，preprocessing_id可选；额外字段被拒绝。时间须带时区和秒，接受Z或偏移量，统一到UTC比较。截止时点由调用者明确给出，不使用当前墙钟。该对象只表示调用方声明，不自动成为经认证的设备台账。

agent在外部调用前检查上下文格式，以及采样率与本次Settings配置是否一致。无效时返回ASKING_USER，外部调用为0。上下文未提供时，不自动从sample_info或检索文本猜测。

## 检索元信息

现有 `/v1/search/text` 响应每条results记录的meta需提供对应声明，例如：

```json
{
  "fault_type": "滚动体故障",
  "source_id": "example-source",
  "record_id": "example-original-record",
  "asset_id": "another-asset",
  "evidence_role": "historical_reference",
  "units": "g",
  "sampling_rate": 50000,
  "preprocessing_id": "example-preprocessing-v1",
  "window_start": "2026-10-31T09:00:00+08:00",
  "window_end": "2026-10-31T09:00:00.020480+08:00",
  "available_at": "2026-10-31T10:00:00+08:00"
}
```

检索接口未声明evidence_role时，按该接口已有的历史参考用途处理。显式声明其他角色不能用作这条路径中的参考支持；非法角色类型导致响应错误。显式和隐式的默认角色不会制造新的候选身份。

单位与采样率须与请求相同；请求指定预处理版本时也必须相同。单位只去掉首尾空白，不自动换算；数值字符串不能充当采样率。缺少必需元信息为unknown，有已知错配或非法格式为failed，两者都不计入支持。

window_end和available_at须带时区且不晚于decision_at。资料可用时间不能早于观测窗口结束。window_start可选，存在时需合法且不晚于window_end。本阶段检查历史参考的时点可用性，不是在线测点新鲜度TTL，也不把未来计划等同于未来实测值。当前不校验窗口长度、转速、设备几何、载荷或工况适用性。

## 结论与报告

只有通过检查的候选参与标签和候选数量检查。返回数量一致性仍与收到的列表核对，不能把筛掉案例误算成上游数量错误。逐条判断保存在outcomes.verify.data.reference_assessments，包含candidate_index、四态结果、原因码和允许用途。

启用后若没有任何可用参考，该必需检查为failed或unknown，支持至多low，结论至多weak。存在可用参考时检查通过，其他候选被排除；是否达到high还取决于其他所有必需检查。例如，3条兼容参考加1条单位错配参考，可以保留3条正常支持，不因被排除案例的不同标签而误报正常集合冲突。

报告接收同一显式evidence_context并独立复验，旧成功结果不能替代当前时点判断。直接调用generate_report时也应显式传入上下文。传入信号统计中已知的采样率与上下文不一致时，报告拒绝生成。

未通过和未知候选仍以外部材料形式展示，并附判定和原因。通过只表示保留下来的候选满足声明条件，不表示所有原始返回都有效。实际强度仍取请求强度和规则上限中的较低值。

策略版本为diagnosis-evidence/1.3。报告不接受旧版显式校验对象。适用性字段参与身份冲突检查，防止同一编号声明不相容元信息而静默保留其中一条；检索分数仍不参与证据身份。

## 下一步

先用真实数据和分类服务确定单位、预处理和标签映射，再验证规则是否在相同输出覆盖率下降低错误接受。真实数据、训练集检索、冻结输出和显式健康类别已接入，见[首轮真实数据实验](real_data_benchmark.md)。来源独立性、未知类别、现场设备/工况、多主体观测及预测协议仍需后续工作。当前不会为补足候选自动重新检索，不保证召回所有兼容案例。
