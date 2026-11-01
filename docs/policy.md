# DiagEvidence 诊断证据策略

当前开发策略版本：`diagnosis-evidence/1.3`。这是一个固定规则策略，用来限制工具链能够给出的结论强度。`support_level` 是规则证据支持等级。`trust_level` 保留为兼容字段，值与它相同。`high` 只说明本版本的必需检查通过，不是诊断正确概率，也不证明模型已在真实设备上得到验证。

新增可选的[历史参考适用性检查](reference_applicability.md)：传入evidence_context后只将满足声明条件的候选计入标签和数量支持。没有任何适用参考时，必需项failed/unknown将结论限制为weak。未启用时该项not_applicable且required=false，旧调用仍使用原有支持规则。报告以同一显式上下文独立复验。

## 四种检查状态

每项检查包含 `state`、`required`、`detail`。旧版 `passed` 字段继续保留，通过为 `true`，失败为 `false`，未知和不适用均为 `null`。

| state | 含义 | 对等级的影响 |
|---|---|---|
| `passed` | 已获得本项所需信息，满足规则 | 必需检查均如此时才可能为 high |
| `failed` | 已检查但不满足规则 | 按失败项降低支持等级与强度上限 |
| `unknown` | 缺少可判断的信息 | 必需项未知不能通过策略 |
| `not_applicable` | 当前流程无需该项检查 | 同时标记 required=false；不等同于已通过 |

模板流程默认不调用 LLM，`llm_prompt_text=None` 且 `llm_expected=False` 时，标签泄漏检查不适用。声明 `llm_expected=True` 却未提供有效提示词时，该项为未知。提供提示词时使用真实标签、ground truth、true label、诊断正确/错误等启发式标记检测。通过仅表示未命中这些标记，不能证明不存在泄漏。提示词不会写入报告正文。

## 检查和分级

- 分类器必须明确声明可用；`error`、`unavailable`、`invalid_input` 或无效执行状态中的残留分类字段不予采信。
- 默认faults-v1类别必须为整数 `0/1/2`；显式bearing-health-v1加入3健康。类别中文名称须与编号及声明契约一致。健康不等于未知状态。布尔值不能充当类别编号。
- 置信度必须为 `[0, 1]` 内有限实数，拒绝 NaN、Infinity、字符串和布尔值。`0.5` 是工程阈值，未经数据校准；低于它不能通过本策略。
- 候选标签检查只检查检索候选标签的一致性。旧兼容名称 `evidence_conflict` 的 `semantic` 为 `candidate_label_agreement`。多种候选标签不等于已经发现同一设备、同一时刻、同一工况下的语义矛盾。
- 分类名称须匹配至少一个候选标签；缺少标签时该检查为未知。
- 默认至少需要三个有效候选，并核对实际案例列表和本地返回数量。HTTP 检索适配器会先去重。案例条数不代表独立来源数；未验证来源独立性时 `independent_evidence_count=null`。

分类器或有效类别信息不可用时，等级为 `unavailable` 或 `low`，并禁止类别结论。关键置信度或标签泄漏检查失败、未知时，最多为 `low` / `weak`。检索支持不足、标签不一致或比对未知时，最多为 `medium` / `moderate`。所有必需检查通过时为 `high` / `strong`。合成输入始终标记为工程流程验证，不能用于报告真实诊断精度。

“规则支持等级”和“执行状态”分别保留。合法的低置信度输出可被正常解析，同时产生弱结论；服务故障、格式错误和模型未加载不会转成一个成功分类。

## 报告执行限制

报告实际结论强度为：`min(requested_strength, max_allowed_strength)`，顺序为 `no_conclusion < weak < moderate < strong`。报告返回 `effective_strength`，正文使用实际强度。调用方传入 `strong` 不能绕过较低上限。

省略 `verify_data` 时报告本地运行本版本校验；显式传入缺失、过期或失败的校验结果时禁止结论。报告还重新验证分类与检索字段，避免陈旧的校验结果为无效数据放行。LLM 使用方式与合成标记由运行参数决定，普通样本元数据不能覆盖。

外部案例与样本字段均以转义引用呈现，不作为执行指令。报告不根据外部文本生成维修承诺；真实维修决策仍需设备现场依据。文件名包含运行编号和随机后缀，使用独占创建，不能覆盖已有报告。`write_file=False` 不创建输出目录；写入失败返回正文和明确警告。

## 已知边界

策略没有估计真实正确率、来源独立性、案例语义矛盾、分布外有效性、故障严重度或剩余寿命。健康类别须显式启用，不能保证其识别能力；真实基线的健康误报结果见real_data_benchmark.md。固定阈值和启发式泄漏检测需要在后续真实数据评测中验证。项目的合成场景评测仅检查契约与失败处理是否符合规则，不能替代诊断算法效果评测。

## 主动拒绝分类（1.3）

分类服务可返回abstained=true及受支持的abstention_reason：outside_training_support、below_selection_threshold、validation_quality_unmet。此时不得同时提供类别、置信度或概率映射。适配器返回degraded，model_available=true，predicted_class=null，conclusion_strength=no_conclusion。它表示服务声明主动拒绝，和模型未加载、健康类别不同。

校验保留classifier_available的通过状态，将类别/置信度检查标为不适用，并把上限限制为no_conclusion。报告独立复验；旧成功校验不能为已拒绝的分类恢复类别。reason只接受静态代码，未声明的自由文本不进入报告。此协议不认证服务声明，也不证明拒绝方法有效。

[健康与泛化实验](health_generalization_study.md)中，候选模型未满足事先声明的验证门槛，实验接入器据此主动拒绝全部类别。覆盖率为0，不能据此宣称开放集识别或诊断准确率改善。
