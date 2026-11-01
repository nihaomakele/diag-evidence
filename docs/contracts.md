# DiagEvidence 接口约定（0.2.0.dev2，本地开发）

新增顶层可选evidence_context用于[历史参考声明适用性检查](reference_applicability.md)。Python的run、run_agent、verify_result及generate_report支持该参数，CLI run从输入JSON读取；普通sample_info不能覆盖它。策略版本为diagnosis-evidence/1.3。以下HTTP接口保持原形态，适用性字段从候选meta读取；未启用该功能时不声明已验证适用性。

品牌为 `DiagEvidence`；分发包与命令为 `diag-evidence`；Python 导入为 `diag_evidence`。当前尚未发布 PyPI，下面的命令适用于安装了本地源码或 wheel 的环境。核心函数位于 `diag_evidence.hatools`。所有响应均含：

```json
{"ok": true, "status": "ok", "data": {}, "warnings": [], "errors": [], "evidence": [], "duration_ms": 0}
```

`ok`仅在`status=ok`时为真。`degraded`仍可带合法的受限制结果；`error/unavailable/invalid_input`不能作为有效分类支持。调用方必须检查状态，不应只读取残留data。

## 输入

默认`signal`为1024个有限实数的一维序列，不接受布尔值、数字字符串、嵌套数组、NaN/Inf。标准差须至少1e-9、均方能量至少1e-12，幅值不超过1e100。这些是防止空信号和数值溢出的工程边界，采样单位与真实模型预处理需由接入方确认。默认50kHz，`Settings(signal_window_size=...,sampling_rate=...)`可显式改变；不能由此假定后端支持新配置。

CLI输入文件：

```json
{"signal": [0.1, 0.2], "sample_info": {"source_file": "example"}, "input_is_synthetic": true}
```

这里只示意字段，实际长度须满足契约。可选`llm_prompt_text`与`llm_expected`仅用于检查声明的待发送文本；本程序不调用LLM。流程控制参数必须是布尔值，`sample_info`不能覆盖它们。CLI拒绝未知顶层字段和JSON中的NaN/Infinity。

## 分类服务

`POST /diagnose`接收`{"signal":[...]}`，最小有效响应：

```json
{"success": true, "diagnosis": {"predicted_class": 2, "confidence": 0.81,
 "predicted_fault": {"name": "滚动体故障", "name_en": "Ball Fault"}}}
```

类别0内圈、1外圈、2滚动体，名称若提供必须与编号一致。名称省略时使用固定契约映射。置信度为[0,1]有限实数；低值与模型不可用分开处理。`predicted_class=-1`为未加载模型的兼容标记。其他未知类别、缺失置信度或畸形对象返回结构错误，`model_available=false`。

默认`faults-v1`的可选`class_probabilities`须覆盖0/1/2全部类别，每项为有限非负数或含`probability`的对象，概率和为1并与预测类别置信度一致（容差1e-5）。空对象表示未提供。原服务任意error/note和未知字段不原文转发。`GET /`只说明进程在线。

显式配置`Settings(label_schema="bearing-health-v1")`或`DIAG_EVIDENCE_LABEL_SCHEMA=bearing-health-v1`后，增加3健康，概率映射若提供则须覆盖0/1/2/3。默认契约不接受3；未知类别仍拒绝，不映射为健康。`verify_result`与`generate_report`直接调用也须显式传入同一`label_schema`，Agent会自动传递配置。返回校验与报告携带`label_schema`。普通`sample_info`不能替换配置。健康预测不证明实际无故障。

分类器可明确主动拒绝：

```json
{"success": true, "diagnosis": {"abstained": true,
 "abstention_reason": "validation_quality_unmet", "predicted_class": null, "confidence": null}}
```

已知reason为validation_quality_unmet、outside_training_support、below_selection_threshold。不得同时返回有效类别、置信度或非空概率映射。返回model_available=true、abstained=true和no_conclusion；与predicted_class=-1表示模型未加载不同。reason为未认证的上游声明，不构成模型验证证书。报告不将拒绝解释为健康。

## 检索服务

`POST /v1/search/text`接收signal、top_k（1..100整数，默认5）、normalize、normalize_scores（默认false）。响应：

```json
{"results":[{"case_id":"case-1","score":0.73,"score_type":"cosine",
 "score_is_normalized":false,"text":"候选描述",
 "meta":{"fault_type":"滚动体故障","record_id":"record-1","source_id":"source-1"}}],"num_results":1}
```

有效条目需非空、可编码为UTF-8的文本。`case_id`（兼容`id`）在`source_id`范围内标识案例；编号可为字符串或整数，正规化后作为字符串处理。相同来源/案例身份若对应不同内容、记录或标签，整次返回`error`，不静默保留第一条。没有案例编号时，以来源、记录、稳定上下文字段和规范空白后的文本摘要确定内容身份；不同案例编号的相同记录/内容也合并。分数、rank、评分类型和链接签名不会使相同证据变成新条目。

`num_results`由本地有效列表重算，服务声明另存并核对。`independent_record_count`按`(source_id, record_id)`计算，只表示不同的服务声明身份，不证明统计独立；不同内容的多个片段仍可能来自同一记录。缺少source_id时只能在未声明来源的共同范围内去重，错误或伪造身份无法由本版自动验证。

响应最多100条原始条目，每条文本最多16384字符，保留的文本元信息与身份最多2000字符。超预算或身份冲突整次失败，避免先截断再隐藏冲突；正常校验和去重完成后才按`top_k`裁剪。

score缺失或非有限值标记不可用，类型未知为unknown，归一化状态未知为null。发出`normalize_scores=false`请求不证明服务实际返回原始分数。分数不解释为故障概率。

可保留白名单元信息：对象、通道、单位、采样率、记录/来源身份、窗口与可用时点、处理版本、证据角色和来源链接。可选evidence_context仅检查已声明单位、采样率、可选预处理版本和历史参考时点；未启用时保留这些字段供追溯，不声明已完成适用性校验。来源链接只保留HTTP(S)的主机和路径，去除userinfo、全部查询参数和片段；外部文本中识别出的HTTP(S)链接也应用此规则。路径和任意自由文本仍可能包含未识别信息，不能据此认定结果可无条件共享。外部文本在报告中转义引用，不能作为执行指令。

可选`GET /health`返回model_loaded/index_loaded布尔值与num_texts非负整数；不以HTTP在线替代能力检查。

## 配置、输出与退出码

`diag_evidence.haconfig.settings.Settings.from_env()` 读取同名大写字段加 `DIAG_EVIDENCE_` 前缀，例如 `DIAG_EVIDENCE_RAG_SEARCH_TIMEOUT`。这是新名称使用的配置前缀，原 `HARNESS_` 名称不作为新版配置入口。

| 用途 | 配置示例 |
|---|---|
| RAG 服务 | `DIAG_EVIDENCE_RAG_BASE_URL=http://127.0.0.1:8000` |
| 分类服务（可选演示替身） | `DIAG_EVIDENCE_FD_BASE_URL=http://127.0.0.1:5001` |
| 报告目录 | `DIAG_EVIDENCE_REPORTS_DIR=artifacts/reports` |
| 日志目录 | `DIAG_EVIDENCE_LOGS_DIR=artifacts/logs` |

URL需HTTP(S)，禁止在URL中嵌凭据、查询或片段；超时正有限数，重试0..10、退避0..60秒。默认输出位于当前目录`diag-evidence-output/`；CLI显式使用`--output-dir`，默认`artifacts/`。不会自动读取`.env`。

任务结果包含schema_version、run_id、input_digest（规范化信号JSON的SHA256）、终态、支持等级、实际/上限强度、工具结果、报告和日志路径。摘要值用于关联同一输入，不证明数据出处。高支持也不是模型正确性保证。

Python便利函数`run_agent`在创建实例、发请求或写文件前拒绝未知参数名，抛出不回显传入名称和值的`TypeError`。直接调用`verify_result`遇到无法表示为有限浮点数的超大整数置信度，会将置信度检查判为失败。`generate_report`对显式提供的畸形、重复名称或过期校验记录禁止结论；合法额外检查保留，必需项失败/未知按规则收紧上限，不能覆盖本地失败。

`run`退出码：0完整完成，2降级，3输入/配置问题，4关键步骤失败。`demo/evaluate`退出0表示场景全部匹配预期，1表示有不匹配，3表示输入/配置错误。

HTTP重试总尝试次数为max_retries+1；4xx立即返回，临时连接/超时/5xx按预算重试。FD默认不重试，RAG默认最多重试2次，健康探测最多1次；无无限重试。

真实HTTP客户端拒绝全部3xx，不向跳转地址转发信号。仅流式读取成功响应JSON，最多2097152字节；请求`Accept-Encoding: identity`，非identity压缩响应直接拒绝。4xx/5xx错误正文不读取、不转发。重定向、编码或字节预算违规映射为工具`error`，超时和并发额度耗尽映射为`unavailable`。

每次`http_call`有总调用时间预算：默认`min(300, timeout × (max_retries+1) + backoff × max_retries × (max_retries+1) / 2)`秒，覆盖尝试与退避。低层调用者可用`total_timeout`设0到300秒内的正数、`max_response_bytes`调低字节上限；高层Settings沿用超时与重试设置。进程内最多4个活动HTTP工作线程，额度耗尽立即失败。到期后调用方返回并发出取消信号；Python不能强制终止底层DNS/套接字阻塞，工作线程在底层操作返回后关闭响应并归还额度，不能将此解释为立即取消远端计算或收回已发送数据。自定义transport须自行实现等价的网络预算；本地RAG条目/文本检查仍会执行。

## 可选HTTP替身演示

安装`python -m pip install '.[demo]'`，在两个终端分别启动：

```bash
STUB_MODE=ok python examples/fd_stub.py
RAG_STUB_MODE=consistent python examples/rag_stub.py
```

第三个终端按README生成输入并运行，FD地址设为`http://127.0.0.1:5001`。这些替身仅模拟接口，返回自主构造响应，不计算诊断或检索。通常优先使用无需启动服务的`diag-evidence demo`。
