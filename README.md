# DiagEvidence

> **开发版：0.2.0.dev3，尚未发版。** 增加[特征偏移分析与转速包络对照](docs/feature_drift_and_order_study.md)。新B条件角域准确率29.17%，低于同窗口时间域35.42%，不替换原基线。此前[独立来源检验](docs/grouped_and_external_validation.md)也未通过；当前模型不适合实际诊断。

[English](README.en.md) · [接口](docs/contracts.md) · [校验规则](docs/policy.md) · [验证方法](docs/experiments.md)

**振动诊断工具链的契约校验、结论约束与可回放评测。**

输入振动信号，调用外部分类服务和案例检索服务，检查响应是否有效、候选标签是否一致、必需检查是否完整，再生成受规则限制的报告和运行记录。流程固定，规则可检查，默认演示不需要密钥、模型权重或网络。

GitHub已发布版本为 **v0.1.1 研究预览**；本目录为后续开发源码。这里的 `high` 是“本版本声明的规则检查通过”，不是诊断正确概率。开发版提供一个从公开实验台数据训练的简单分类基线；尚无现场设备验证或可部署诊断模型。流程是固定工具编排，尚未实现无人系统任务分配或多智能体协同决策。

## 5 分钟运行

需要 Python 3.10 或更高版本。先下载或克隆本项目，在项目目录内执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
diag-evidence demo --scenario all --no-write
```

Windows PowerShell 将激活命令替换为 `.venv\Scripts\Activate.ps1`。也可用 `python -m diag_evidence` 代替 `diag-evidence`。

安装依赖需要联网；安装完成后的 `demo` 和 `evaluate` 完全离线。输出 JSON 中的 `passed: true` 表示所有合成场景符合预期，不能当作真实诊断准确率。

保存报告、审计和评测记录：

```bash
diag-evidence demo --scenario consistent --output-dir artifacts/demo
diag-evidence evaluate --output-dir artifacts/evaluation
```

每次运行使用独立 `run_id`，生成 Markdown 报告、JSONL 轨迹和审计。`--no-write` 禁用全部产物落盘。重复运行不会覆盖已有报告。

## 已提供的功能

- 输入检查：一维、长度、有限数值、恒定/近零能量及数值溢出；无效输入不调用外部服务。
- 分类响应契约：类别与名称、有限置信度和范围、可选概率字段；服务在线与分类能力分开处理。
- 检索响应整理：按返回内容核对有效条数、去重、保留分数声明，未知分数类型不解释成概率。
- 版本化规则：`passed / failed / unknown / not_applicable`，必需检查未知时限制支持等级。
- 报告约束：实际强度取请求值与规则上限中较低者；外部文本转义引用；准确描述标签泄漏检查状态。
- 独立运行记录：信号摘要和输入摘要值，不保存原始信号/提示词/完整检索文本到日志；递归脱敏并报告写入失败。
- 12 个自主构造场景、真实本机 HTTP 集成测试、构建安装检查和 GitHub Actions 配置。
- 可选四类契约：`Settings(label_schema="bearing-health-v1")`加入健康类别3，默认三类契约保持原样。健康不等于未知状态。
- 真实数据实验：固定48条来源及SHA256，按轴承型号划分后取窗，训练集专用分类与检索，相同工具输出下比较规则和覆盖率。

场景包括一致候选、候选标签多样、低置信度、无结果、服务失效、缺失类别、NaN/越界置信度、非有限输入、重复记录、标签泄漏、缺失提示词。

## Python 接口

```python
from diag_evidence import DiagnosisAgent, Settings
from diag_evidence.demo import SyntheticTransport, synthetic_signal

agent = DiagnosisAgent(
    config=Settings(),
    transport=SyntheticTransport("consistent"),
    persist_trace=False,
)
result = agent.run(synthetic_signal(), skip_services_check=True)
print(result["terminal_state"], result["support_level"])
```

传入自己的 `transport` 或不传以使用真实 HTTP 服务。真实服务必须满足[接口约定](docs/contracts.md)。`Settings` 为每个实例独立配置，环境变量在创建时读取；程序不自动加载 `.env`。

真实 HTTP 入口示例：

```bash
python examples/make_input.py --out artifacts/input.json
export DIAG_EVIDENCE_RAG_BASE_URL=http://127.0.0.1:8000
export DIAG_EVIDENCE_FD_BASE_URL=http://127.0.0.1:5001
diag-evidence run artifacts/input.json --output-dir artifacts/run
```

这条命令需要提前启动后端；项目自带的可选 Flask 替身只用于接口演示，安装方法和启动命令见[接口文档](docs/contracts.md)。程序不上传模型、不自动连接付费 LLM。

## 测试与构建

```bash
python -m unittest discover -s diag_evidence/tests -t . -v
python -m pip install build
python -m build
python scripts/check_release.py dist
```

测试包含真实本机随机端口服务，运行环境需允许本机回环连接；不连接公网。CI 配置为 Python 3.10—3.13，首次远程运行状态由 GitHub Actions 显示；这里不使用未经运行的通过徽章。本次实际验证环境与结果见 [VALIDATION.md](VALIDATION.md)。

## 研究方向与边界

项目可以作为“平台状态评估 → 决策约束”中的工具层原型。多智能体协同需要另外定义局部观测、通信、任务与约束，并以单流程和其他决策方法为基线。现有代码不足以证明协同收益。

导师提出的 TimesFM 3.0 对应数值预测研究，后续实验入口和冻结协议见 [TimesFM 研究说明](docs/timesfm_research.md)。它的预测值不能直接解释为故障概率或剩余寿命。本版核心无 TimesFM 依赖；实际模型实验结果与合成契约测试分别记录。

后续优先补真实任务数据与记录级划分、独立证据身份与时间适用性，再比较校验规则的收益和输出覆盖率。可执行的阶段目标和验收条件见[研究与维护路线](docs/roadmap.md)，详细限制见 [limitations](docs/limitations.md)。

## 许可与贡献

首个公开版本统一使用项目名 `DiagEvidence`、发行包和命令 `diag-evidence`、Python 导入 `diag_evidence`、配置前缀 `DIAG_EVIDENCE_`。之前的本地原型名 `Harness Agent` 仅保留在来源记录中。项目尚未发布到 PyPI，上面的安装命令从本地源码安装。PyPI 上的 [`harness-agent`](https://pypi.org/project/harness-agent/) 属于其他项目。

项目代码使用 [MIT](LICENSE)，保留原作者署名。发布源不包含先前来源授权待核实的演示权重或训练数据；资产范围见 [DATA_PROVENANCE.md](DATA_PROVENANCE.md)。第三方依赖和外部模型遵守各自条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

欢迎提交带最小复现的缺陷报告、测试或文档改进。参见 [CONTRIBUTING.md](CONTRIBUTING.md)、[SECURITY.md](SECURITY.md) 和 [CHANGELOG.md](CHANGELOG.md)。
