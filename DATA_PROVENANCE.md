# 发布资产与来源

DiagEvidence v0.1.1 研究预览源自作者原项目 `nihaomakele/harness-agent`，在其源码快照基础上修订并改名。原代码的 MIT 许可及 `Copyright (c) 2026 nihaomakele` 署名保留在 `LICENSE` 中。`diag-evidence` 是本版使用的新名称，不是原快照的仓库名。

来源记录为2026-10-15下载的原项目 `main` ZIP；当时仓库网页显示的提交为 `2190852f60d5431d80f3d42ca5c484c259019cf5`，下载归档的 SHA256 为 `d85e3b453066c09579d282e5a16637161e755d07eae9bc0bfe9f9cef1b9d8db0`。2026-10-21按该提交的Git树逐项计算并核对全部41个文件的Git blob SHA，均与本地原快照一致。此核对限于该文件树，不代表完整历史或所有权审计；原Git历史不随本发布源交付。

本次改名对应：品牌 `DiagEvidence`、拟用仓库/分发包/命令 `diag-evidence`、Python 导入 `diag_evidence`、配置前缀 `DIAG_EVIDENCE_`。历史来源名称保留用于追溯。

| 资产 | 来源 | 发布范围 |
|---|---|---|
| `diag_evidence/`代码、测试、文档 | 原 `harness-agent` 快照代码及本版修订 | MIT，保留原署名 |
| `demo.py`波形和响应、测试fixtures | 自主数学构造与控制响应 | MIT；不含设备实测数据 |
| `examples/synthetic_signal_1025.csv` | 原项目固定种子正态噪声生成脚本 | 保留原CC0-1.0数据声明 |
| `generate_synthetic_signal.py` | 原项目生成脚本 | MIT；随机种子20260930 |
| Flask替身 | 自主构造HTTP响应 | MIT；不含真实模型 |
| 先前演示权重及训练数据 | 授权来源未核实 | 不在本发布源中 |
| TimesFM权重、第三方研究PDF | 外部研究材料 | 不在本发布源中 |

合成CSV只为复现输入格式，不用于证明诊断准确率。生成脚本与CSV字节一致性以本版`VALIDATION.md`记录为准。

本发布不重新分发原有`cnn1d_demo.pt`、其训练脚本或依赖该权重的服务。重新公开旧仓库前需单独检查其历史；新建干净仓库可直接采用本发布源，避免携带未核实历史。不要把整个聊天文件夹上传：其中保留了供审查的私有原快照和研究材料。
