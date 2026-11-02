# 发布资产与来源

0.2.0.dev4增加哈工大传感与测量实验室的[HIT-SM来源清单](https://github.com/hitwzc/Bearing-datasets)，固定提交`12e46cd1b4a40f396c2bd6e245be5bafe18b1989`，两套实验台42文件、73,110,112字节。README允许将数据用于轴承故障状态分类；清单保留该说明，数据许可SPDX标识留空。源码MIT适用于代码，原始信号通过上游获取。处理包括非重叠取窗、55维归一化特征、冻结模型推理和源记录距离筛选。公开派生统计保留机构、论文DOI、固定来源与处理说明，详见[支持范围研究](docs/source_support_screening.md)。

0.2.0.dev3另固定同一渥太华v2的B减速条件15文件，199,528,393字节，数据作者、DOI及CC BY 4.0许可不变。该条件读取前已冻结新方法、HUST训练参数及选择文件；它是同设备条件留出，不是新设备验证。新增处理为Channel_2编码器解码、抗混叠重采样、离线双向包络滤波和按转数对齐。A/B清单仅修正旧流程沿用的编码器描述，原始文件、SHA与预测不变。派生汇总见[研究说明](docs/feature_drift_and_order_study.md)与[带署名的结果摘要](docs/results/envelope_order_20261101.json)；代码不分发原始MAT、波形特征矩阵或学习参数，也不暗示数据作者认可本结果。

0.2.0.dev2新增渥太华大学Huan Huang、Natalie Baddour的Bearing Vibration Data under Time-varying Rotational Speed Conditions v2（DOI 10.17632/v43hmbwxpm.2，CC BY 4.0），固定A条件15个文件的地址、字节数及SHA256清单。原始MAT不随源码分发。处理为Channel_1抗混叠重采样200kHz至51.2kHz、取窗、去均值/RMS归一化及冻结HUST模型推理，无目标数据训练或调阈值。分享派生产物保留两份数据作者、版本、许可及处理说明，不暗示原作者认可结果。详见[分组与跨来源说明](docs/grouped_and_external_validation.md)。

本地0.2.0.dev1另固定15条6204原始记录的来源与SHA256，仍为同一HUST v3、CC BY 4.0；仅清单随源码。新增9条已知三类留出与6条组合故障压力测试，原始MAT文件不随源码发布。新候选没有替换默认基线，主动拒绝协议不构成开放集识别保证。详情见[第二轮实验](docs/health_generalization_study.md)。

本地0.2.0.dev0增加[HUST v3真实数据实验](docs/real_data_benchmark.md)。只随代码提供来源、许可、文件SHA256和下载地址清单；48条原始MAT记录及训练后参数不在源码包中。HUST数据作者为Hoang Si Hong、Nguyen Thuan，河内理工大学，DOI 10.17632/cbv7jyx4p9.3，CC BY 4.0。处理包括按型号分组、窗口截取、去均值/RMS归一化和特征计算。代码MIT不替代数据许可；分享派生产物需保留数据署名、许可及处理说明。所有人工时间场景明确标记为虚拟回放，未核实的真实采集时间保持缺失。

以下为已发布v0.1.1的历史来源记录。

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
