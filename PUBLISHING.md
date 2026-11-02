# 发布 DiagEvidence v0.1.1

> 本文件保留v0.1.1历史发布说明。当前0.2开发版验证见[NEXT_VALIDATION.md](NEXT_VALIDATION.md)，尚未创建新的正式版本。

本目录是研究预览发布源。GitHub源码仓库为 [nihaomakele/diag-evidence](https://github.com/nihaomakele/diag-evidence)，创建时采用私有可见性；后续可见性以GitHub显示为准。项目尚未发布到PyPI。公开范围仅为本项目源码与检查过的分发包，不包含原私有快照、研究PDF、模型实验、运行日志或聊天材料。

## 干净的来源与历史

本仓库从修正后的源码建立新历史，保留原MIT许可、作者署名和[来源记录](DATA_PROVENANCE.md)。旧`harness-agent`仓库及其历史包含授权待核实资产，应留在旧私有仓库中追溯；不将其Git历史、权重或训练数据合并过来。删除最新提交中的文件不会清除其历史副本。

## 发布检查

```bash
python -m unittest discover -s diag_evidence/tests -t . -v
diag-evidence demo --scenario all --no-write
python -m pip install build
python -m build
python scripts/check_release.py dist
```

在新环境安装wheel，并从项目目录外运行版本、全部离线场景、公共API与包资源检查，执行`pip check`。实际本地验证范围见[VALIDATION.md](VALIDATION.md)。本地通过不替代远程GitHub Actions结果；根据待发布提交的检查结果判断，首次运行前不添加通过徽章。

源码仓库上传、切换为公开、创建GitHub Release、上传PyPI是分别进行的操作。当前安装说明使用本地源码或wheel；PyPI上传成功前不宣称`pip install diag-evidence`可从PyPI安装。

## Release说明草稿

建议标签`v0.1.1`，标题`DiagEvidence v0.1.1 — Research Preview`。只有测试通过并决定发布后才创建标签与Release。

> 本版提供振动诊断工具链的契约校验、来源范围内的证据身份检查、受约束报告、独立运行记录及12个合成场景。修正重复证据计数、跨来源编号冲突、未知配置静默忽略、评测强度盲点及若干输入/网络边界。合成场景与自动化测试仅验证声明的工程行为；真实设备诊断效果、TimesFM真实推理和无人系统多智能体协同尚未得到本项目验证。

Release可附wheel、sdist、源码ZIP和SHA256摘要。分发检查只覆盖声明的文件与常见秘密形态，不是完整许可或秘密审计。下一阶段验收条件见[研究与维护路线](docs/roadmap.md)。
