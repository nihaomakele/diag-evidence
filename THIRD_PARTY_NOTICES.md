# 第三方声明

本项目代码使用MIT，保留`LICENSE`中的署名。依赖通过包管理器单独安装，依赖自身的许可证不由MIT替代；以所安装版本附带文本为准。

| 依赖 | 用途 | 上游许可标识 |
|---|---|---|
| NumPy | 数值计算 | BSD-3-Clause及其分发内第三方声明 |
| Requests | HTTP客户端 | Apache-2.0 |
| Flask（可选） | 示例HTTP服务 | BSD-3-Clause |
| pandas（可选） | 原合成CSV生成脚本 | BSD-3-Clause |
| Matplotlib（可选） | 绘图 | 上游Matplotlib许可 |
| PyWavelets（可选） | 小波绘图 | MIT |

合成CSV保留原项目单独声明的[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/)数据许可；生成代码为MIT。

研究文档链接的TimesFM源码与模型权重遵循各自条款；它们不随本项目分发，不由本项目MIT重新授权。正式条款及适用版本见[Google官方源码](https://github.com/google-research/timesfm)和[3.0权重许可](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)。

本发布源不包含授权来源未核实的旧演示权重、训练数据、研究PDF或第三方截图。具体范围见`DATA_PROVENANCE.md`。
