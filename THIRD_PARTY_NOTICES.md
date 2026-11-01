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
| scikit-learn（study可选扩展） | 简单分类器对照 | BSD-3-Clause；各传递依赖保留自身许可 |
| SciPy（benchmark/study可选扩展） | 读取MAT、抗混叠重采样及离线包络滤波 | BSD-3-Clause及其分发内第三方声明 |

可选真实数据基线引用Hoang Si Hong、Nguyen Thuan的[HUST bearing v3](https://data.mendeley.com/datasets/cbv7jyx4p9/3)，DOI 10.17632/cbv7jyx4p9.3，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。清单保留来源与许可，原始数据不随本源码分发；窗口、归一化、特征、参数和预测为实验处理结果，分享时应保留署名、版本、许可与处理说明。原作者未因此认可本项目。源码MIT许可不重新授权该数据。

合成CSV保留原项目单独声明的[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/)数据许可；生成代码为MIT。

研究文档链接的TimesFM源码与模型权重遵循各自条款；它们不随本项目分发，不由本项目MIT重新授权。正式条款及适用版本见[Google官方源码](https://github.com/google-research/timesfm)和[3.0权重许可](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)。

本发布源不包含授权来源未核实的旧演示权重、训练数据、研究PDF或第三方截图。具体范围见`DATA_PROVENANCE.md`。

独立检验数据：Huan Huang、Natalie Baddour，[渥太华轴承数据v2](https://data.mendeley.com/datasets/v43hmbwxpm/2)，DOI 10.17632/v43hmbwxpm.2，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。仅固定文件清单随代码，原始MAT不随源码分发。预测、重采样和派生参数说明须保留数据署名与处理范围；项目MIT不替代数据许可。

`docs/results/envelope_order_20261101.json`及对应研究表格为上述两份数据的派生摘要，按CC BY 4.0署名分享，内含作者、版本、来源、许可和处理说明。MIT声明适用于代码，不覆盖此单独声明的数据许可。
