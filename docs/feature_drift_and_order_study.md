# 特征偏移与转速包络对照

0.2.0.dev3，2026-11-01。**角域方案在新B条件上未优于同窗口时间域对照，不替换默认基线，不作为诊断服务发布。** 本轮补上失败分析和可复现对照，未提出新的阶次跟踪算法。

## 先排查上一轮为何全部判健康

对已见渥太华A条件的480窗，分解冻结线性模型的“健康相对最强故障”原始logit差值。过零率是平均最大的正向贡献项，贡献为+6.101；目标中位标准化值为−3.392，全部480窗超出仅由HUST确定的1%—99%区间[−2.372, 2.424]。其他频带也有正负贡献，不能据此认定单一原因。

对16条HUST记录、512窗做51.2kHz→200kHz→51.2kHz重采样往返，仅3窗预测变化，健康预测数仍为128。因此，该受控扰动没有复现整体塌缩；不支持简单归因于重采样实现错误。检查未发现通道、采样率或标签映射错误。

`feature_diagnostics.py`提供线性差值的精确加性分解和源分布区间统计。这是事后描述工具；源分位区间不是校准的域外检测器，也不是因果证明。

## 固定一个物理依据明确的对照

变转速下，故障特征频率随轴转速变化。尝试用包络和每转角度对齐，检验固定频率特征是否过于依赖工况。方法在新B信号读取前固定：

- HUST使用记录内`fs`作为恒定轴频率，振动采样率为51,200Hz；它没有实测瞬时转速。
- 渥太华Channel_1是振动，Channel_2是编码器；按1024计数/转的上升沿恢复相对转数。振动从200kHz抗混叠重采样至51.2kHz。缺失、异常或超出8—45Hz支持范围的转速会报错，不补造标称转速。
- 全记录去均值/RMS归一化，四阶2—10kHz带通，Hilbert包络及1kHz低通；使用双向滤波，**离线、非因果**。
- 每记录8个互不重叠的8转窗口，记录两端各留2转。角域每转512点，每窗4096点。提取4个无量纲包络统计量与0—20阶的40个对数功率带，共44维。
- 配对时间域对照使用相同包络、相同物理窗口、相同44维结构和所选学习器，改为均匀时间采样，以固定25Hz标称转速换算横轴。**它仍用编码器确定窗口**，检验的是窗内角度对齐和阶次坐标，不是彻底移除转速信息。
- 旧冻结55维模型只取同窗口中央4096个原时间采样点作参照；其时间上下文和训练取窗不同，不能把与它的差值全部归于角度对齐。

HUST四型号各轮留一型号，仅比较类中心、C=1逻辑回归及固定随机森林。按最差型号、平均分、健康误报及固定顺序选型，入选逻辑回归；再分别在全部48条HUST记录上拟合角域和时间域模型。缩放器只从源训练数据学习，不对目标拟合或调阈值。开发选型平均87.50%、最差73.96%；这些源型号此前已见，该分数不作为新的无偏泛化成绩。源质量筛查仍未通过。

## A是开发诊断，B只作冻结后的条件检验

A已用于发现特征偏移，并影响本轮方法设计。B为此前未读取的减速工况；仍来自相同实验台和轴承状态原型，**不是新设备或新的独立数据来源**。每个条件已知四类12条记录、96窗，另有组合故障3条、24窗。窗口同记录相关，不能当作96个独立设备。

| 方法 | A窗准确率 | A记录多数票 | B窗准确率 | B记录多数票 | B故障判健康 |
|---|---:|---:|---:|---:|---:|
| 角域包络 | 44.79% | 41.67% | 29.17% | 25.00% | 79.17% |
| 配对时间域 | 39.58% | 41.67% | 35.42% | 33.33% | 72.22% |
| 旧短窗口参照 | 25.00% | 25.00% | 25.00% | 25.00% | 100.00% |

B角域模型健康误报1/24；以未校准分数≥0.5筛选时，接受93/96窗，其中65窗错误；组合故障仍接受23/24窗。高分和低健康误报都没有解决大量故障被判健康的问题，不能宣称可靠拒绝未知。

结论：A上的局部提升未在B的配对对照中成立。保留本轮作为否定性研究结果，不以此升级模型或继续对B调参。后续应先明确适用域、独立设备验证和拒绝依据；如重新研究B，应明确降为开发数据，再另留新检验数据。

[机器可读结果与署名](results/envelope_order_20261101.json)包含混淆矩阵、覆盖率及冻结摘要。首次冻结检查点的规范化JSON SHA256为`8dc73d28d4cfdad9772da0f8f6e452b4466ee9b4b2c12772a29ad9908a98d2d4`；冻结选择文件SHA256为`bebd96b209d431c6e3029613e9cdbb6237328acb9788d77c79476fd7b9b81e00`，先于B下载。安装复现会产生新时间戳，不能用复现的时间戳替代首次冻结记录；协议中的未读取声明也不是外部审计证明。

## 复现

安装`python -m pip install ".[study]"`。如无缓存，先执行`benchmark-hust --download`下载并校验48条源记录；完整的旧参照需先生成分组实验产物。输出目录必须是新目录。

```bash
diag-evidence benchmark-hust --cache-dir artifacts/data/hust --output-dir artifacts/baseline --download
diag-evidence study-hust-groups --cache-dir artifacts/data/hust --output-dir artifacts/groups
diag-evidence study-envelope-order --cache-dir artifacts/data/hust --output-dir artifacts/order --legacy-selection-dir artifacts/groups
diag-evidence evaluate-envelope-order --selection-dir artifacts/order --cache-dir artifacts/data/ottawa-a --output-dir artifacts/order-a --profile A --download
diag-evidence evaluate-envelope-order --selection-dir artifacts/order --cache-dir artifacts/data/ottawa-b --output-dir artifacts/order-b --profile B --download
```

仅需新方案与配对时间域时可省略分组实验及`--legacy-selection-dir`。源码不携带原始MAT或学习参数。数值JSON学习器内部复用`shape/learner`参数格式，真正的44维物理特征由外层`offline-order-study/v1`与`spec`限定；不可和旧55维模型交换输入。

实际复现环境：Python 3.12.14、NumPy 2.5.4、SciPy 1.18.1、scikit-learn 1.9.1。独立安装目录复用依赖，参数、逐窗预测和转速元数据与首次运行逐字节一致；不承诺不同库版本或平台的逐位一致。288项本地回归与12+8合成场景验证工程行为，不能作为诊断有效性的证明。

本实验需要完整振动记录和转速信息，**没有接入现有单信号HTTP接口**。离线预测文件保留错误类别用于分析，不表示质量筛查允许向用户作诊断结论。原始单位和真实采集时间未知时仍留空。此次同时纠正A/B清单中沿用旧流程的“编码器未使用”说明，仅改描述，文件列表、数据SHA、冻结算法及预测不变；首次结果原文和此项修订另行保留。

## 数据与方法依据

- HUST v3：Hoang Si Hong、Nguyen Thuan，河内理工大学，[数据](https://data.mendeley.com/datasets/cbv7jyx4p9/3)，[原论文](https://doi.org/10.1186/s13104-023-06400-4)，CC BY 4.0。
- 渥太华v2：Huan Huang、Natalie Baddour，[数据与编码器说明](https://data.mendeley.com/datasets/v43hmbwxpm/2)，[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6249544/)，CC BY 4.0。v2新增的状态不能全部归于原论文的v1子集。
- 数值实现：[抗混叠重采样](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)、[Hilbert变换](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html)、[双向滤波](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)。转速及采样参数来自数据说明，固定滤波带和特征是本轮假设，不代表原作者推荐。

派生表格与机器可读数据摘要按CC BY 4.0署名分享；代码MIT不替代源数据许可。处理包括重采样、转速解码、包络/阶次特征及源训练模型推理，不暗示数据作者认可结论。
