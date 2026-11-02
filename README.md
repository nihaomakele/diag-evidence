# DiagEvidence Patrol

**多机器人协同巡检：在线任务分配、信息核验与可回放评估。**

当前开发版 **0.3.0.dev0**。在二维栅格中，机器人接收任务、避障移动、执行巡检，并在消息到达或机器人故障后更新派工。核验直接参与每次调度决策。

[English](README.en.md) · [调度设计](docs/patrol_scheduling.md) · [项目讲解与面试准备](docs/patrol_project_story.md) · [实验记录](NEXT_VALIDATION.md)

## 5分钟运行

Python 3.10或更高版本，在源码目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
diag-evidence patrol-demo --scenario mixed --seed 7 --output-dir artifacts/patrol
```

Windows PowerShell使用`.venv\Scripts\Activate.ps1`激活。命令输出`replay_path`：用浏览器打开该HTML，即可播放、暂停、逐步查看或拖动时间轴，同步比较三种策略。页面内含全部数据，可离线使用。也可用`python -m diag_evidence`代替`diag-evidence`。

默认示例为3台机器人、8个任务。混合扰动、种子7中，直接采用消息按时完成4项，核验后完成8项。任务轮次、位置布局与扰动脚本在三种策略间保持一致。

## 系统流程

```mermaid
flowchart LR
    O[巡检观测与消息] --> E[身份、时间、版本与冲突核验]
    E --> A[任务分配与重新派工]
    A --> P[A*避障与交通仲裁]
    P --> X[多机器人执行]
    X --> O
```

- **在线调度：**处理动态任务发布、时限、任务优先级、剩余能量和机器人临时故障。
- **算法对照：**贪心分配与精确单步匹配；后者以位掩码动态规划求解当前时刻的最大总派工收益。
- **路径与执行：**四邻域A*、路径缓存、单元格互斥、禁止迎面交换、阻塞重规划与回位。
- **信息核验：**检查任务轮次/版本、观测与可用时刻、原始记录身份，以及同一时刻的相反观测。
- **过程可追踪：**保存派工、取消、故障、重派、完工与消息处理事件，以及逐时刻状态。
- **可配置任务：**修改[任务JSON示例](examples/patrol_mission.json)，用`--config`加载自己的地图、机器人与任务。

## 对照与结果

三种信息策略共用相同分配器、物理规则与外部扰动：直接采用最新合规消息、核验后更新任务判断、直接获得真实任务状态。真实状态参照用于分析信息条件；分配器仍优化当前时刻的收益。

先执行20个开发种子，再冻结代码与设置，最后执行20个留出布局种子。8场景×2分配器×3策略×40种子，共**1920次仿真运行**，对应320组场景/种子组合。项目完整回归**340项**，其中本轮新增**32项调度检查**。

下表为匹配分配器、每场景20个留出种子的平均按时任务完成率：

| 留出场景 | 直接采用消息 | 核验后调度 | 真实状态参照 |
|---|---:|---:|---:|
| 正常通信 | 98.75% | 98.75% | 98.75% |
| 旧轮次消息 | 74.38% | 98.75% | 98.75% |
| 转发重放与身份冲突 | 86.88% | 98.75% | 98.75% |
| 相反观测 | 87.50% | 98.75% | 98.75% |
| 报告延迟 | 100.00% | 100.00% | 98.75% |
| 机器人临时故障 | 99.38% | 99.38% | 99.38% |
| 新鲜错误观测 | 87.50% | 87.50% | 98.75% |
| 混合扰动 | 49.38% | 99.38% | 99.38% |

正常与故障场景的两种策略表现接近；旧轮次、重放和冲突消息改变了候选任务集合，从而影响任务完成。延迟场景中两种策略平均复检33.3次、能量消耗221.35单位，任务确认与幂等处理是下一步工程重点。

[完整机器可读结果](docs/results/patrol_20261102.json) · [方法与假设](docs/patrol_scheduling.md)

## 重复实验

```bash
diag-evidence patrol-evaluate --split development --output-dir artifacts/patrol-development
diag-evidence patrol-freeze --development-dir artifacts/patrol-development --output artifacts/patrol-freeze.json
diag-evidence patrol-evaluate --split held_out --freeze artifacts/patrol-freeze.json --output-dir artifacts/patrol-held-out
```

评测目录须为新目录。留出评测会比对核心源码与冻结文件，记录每个场景输入摘要及逐次指标。`patrol-demo --no-write`仅输出指标。

## 工程结构

以下模块均位于`diag_evidence/`。

| 模块 | 职责 |
|---|---|
| `patrol/models.py` | 不可变任务、地图、机器人与消息契约 |
| `patrol/evidence.py` | 决策前的信息接收与核验 |
| `patrol/planning.py` | A*与贪心/精确单步分配 |
| `patrol/simulator.py` | 物理执行、消息队列与调度反馈 |
| `patrol/evaluation.py` | 配对实验、冻结与指标汇总 |
| `patrol/demo.py` | 可复现输入、轨迹与离线动画导出 |

## 版本与扩展

首版采用集中式调度、已知静态地图及可靠的机器人位置/能量状态，巡检结果经异步消息到达。移动和能耗以离散单位计算。后续按任务确认、ROS2状态/导航适配、分布式分配的顺序扩展。

原有振动诊断工具链与真实数据研究保留在`DiagnosisAgent`和`benchmarks/`，作为工具接入与证据核验的历史模块。诊断演示使用`diag-evidence demo --scenario all --no-write`；相关接口见[契约文档](docs/contracts.md)。已发布标签v0.1.1继续保留。

代码MIT；历史实验数据分别保留来源使用说明，见[数据来源](DATA_PROVENANCE.md)。问题设置参考[动态多机器人任务分配研究](https://www.roboticsproceedings.org/rss16/p068.html)，后续工程对接参考[Open-RMF](https://github.com/open-rmf/rmf_demos)。
