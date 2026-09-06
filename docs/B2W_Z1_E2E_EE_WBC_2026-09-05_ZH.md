# B2-W + Z1 端到端 EE-WBC：第一阶段实现与交接

日期：2026-09-05  
分支：`codex/b2w-z1-e2e-ee-wbc`  
远端工作目录：`/home/mingqian/LeggedManip_Lab-e2e-ee-wbc`

## 1. 当前结论

已经建立一个与原 Hybrid 控制器并行存在的端到端 EE tracking 训练任务。这里“端到端”指：一个 PPO actor 直接接收末端轨迹与本体状态，并同时输出腿、机械臂和轮子的低层目标；它不再调用冻结的 B2-W locomotion policy，也不再调用 Z1 differential IK。

训练基础设施、100 iterations 初训和三轮稳定性微调均已完成。当前最佳 checkpoint 是 Stage 1d 的 `model_324.pt`：在 seed 42/43/44、共 192 个确定性首回合中存活 190 个（98.96%），所有存活环境均满足 3 cm / 6°，三组最终误差均值约为 3.31 mm / 1.09°。因此 **Stage 1 nominal 短程 EE tracking 已通过既定鲁棒性门槛**。

这仍不是开门策略或真机策略：当前目标只有 3–12 cm 平移和 ±0.15 rad yaw，尚未加入大范围移动、低位倾斜、随机化、视觉和门接触。下一步应进入 Stage 2 扩展工作空间，而不是直接跳到真机。

旧 Hybrid 控制器仍应保留，作为精度/安全基线和以后 residual controller 的候选后端。当前最重要的工程结论是：不要用训练窗口中的平均 episode length 或单个成功视频替代确定性批量评估。

## 2. 控制结构

策略频率为 50 Hz，物理与 PD 执行频率为 200 Hz。

actor 的 22 维动作顺序固定为：

1. 12 个腿关节位置残差：FR、FL、RR、RL，每条腿 hip/thigh/calf；
2. 6 个 Z1 关节位置残差：joint1 到 joint6；
3. 4 个轮速：FR、FL、RR、RL。

夹爪不属于这 22 维动作。第一阶段通过零维 `gripper_hold` action term 保持在 reset 位置；进入门任务后再增加独立的离散开/合命令或 grasp state machine。

actor 输入为 216 维：108 维单帧 deployable observation 的两帧历史。内容包括本体速度、重力方向、腿/臂关节状态、轮速、当前 TCP 6D 位姿误差、最终 TCP 6D 位姿误差、期望 6D twist、0.10/0.25/0.50 秒未来轨迹预览、轨迹进度和上一步 22 维动作。

critic 输入为 243 维：actor 的 216 维输入，再加 27 维仿真特权信息，包括已知可行 ghost-root 的平面误差、22 个关节力矩、轮地接触数量和最终 TCP 距离。部署时不需要这 27 维信息。

## 3. 目标坐标系与 reset 流程

最终目标建立在世界坐标系中，而所有策略误差都转换到当前 root frame 表达，因此机器人移动后目标不会跟着本体漂移。

每个 episode 先从当前有效 Z1 姿态的 FK 构造一个已知存在全身解的目标：对虚拟 root 采样平面位移和 yaw，再把当前 TCP 相对 root 的位姿组合到该虚拟 root 上。第一阶段范围是 3–12 cm 平移、±0.15 rad yaw。

原始资产从 0.65 m reset 后会自然落到约 0.503 m，期间腿和臂会有负载挠曲。如果在 reset 瞬间冻结世界目标，会人为制造约 15 cm 的初始 TCP 误差。现在采用以下流程：

1. 前 2 秒保持 minimum-jerk progress 为 0；
2. 这 2 秒内持续用当前物理状态重新锚定 TCP 起点和可行最终目标；
3. 2 秒结束后冻结世界目标；
4. 随后用 2.5 秒 minimum-jerk 轨迹运动到目标。

这相当于真机的“先进入站立状态，再接受任务轨迹”。它也形成了一个自然课程：随机初始 policy 先学会保持稳定，能存活超过 2 秒后才开始面对运动 tracking。

## 4. 奖励和安全项

任务奖励包括：

- TCP position 的 coarse/fine 指数核；
- TCP orientation 的 coarse/fine 指数核；
- 6D twist tracking；
- 最终 3 cm / 6° 成功奖励；
- 轨迹后段的 Z1 home posture 正则，使移动底盘而不是一直折叠机械臂成为有利解；
- 机械臂关节限位、动作变化、腿/臂/轮动作幅值、力矩、轮子侧滑、本体高度/姿态/竖直速度和到位后速度惩罚。

终止条件包括低本体、超过 35° 倾角，以及 base、小腿、lidar、Z1 links 或 gripper 的非期望接触。训练资产开启 self-collision。当前阶段关闭执行器延迟和动力学随机化，避免在 nominal policy 尚未建立时混入 sim-to-real 难度。

## 5. 已完成验证与训练结果

最终验证全部使用 RTX 4090、Isaac Sim 5.1 / Isaac Lab 对应环境：

| 验证 | 结果 | 关键数据 |
|---|---|---|
| 16 env，零动作 300 steps（6 s） | PASS | 最低 root 0.5035 m；最大倾角 6.00°；无 NaN、碰撞或 reset |
| 16 env，22 维映射 20 steps | PASS | 12/6/4 三组全部收到非零目标；无 reset |
| 1024 env，零动作 20 steps | PASS | policy/critic batch 分别为 1024×216、1024×27；无 NaN 或 reset |
| 256 env，PPO 2 iterations | PASS | actor 216→22、critic 243→1；16384 steps；checkpoint 正常写出 |

两次 PPO 更新只验证 rollout、反向传播、观测分组和保存链路。第二次更新的吞吐约 4235 steps/s。

随后完成两段正式训练：

| 阶段 | 训练目录 | 训练量 | 结果 |
|---|---|---:|---|
| Stage 1 nominal | `2026-09-05_23-12-01_stage1_nominal_100` | 100 iterations / 3.277M steps / 229 s | tracking 已形成，但普遍通过降低本体换取可达性 |
| Stage 1b height | `2026-09-05_23-25-26_stage1b_height_200` | 从 model 99 续训 200 iterations / 6.55M steps / 460 s | 加强高度和终止惩罚；中途 checkpoint 优于最终 checkpoint |
| Stage 1c safety | `2026-09-05_23-46-37_stage1c_safety_100` | 从 model 200 续训 100 iterations / 3.28M steps / 235 s | 加入单边高度 barrier、接触惩罚；model 250 达到三 seed 90.1% 存活 |
| Stage 1d scaled safety | `2026-09-05_23-56-53_stage1d_safety_scale_75` | 从 model 250 续训 75 iterations / 2.46M steps / 176 s | 按 0.02 s 奖励积分缩放安全代价；model 324 通过 Stage 1 门槛 |

4090 上四段训练累计约 18.3 分钟（不含启动、评估和视频渲染）。

固定 `seed=42`、64 environments、599 steps（11.98 s）、确定性 actor 的首回合评估如下。误差只统计仍存活的环境，因此必须与存活率一起阅读：

| checkpoint | 完整存活 | 主要终止原因 | 存活者最终位置误差 | 存活者最终姿态误差 | 3 cm / 6° 成功率（存活者） |
|---|---:|---|---:|---:|---:|
| Stage 1 `model_99` | 9/64 = 14.1% | `low_base` 55 | 8.64 mm | 5.65° | 66.7% |
| Stage 1b `model_200` | **31/64 = 48.4%** | `undesired_contact` 32，`low_base` 1 | **5.54 mm** | **1.24°** | **100%** |
| Stage 1b `model_298` | 21/64 = 32.8% | `low_base` 43 | 5.62 mm | 1.05° | 100% |
| Stage 1c `model_250` | 60/64 = 93.8% | `undesired_contact` 4 | 5.14 mm | 1.33° | 98.3% |
| Stage 1d `model_324` | **64/64 = 100%** | 无 | **3.33 mm** | **1.05°** | **100%** |

碰撞诊断将 Stage 1b 的失败定位到 `lidar_link`/Z1 `link2`，其次是前腿 calf 与 gripper。Stage 1c 加入连续接触惩罚和归一化单边高度 barrier；Stage 1d 进一步考虑 Isaac Lab 会用 0.02 s 控制周期积分奖励，将接触权重设为 -50、终止权重设为 -200。最终 `model_324` 在 seed 42/43/44 的结果分别为 64/64、62/64、64/64，聚合 190/192 = **98.96%**；全部 190 个存活环境均通过 3 cm / 6°。

选择 `model_324.pt`，而不是任何训练段的最后 checkpoint。它证明当前任务下端到端策略可以同时学到准确短程 6D EE tracking、全身协调和 nominal 安全性。

### 5.1 平面大范围课程（2026-09-06 已通过）

在 Stage 1 安全基线之上，命令生成器增加了 30% 的旧工作空间 replay，并逐级扩大目标半径。每一级都先做 zero-shot 测试；低于门槛才继续 PPO 微调，最终仍以固定首回合、多随机种子评估选择 checkpoint。

| 课程 | 热启动模型 | 是否训练 | 最佳模型 | 三 seed 存活 | 幸存者 3 cm / 6° | 动态位置误差均值 |
|---|---|---:|---|---:|---:|---:|
| Stage 2: 12–20 cm | Stage 1 `model_324` | 否，zero-shot 已通过 | `model_324` | 188/192 = 97.92% | 100% | 3.57 mm |
| Stage 3: 20–30 cm | Stage 1 `model_324` | 100 iterations | `model_350` | 190/192 = 98.96% | 100% | 5.36 mm |
| Stage 4: 30–50 cm | Stage 3 `model_350` | 100 iterations | `model_449` | 187/192 = 97.40% | 100% | 6.95 mm |
| Stage 5: 50–70 cm | Stage 4 `model_449` | 100 iterations | **`model_525`** | **192/192 = 100%** | **100%** | **5.17 mm** |

Stage 5 的平均底盘平移约 0.226 m，三个 seed 中观察到的最大底盘平移约 0.593 m，说明策略已经学会在目标超出机械臂舒适工作空间时同步移动 B2-W，而不是只把 Z1 拉到极限。Stage 5 零样本只有 49/64 存活；微调后达到 192/192，说明逐级课程与旧目标 replay 都是必要的。

当前新的平面工作空间最佳 checkpoint：

```text
/home/mingqian/LeggedManip_Lab-e2e-ee-wbc/logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_00-37-29_stage5_radius70_100/model_525.pt
```

### 5.2 低位 TCP 与本体倾斜课程（2026-09-06 已通过）

最初把所有 ghost-root 平移直接改成重力对齐坐标，破坏了 Stage 5 的已学分布；该尝试已停止。最终采用向后兼容的命令定义：平面 replay 保持原几何不变，仅对明确抽中的 spatial 样本把 TCP 世界 z 锁定为相对落稳 TCP 的负偏移。评估器也增加了 spatial/planar 分组完成率和相对落稳姿态的 root-pitch 变化。

低位课程先从 -0.5～-1.5 cm、20% 样本开始，并把底盘高度 barrier 权重从 -4 增到 -12、flat-orientation 权重从 -1 降到 -0.2。`model_625` 在 seed 42/43/44 共 384 环境中达到 384/384，其中低位 75/75、平面 309/309。之后将低位目标限制在面向开门任务的前方 ±0.60 rad 扇区，并逐级 zero-shot 扩展：

| 低位范围 | 训练 | 三 seed 总体存活 | 低位存活 | 平面存活 | 低位平均额外前倾 |
|---|---:|---:|---:|---:|---:|
| -0.5～-1.5 cm | 175 iterations（两段） | 384/384 | 75/75 | 309/309 | — |
| -1.5～-4 cm | 0 | 380/384 | 83/87 | 297/297 | — |
| -4～-8 cm，前方扇区 | 0 | 384/384 | 100/100 | 284/284 | 约 +1.8° |
| -8～-12 cm，前方扇区 | 0 | 384/384 | 123/123 | 261/261 | 约 +2.39° |
| -12～-18 cm，前方扇区 | 0 | 374/384 = 97.40% | 129/139 = 92.81% | 245/245 | 约 +3.24° |

当前低位/平面统一最佳 checkpoint 仍为首级课程的 `model_625`；更深的目标靠任务结构泛化通过，没有额外覆盖这个稳定模型：

```text
/home/mingqian/LeggedManip_Lab-e2e-ee-wbc/logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_01-13-13_stage6_low15mm_prob20_refine75/model_625.pt
```

### 5.3 门资产与把手预抓取对齐（Stage 11，2026-09-06 已通过）

已加入每个环境独立复制的铰接门：18 kg 门板、竖直门轴、可向下旋转的 lever handle，以及位于握持中心的 `handle_grasp` 标记。独立动力学测试确认把手和门轴均可在力矩下旋转，状态无 NaN/Inf。

Stage 11 没有改变 actor：仍由同一个 216 维观测、22 维动作的策略同时控制 12 个腿关节、6 个 Z1 关节和 4 个轮子。门把手只生成世界坐标 TCP 命令。为隔离问题，目标先设在把手局部 x 负方向 10 cm 的无碰撞预抓取点，并对门目标施加 x ±1.5 cm、y ±2.5 cm、z ±2 cm 扰动；目标姿态暂时保持落稳时 TCP 姿态。

直接加载 `model_625` 做 zero-shot，seed 42/43/44、每个 128 环境的结果为：

| 指标 | 结果 |
|---|---:|
| 首回合存活 | **384/384 = 100%** |
| 幸存者 3 cm / 6° 成功率 | **100%** |
| 最终位置误差 | 平均 **4.80 mm**，最大 **5.50 mm** |
| 最终姿态误差 | 平均 **0.243°**，最大 **0.383°** |
| 底盘平面移动 | 平均 **0.468 m**，最大 **0.503 m** |
| 低底盘 / 坏姿态 / 意外碰撞终止 | **0 / 0 / 0** |

这一步证明现有 EE-WBC 能把动态物体坐标系中的把手点转换为全身协同目标，并在手臂单独不可达时主动移动底盘。下一步不应立即训练整套开门，而应先把 10 cm 预抓取偏移逐渐缩到 2–3 cm，检查夹爪与把手的真实几何，再加入脚本化闭合和抓持判据。

### 5.4 近接触、抓持与压把手（Stage 12–14，2026-09-06 已通过）

Stage 12 把目标从把手前 10 cm 缩到 3 cm，并将 Z1 夹爪保持在 -1.45 rad 打开位。seed 42/43/44 共 384 个环境全部存活且通过 tracking；未接触时把手由 12 N·m/rad 回位弹簧保持在约 0.019 rad，而不是靠重力自动落下。

直接把 TCP 目标放在实体把手中心会形成约 2 cm 的不可实现穿透误差，WBC 会持续向门内顶并导致本体漂移。因此 Stage 13 使用相对把手 -2 cm 的接触一致目标：夹爪只在轨迹进度超过 90%、TCP 进入 4 cm 捕获区后，用 1.5 s 从 -1.45 rad 平滑闭合到 -0.35 rad。该设置可稳定夹住 50 mm 把手而不产生最初约 100 N 的过大预紧力。

Stage 14 在闭合后把 TCP 沿把手圆弧近似向下 7.5 cm、横向 1.5 cm。`model_625` 未再训练即完成 3×128 环境验证：384/384 存活，最终把手角平均约 0.359 rad（20.6°），TCP 位置误差平均约 1.38 cm，无非法接触终止。

### 5.5 完整拉门（Stage 15，2026-09-06 已通过）

加入了有状态锁舌：把手超过 0.30 rad 前，门轴由 500 N·m/rad 的 PD 锁在 0；超过阈值后本 episode 永久释放。随后 TCP 目标相对压把手终点向后 22 cm、横向 13 cm，引导 B2-W 后退并沿门扇圆弧拉动。

当前 Z1 USD 的夹爪是粗略单自由度模型，TCP 也尚未真机标定。纯网格摩擦在拉门时会滑脱，因此最终仿真使用“有效闭合后激活”的双向柔顺抓持代理：把 TCP—把手误差转换为有限弹簧力，把对应广义力矩施加到把手轴/门轴，同时在机器人 `gripper_stator` 的 TCP 位置施加等大反力。它保留门负载对本体的作用，不是把门运动学瞬移到手上；但在真机几何标定前，不应把它当作精确接触 sim-to-real 结论。

门资产还修复了两个会造成假结果的问题：`handle_grasp` 标记 link 显式设为 1 g，避免导入器默认质量把把手压下；门板改为高 1.98 m、中心 z=1.02 m，留出 3 cm 门底缝，避免 18 kg 门板与地面摩擦造成不同并行环境随机卡门。

最终任务 `B2W-Z1-EE-WBC-Door-Pull-Play-v0` 加载原 `model_625`，seed 42/43/44、每 seed 128 环境：

| 指标 | 结果 |
|---|---:|
| 首回合存活 | **384/384 = 100%** |
| 完整开门成功 | **384/384 = 100%** |
| 最终门角 | 平均 **0.4175 rad (23.9°)**，最小 **0.4078 rad** |
| 最终把手角 | 平均 **0.3021 rad**，最小 **0.2743 rad** |
| TCP—把手距离 | 平均 **3.59 cm**，最大 **4.01 cm** |
| 最终 TCP 位置误差 | 平均 **1.95 cm**，最大 **2.20 cm** |
| 最终 TCP 姿态误差 | 平均 **2.57°** |
| 柔顺抓持最大力 | 跨环境平均 **21.7 N**，最大 **29.4 N** |
| 低底盘 / 坏姿态 / 非法碰撞终止 | **0 / 0 / 0** |

本阶段实现的是“已知门把手位姿 + 已训练 EE-WBC + 脚本化任务相位”的自主执行基线，还不是视觉端到端开门策略。下一研究阶段应把脚本相位替换为可学习高层策略，并把真实腕部相机估计、接触/力判据和准确夹爪/TCP 标定接入。

一个失败的 Stage 8 标准 PPO 试验显示：30% 的 360° 低位目标会导致平面遗忘和姿态冲突。最终版本将低位目标放到前方扇区，并提供低位专用 pitch shaping 与保守 PPO 配置（1e-4 学习率、0.1 clip、0.001 entropy），供后续更深目标或门接触微调使用。当前 Stage 8–10 因 zero-shot 已过门槛，没有不必要地继续训练。

训练窗口在 `model_298` 附近曾显示 mean episode length 约 583、timeout 约 95.6%，但确定性首回合测试只有 32.8% 存活。差异来自训练指标的滚动/随机分布与 reset 混合，而确定性评估严格追踪同一批环境的第一次 episode。以后 checkpoint 选择一律以后者为准。

还额外做过两项失败诊断并保留结果：

- 直接使用 RobotLab actuator preset 而不运行它的冻结 locomotion policy，零动作会反复触发 `low_base`；因此它不适合作为第一阶段的自然稳定 plant，应作为后续 actuator-transfer 阶段。
- 只把 root 高度设置成落稳后的 0.505 m，而不同时匹配姿态和关节状态，会导致初始几何接触；因此没有采用这种不完整 reset。

### 5.6 项目目标校正与动态 6D EE tracking（Stage 16，2026-09-06）

本项目的最终目标已经明确校正为：**通用、稳定、姿态合理的端到端 whole-body 6D EE/TCP tracking WBC**。开门只保留为回归场景，不再作为策略设计中心。低层 actor 仍一次输出 22 维动作，同时控制 12 个腿关节、6 个 Z1 关节和 4 个轮子；不存在“机械臂 IK + 独立底盘控制器”的人工切换。

新增 `PeriodicWorldPoseCommand` 后，同一策略可在 episode 内连续接收并预览下列世界坐标 6D 目标：

- 固定点、x/y 直线、xy 圆、xy 8 字、竖直扫描；
- yaw 扫描、位置和姿态同时变化的 6D 轨迹；
- 30–70 cm 任意方位平面 waypoint；
- 前方低位 waypoint。

轨迹采用 minimum-jerk 相位插值并向 actor 提供当前目标和短时 preview。训练分布中 60% 为大范围/低位 waypoint replay，40% 为连续轨迹，防止只训练平滑小轨迹后遗忘大范围本体移动。

#### 初始化四腿歪斜的诊断

截图中的四腿歪斜不是关节没有锁死。腿部使用 `DelayedPDActuator`，当前 stiffness=250、damping=5；落稳后 calf 因承重产生约 0.16–0.26 rad 静态偏差属于正常柔顺。地面摩擦系数 0.8 会影响是否滑移，但不是形成固定不自然站姿的根因。

真正原因是旧 `model_625` 的奖励只要求 TCP tracking、存活和基本姿态，没有显式约束自然站姿、左右对称和髋关节偏置。策略因此找到“撑宽四腿换取稳定”的 reward loophole。确定性测量显示，它在完全相同 reset 下每次都会产生相似偏置，最严重髋关节偏离默认值约 0.29 rad（16.5°），平均约 0.10 rad（5.9°）；这说明是策略主动输出，而不是关节自由下落或随机摩擦滑散。

修复没有锁死关节，也没有盲目增大 PD 或摩擦。新增奖励为：

- 落地/静态阶段腿关节回到默认姿态；
- 髋关节默认姿态与左右镜像对称；
- root roll 和 roll rate；
- 低位目标继续保留按目标高度逐渐增加的 pitch shaping。

因此，普通保持时腿趋向自然；当目标确实不可达时，策略仍可移动/转向底盘、改变腿长并俯仰本体。

#### 动态基准与 checkpoint 选择

新增 `scripts/standalone/b2w_z1_dynamic_ee_benchmark.py`，从落地完成后开始统计每种轨迹的存活、位置/姿态误差、3 cm/6° 达标率、速度误差、相位滞后、TCP/底盘速度、底盘加速度、roll/pitch、动作变化和落地髋偏差。checkpoint 选择优先级固定为：存活与非法碰撞 > 大范围能力 > tracking 误差 > 姿态美观。

从原 `model_625` 直接以 60% 大范围 replay、5e-5 学习率微调 100 iterations，得到 Stage 16d。候选结果如下：

| checkpoint | 三 seed 混合场景存活 | 位置误差均值 / p95 | 姿态误差均值 | 落地最大髋偏差 | 结论 |
|---|---:|---:|---:|---:|---|
| 原 model_625 | seed 42 为 256/256 | 7.25 / 13.88 mm | 0.94° | 约 16.5° | 稳定但站姿不自然 |
| model_700 | **768/768** | 约 4.7–5.1 / 9.6–9.9 mm | 约 1.0° | **约 12.0°** | **默认主模型** |
| model_711 | 766/768 | 约 6.1–6.5 / 10.9–11.6 mm | 约 0.9–1.0° | 约 10.4° | 拒绝：安全性和误差均退化 |
| model_724 | 766/768 | 约 5.3–5.5 / 9.6–10.2 mm | 约 1.1° | **约 9.0°** | 姿态更自然的实验模型，不作为默认 |

默认 `model_700` 已完成 0.1/0.2/0.35 Hz、seed 42/43/44、每项 64 环境的速度基准，总计 **576/576** 存活：位置误差均值分别为 **3.32/4.81/9.26 mm**，p95 为 **6.85/9.94/23.22 mm**，姿态误差均值为 **0.88/1.07/1.42°**，平均相位滞后为 **16.7/20.0/33.3 ms**。该 PLAY 配置关闭随机扰动且按 env index 分配轨迹，因此三个 seed 的数值完全一致；多 seed 的随机性检验由上面的混合训练配置完成。结果说明当前 actor 能连续 tracking，而不仅是 episode 内一次到点。

`model_700` 的旧任务回归：Stage5 128/128、Stage6 128/128、门前对齐 128/128；Stage10 极低前方目标为 77/128，失败来自 gripper/lidar 邻域非法接触。完整拉门虽然 128/128 不倒且全部解锁把手，但开门成功率退化为 0。因此 checkpoint 必须按任务分开保存：

- 通用动态 EE tracking 默认模型：Stage16d `model_700`；
- 更自然姿态研究对照：Stage16d `model_724`；
- 当前脚本化开门回归模型：Stage6 `model_625`。

目前剩余的主要 WBC 问题不是普通目标 tracking，而是极端低位目标下的自碰撞约束、不可达边界附近更自然的姿态选择，以及把当前 PD plant 迁移到真机执行器模型。

## 6. 如何运行

新 worktree 未安装到共享 conda 环境，运行时要让本分支源码优先：

```bash
cd /home/mingqian/LeggedManip_Lab-e2e-ee-wbc
export PYTHONPATH="$PWD/source/LeggedManip_Lab:${PYTHONPATH:-}"
PY=/home/mingqian/miniforge3/envs/env_isaaclab51/bin/python
```

复跑 1024 环境冒烟测试：

```bash
$PY scripts/standalone/b2w_z1_e2e_ee_wbc_smoke.py \
  --task B2W-Z1-EE-WBC-Flat-Play-v0 \
  --headless --device cuda:0 --num_envs 1024 --steps 20 --mode zero
```

当前 Stage 1 最佳 checkpoint：

```text
/home/mingqian/LeggedManip_Lab-e2e-ee-wbc/logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-05_23-56-53_stage1d_safety_scale_75/model_324.pt
```

当前 70 cm 平面 EE tracking 最佳 checkpoint：

```text
/home/mingqian/LeggedManip_Lab-e2e-ee-wbc/logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_00-37-29_stage5_radius70_100/model_525.pt
```

当前低位 + 平面统一最佳 checkpoint：

```text
/home/mingqian/LeggedManip_Lab-e2e-ee-wbc/logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_01-13-13_stage6_low15mm_prob20_refine75/model_625.pt
```

当前通用动态 6D EE tracking 默认 checkpoint：

```text
/home/mingqian/LeggedManip_Lab-e2e-ee-wbc/logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_09-41-30_stage16d_mixed_from625_lr5e5_100/model_700.pt
```

Windows/WSL 本地备份（不提交 GitHub）：

```text
implementation/e2e_ee_wbc/artifacts/stage16_dynamic_ee_wbc_model_700.pt
implementation/e2e_ee_wbc/artifacts/stage16_dynamic_ee_wbc_posture_model_724.pt
implementation/e2e_ee_wbc/artifacts/stage16_dynamic_sixd_model_700.mp4
implementation/e2e_ee_wbc/artifacts/stage16_unreachable_waypoint_model_700.mp4
```

复跑连续轨迹与混合工作空间基准：

```bash
$PY scripts/standalone/b2w_z1_dynamic_ee_benchmark.py \
  --task B2W-Z1-EE-WBC-Dynamic-v0 \
  --checkpoint logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_09-41-30_stage16d_mixed_from625_lr5e5_100/model_700.pt \
  --headless --device cuda:0 --num_envs 256 --steps 1099 --seed 42 \
  --report docs/validation/e2e_ee_wbc/stage16d_mixed_model_700_seed_42_eval.json
```

可视化连续 6D EE tracking：

```bash
$PY scripts/rsl_rl/play.py \
  --task B2W-Z1-EE-WBC-Dynamic-SixD-Play-v0 \
  --device cuda:0 --num_envs 1 \
  --checkpoint logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_09-41-30_stage16d_mixed_from625_lr5e5_100/model_700.pt
```

可视化超出手臂单独可达范围后，机器人主动移动本体：

```bash
$PY scripts/rsl_rl/play.py \
  --task B2W-Z1-EE-WBC-Dynamic-Waypoint-Play-v0 \
  --device cuda:0 --num_envs 1 \
  --checkpoint logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_09-41-30_stage16d_mixed_from625_lr5e5_100/model_700.pt
```

复现固定批量评估：

```bash
$PY scripts/standalone/b2w_z1_e2e_ee_wbc_eval.py \
  --task B2W-Z1-EE-WBC-Flat-Play-v0 \
  --headless --device cuda:0 --num_envs 64 --steps 599 --seed 42 \
  --checkpoint logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-05_23-56-53_stage1d_safety_scale_75/model_324.pt \
  --report docs/validation/e2e_ee_wbc/stage1d_safety_model_324_eval.json
```

若需从头复现第一段训练：

```bash
$PY scripts/rsl_rl/train.py \
  --task B2W-Z1-EE-WBC-Flat-v0 \
  --headless --device cuda:0 --num_envs 1024 \
  --max_iterations 100 --run_name stage1_nominal
```

训练后可视化：

```bash
$PY scripts/rsl_rl/play.py \
  --task B2W-Z1-EE-WBC-Flat-Play-v0 \
  --device cuda:0 --num_envs 1 \
  --load_run <训练目录名> --checkpoint <checkpoint文件名>
```

复现完整的接近、闭合夹爪、旋转把手和后退拉门回放：

```bash
$PY scripts/rsl_rl/play.py \
  --task B2W-Z1-EE-WBC-Door-Pull-Play-v0 \
  --device cuda:0 --num_envs 1 \
  --checkpoint logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_01-13-13_stage6_low15mm_prob20_refine75/model_625.pt
```

确定性单环境录像没有提交到 GitHub（该 fork 拒绝新的 LFS 对象）。当前保存位置为：

```text
4090 远端：logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_01-13-13_stage6_low15mm_prob20_refine75/videos/play/rl-video-step-0.mp4
Windows/WSL 本地镜像：implementation/e2e_ee_wbc/artifacts/stage15_complete_door_open_model_625.mp4
```

## 7. 下一步顺序

### 阶段 1：nominal EE tracking（已通过）

已完成按 body 的碰撞诊断、单边高度 barrier、25 iterations checkpoint 保存和三 seed 批量验证。聚合首回合存活率 98.96%，整体成功率 98.96%，无 NaN，超过进入阶段 2 所需的 95% / 90% 门槛。

### 阶段 2：扩大 whole-body 工作空间（平面部分已通过）

已按 20、30、50、70 cm 的顺序完成，并始终保留 30% 旧目标 replay；前方低位目标也已扩到 -18 cm，并验证了随目标降低而增加的本体前倾。下一步进入门把手接触任务，同时保留平面与低位 nominal 回归。之后再扩大 yaw。不要一次同时扩大所有范围。

### 阶段 3：提高动态质量和精度（基础版已完成）

直线、圆、8 字、竖直、yaw 和 6D 连续轨迹已经进入训练与确定性基准；Stage16d `model_700` 在三 seed 混合任务中 768/768 存活，位置误差约 5 mm，并完成 0.1/0.2/0.35 Hz、总计 576 环境的 speed sweep。下一步不是回到门任务，而是加入显式自碰撞距离/关节限位 margin，并对极低目标做可达性课程。若真机末端仍需要毫米级接触精度，可让统一 actor 负责大范围 whole-body motion，在末端叠加受限 residual IK/impedance，但不能让该残差绕过碰撞和稳定性约束。

### 阶段 4：actuator transfer 与 sim-to-real

先把 RobotLab actuator preset 混入课程，再加入延迟、摩擦、质量、COM、传感噪声和外力随机化。每引入一项都要保留 nominal 回归测试。真机前还需要实测 Z1 安装变换、惯量、关节零点和 B2-W 低层接口。

### 阶段 5：门任务（仅回归/应用层）

门资产、3 cm 近接触、接触一致抓持、把手旋转和后退拉门均已通过确定性多 seed 批量验证。下一步是把当前已知把手位姿和脚本化相位升级为视觉估计 + 可学习高层策略，同时保留 `model_625` 作为低层全身 EE-WBC。真机前必须替换柔顺抓持代理，并标定夹爪碰撞网格、TCP、门参数和力阈值。

## 8. 当前已知限制

- 已完成 flat plane 70 cm 平面目标、前方 -18 cm 低位目标，以及已知目标位姿下的铰接门接近、夹持、旋转把手和拉门基线；尚未完成视觉定位、可学习高层相位、精确夹爪接触模型或真机 sim-to-real。
- 夹爪只保持，不参与学习。
- 第一阶段 PD plant 是为了建立可学习基线，不等同真机执行器。
- 两帧历史和 216 维输入尚未做消融；后续可以比较单帧、三帧以及显式 acceleration。
- 当前仍是单 critic PPO。只有标准 PPO 基线收敛后，才值得加入 manipulation/locomotion/safety multi-critic，以免无法判断收益来自算法还是环境修复。
- Stage 5 最佳模型在 192 个环境中没有终止；后续低位/接触课程仍必须继续跟踪 `lidar_link`、`link2`、gripper 和 calf，而不能只看平均误差。
- 单环境视频只用于直观检查；鲁棒性结论来自确定性三 seed 批量评估。
- Stage16d 默认模型将落地最大髋偏差从约 16.5° 降到约 12°，仍不是完全对称的官方站姿；继续压到约 9° 的 `model_724` 已出现 2/768 边界失败，因此不能只追求外观。
- 极端低位 Stage10 仍有 51/128 环境因 gripper/lidar 邻域接触终止；下一轮应引入显式 self-collision distance 或安全 critic，而不是继续提高姿态奖励。
- 不同应用使用不同 checkpoint：通用动态 EE tracking 用 `model_700`，当前脚本化开门仍用 `model_625`。不能把门成功与通用 WBC 质量混成单一指标。
