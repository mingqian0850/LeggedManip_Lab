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

训练窗口在 `model_298` 附近曾显示 mean episode length 约 583、timeout 约 95.6%，但确定性首回合测试只有 32.8% 存活。差异来自训练指标的滚动/随机分布与 reset 混合，而确定性评估严格追踪同一批环境的第一次 episode。以后 checkpoint 选择一律以后者为准。

还额外做过两项失败诊断并保留结果：

- 直接使用 RobotLab actuator preset 而不运行它的冻结 locomotion policy，零动作会反复触发 `low_base`；因此它不适合作为第一阶段的自然稳定 plant，应作为后续 actuator-transfer 阶段。
- 只把 root 高度设置成落稳后的 0.505 m，而不同时匹配姿态和关节状态，会导致初始几何接触；因此没有采用这种不完整 reset。

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

## 7. 下一步顺序

### 阶段 1：nominal EE tracking（已通过）

已完成按 body 的碰撞诊断、单边高度 barrier、25 iterations checkpoint 保存和三 seed 批量验证。聚合首回合存活率 98.96%，整体成功率 98.96%，无 NaN，超过进入阶段 2 所需的 95% / 90% 门槛。

### 阶段 2：扩大 whole-body 工作空间（平面部分已通过）

已按 20、30、50、70 cm 的顺序完成，并始终保留 30% 旧目标 replay。下一步单独加入 z 方向目标和可控 body pitch/height，使低目标可以通过前低后高的全身倾斜完成；继续保留平面 nominal 回归。之后再扩大 yaw。不要一次同时扩大所有范围。

### 阶段 3：提高动态质量和精度

加入持续的圆、直线和门把手形轨迹，而不是每 episode 只有一个点；分别统计 0.15/0.4 Hz tracking。若纯 RL 仍停留在厘米级，保留统一 actor 负责全身大范围运动，并在末端增加 residual IK/impedance 做毫米级对准。

### 阶段 4：actuator transfer 与 sim-to-real

先把 RobotLab actuator preset 混入课程，再加入延迟、摩擦、质量、COM、传感噪声和外力随机化。每引入一项都要保留 nominal 回归测试。真机前还需要实测 Z1 安装变换、惯量、关节零点和 B2-W 低层接口。

### 阶段 5：门任务

在 tracking policy 稳定后再加入门、旋转把手和拉门接触：夹爪由独立 grasp state 控制；policy 输入增加腕部/门把手相对位姿和接触/力信息；先训练对准与抓持，再训练把手旋转，最后训练保持连接时的本体后退。不要把开门直接混进第一阶段 TCP tracking。

## 8. 当前已知限制

- 目前已有 flat plane 上最大 70 cm 的平面 ghost-root 目标；还没有低位目标、门、视觉或触觉。
- 夹爪只保持，不参与学习。
- 第一阶段 PD plant 是为了建立可学习基线，不等同真机执行器。
- 两帧历史和 216 维输入尚未做消融；后续可以比较单帧、三帧以及显式 acceleration。
- 当前仍是单 critic PPO。只有标准 PPO 基线收敛后，才值得加入 manipulation/locomotion/safety multi-critic，以免无法判断收益来自算法还是环境修复。
- Stage 5 最佳模型在 192 个环境中没有终止；后续低位/接触课程仍必须继续跟踪 `lidar_link`、`link2`、gripper 和 calf，而不能只看平均误差。
- 单环境视频只用于直观检查；鲁棒性结论来自确定性三 seed 批量评估。
