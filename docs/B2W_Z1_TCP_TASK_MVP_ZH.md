# B2-W + Z1 TCP 协调任务 MVP：实现与验证记录

最后更新：2026-09-04（Europe/Berlin）

## 当前结论

`B2W-Z1-TCP` 与 `B2W-Z1-TCP-Play` 已经成为可实例化、可并行运行、可由
RSL-RL PPO 训练和回放的 Isaac Lab 任务。任务不重新训练 B2-W 的 16 维底层
locomotion；它只训练一个 2 维高层协调器：

```text
57 维可部署 observation
  -> PPO coordinator [v_x, yaw_rate]
  -> 冻结 robot_lab B2-W policy
  -> 12 个腿位置目标 + 4 个轮速度目标

固定世界坐标 TCP reference
  -> 每个 physics step 重新转换到当前 base frame
  -> 200 Hz Z1 differential IK
  -> 6 个机械臂关节目标
```

物理与 Z1 IK 为 200 Hz（`dt=0.005`），冻结 locomotion policy 与 PPO
coordinator 都为 50 Hz（`decimation=4`）。夹爪在当前 free-space tracking 阶段
固定，不参与学习。

这项里程碑证明了模型、坐标语义、控制器组合、并行环境和训练/导出管线能够一起
工作。它还没有证明已经得到高质量全身协调策略。

## 主要实现

- 任务注册与配置：
  [`config/b2w_z1`](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/b2w_z1)
- 冻结低层 B2-W policy 的显式关节名映射：
  [`robot_lab_b2w_policy.py`](../source/LeggedManip_Lab/LeggedManip_Lab/controllers/robot_lab_b2w_policy.py)
- 统一的腿、轮与机械臂 action term：
  [`b2w_tcp_action.py`](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/b2w_tcp_action.py)
- 固定世界目标与 minimum-jerk reference：
  [`b2w_tcp_command.py`](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/b2w_tcp_command.py)
- observation、reward 与 termination：
  [`b2w_tcp_terms.py`](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/b2w_tcp_terms.py)
- 有限步回归与导出策略对比工具：
  [`b2w_z1_tcp_task_smoke.py`](../scripts/standalone/b2w_z1_tcp_task_smoke.py)

任务使用 57 维 actor/critic observation 与 2 维 action。当前 actor observation
由 TCP reference/final error、期望 TCP twist、机械臂状态、底盘速度与姿态、上一
高层 action 和冻结策略的上一 action 组成；不依赖 contact force 等只在仿真中
存在的特权信号。

## 目标怎样生成

当前 MVP 没有手工规定一个 Cartesian XYZ reach box。每次 reset 后先读取实际
TCP pose，然后在平面上采样一个虚拟 `ghost-root` 的平移与 yaw，将该 SE(2)
变换作用于当前 TCP：

- 平移半径：0.05--0.20 m；
- bearing：完整 `[-pi, pi]`；
- yaw：`[-0.25, 0.25]` rad；
- 10% episode 保持 stationary；
- 1 s 保持当前 pose，随后用 2 s minimum-jerk 到达最终 pose；
- 每个 episode 只生成一次最终世界目标，底盘移动不能拖动该目标。

这个分布专门用于第一阶段的平面底盘 relocation：如果底盘完成所采样的虚拟
运动，机械臂可以回到 home，同时 TCP 仍到达目标。它是完整 6D pose tracking
接口，但目标分布目前只覆盖 SE(2) relocation，不是一般 6D 工作空间采样器。

下一版一般 6D 目标应通过“安全机械臂关节采样 -> FK -> 碰撞、lidar clearance、
关节裕度与 manipulability 过滤”建立 pose catalog，再叠加 ghost-root 变换。这样
仍不需要人工绘制笛卡尔可达区域。

## 为什么 reward 要包含 terminal arm-home

只奖励 TCP tracking 时，同一个 TCP pose 可以由许多不同的“底盘 + 机械臂”组合
完成。在当前 0.05--0.20 m 目标中，机械臂往往可以独立完成，因此零底盘动作就
已具有很高 TCP 精度。

任务只在 reference 后 20% 平滑启用 arm-home penalty。它要求到点时机械臂趋近
home，从而给高层策略一个移动底盘、释放机械臂裕度的理由；但不会把隐藏的
ghost-root pose 直接作为 observation 或 reward 答案。运动早期没有全局 home
penalty，以免它与 TCP 轨迹跟踪直接冲突。

## 已完成验证

所有冒烟测试都使用 RTX 4090、CUDA、重力和真实 wheel/contact dynamics。初始
reference 与 reset 后实际 TCP 一致，最终目标保持 immutable，所有 observation、
action、reward、关节和 TCP 状态均 finite；测试期间没有提前 reset 或跌倒。

| 测试 | 终端位置误差 mean / max | 终端姿态误差 mean | 底盘平均位移 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 1 env，200 steps，zero action | 2.25 / 2.25 mm | 0.177 deg | 6.09 mm | PASS |
| 16 env，200 steps，zero action | 3.00 / 12.94 mm | 0.463 deg | 5.40 mm | PASS |
| 64 env，200 steps，zero action | 2.90 / 15.08 mm | 0.341 deg | 5.54 mm | PASS |

机器可读报告：

- [`1 env`](validation/b2w_z1_tcp_task_smoke_1env_gpu.json)
- [`16 env`](validation/b2w_z1_tcp_task_smoke_16env_gpu.json)
- [`64 env`](validation/b2w_z1_tcp_task_smoke_64env_gpu.json)

16/64 环境目标集合中出现过 IK joint-rate/limit clamp，测试仍保持 finite 且没有
reset。这些计数必须作为训练评估指标继续跟踪，不能因为最终 TCP 误差较小而忽略。

RSL-RL 集成测试也已完成：

- 64 env、1 iteration：4,096 transitions，checkpoint 正常保存；
- 256 env、20 iterations：327,680 transitions，约 3,902 steps/s；
- checkpoint 能由标准 `play.py` 读取，并成功导出 TorchScript 与 ONNX；
- 标准 play 路径成功生成短视频。

20-iteration pilot 的本地目录是
`logs/rsl_rl/b2w_z1_tcp/2026-09-04_15-19-44_pilot20`。`logs/` 按仓库规则不提交；
本页和
[`pilot summary`](validation/b2w_z1_tcp_pilot20_summary.json)
保留可审计结果。

## Pilot、零动作与解析协调器的确定性对比

使用相同 seed 43、64 environments、400 task steps：

| 指标 | zero action | analytic diagnostic | pilot20 policy |
| --- | ---: | ---: | ---: |
| 终端 TCP 位置 mean | 1.332 mm | 1.847 mm | 1.879 mm |
| 终端 TCP 姿态 mean | 0.222 deg | 0.385 deg | 0.179 deg |
| normalized arm-home error | 0.4507 | **0.2902** | 0.4598 |
| root-goal position error | 101.10 mm | 87.00 mm | 88.35 mm |
| root-goal yaw error | 7.20 deg | 10.40 deg | 16.18 deg |
| 400-step partial rollout return | **69.30** | 69.02 | 67.95 |
| reset / fall | 0 | 0 | 0 |

Pilot 已开始输出非零底盘命令，但没有降低 arm-home error，综合 reward 也没有超过
zero action。因此 `model_19.pt` 只能称为训练管线 pilot，不能称为可用或最佳策略。
它不应进入真机。

解析器使用 reset 时保存的精确 `sampled_tcp_pose_b`，所以这里只把它当作
diagnostic/oracle comparison，不把它包装成可直接部署的 controller。它通过真实
wheel/contact 运动把 arm-home error 降低约 36%，并消除了本次
400-step 运行中的 IK clamp；但其底盘仍没有到达 ghost-root，且总 reward 仍略低于
zero action。按方向统计显示 left 目标的 root-goal position error 为约 160.6 mm，
明显比 forward 的 59.7 mm 更差，符合已经测得的 yaw/plant 不对称。报告中的
`status=PASS` 只表示 shape、finite、immutable、reset 与最低高度等结构性检查通过，
不表示解析器通过了 TCP、底盘 relocation 或碰撞安全的精度门槛。

完整对比报告：

- [`zero action`](validation/b2w_z1_tcp_task_zero_64env_gpu.json)
- [`analytic diagnostic`](validation/b2w_z1_tcp_task_analytic_64env_gpu.json)
- [`pilot20 policy`](validation/b2w_z1_tcp_task_policy_64env_gpu.json)

## 可复现命令

先确保外部冻结 checkpoint 存在且 checksum 正确：

```bash
./third_party/rl_sar_b2w/fetch.sh
```

运行任务回归：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/standalone/b2w_z1_tcp_task_smoke.py \
  --task B2W-Z1-TCP-Play --headless --device cuda:0 \
  --num_envs 64 --steps 200
```

在 directional gate 与 reward 排序通过后，进行标称平地训练：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/rsl_rl/train.py \
  --task B2W-Z1-TCP --headless --device cuda:0 \
  --num_envs 512 --max_iterations 300 --run_name nominal_stage3
```

播放指定 checkpoint：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/rsl_rl/play.py \
  --task B2W-Z1-TCP-Play --device cuda:0 --num_envs 1 \
  --checkpoint /absolute/path/to/model_N.pt
```

每个候选 checkpoint 都要用相同 seed 和目标集合，与 zero action 和解析
coordinator 对比，不能只看 TensorBoard training reward。

当前先运行解析诊断：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/standalone/b2w_z1_tcp_task_smoke.py \
  --task B2W-Z1-TCP-Play --headless --device cuda:0 \
  --num_envs 64 --steps 400 --seed 43 --controller analytic
```

## 下一阶段训练计划

1. **先建立 directional accuracy gate。** 固定 stationary、forward、reverse、
   left、right 与正/负 yaw 目标集合；分别规定 TCP、root-goal、arm-home、clamp、
   contact 和存活门槛。当前解析器只是诊断，不是合格基线。
2. **校准 reward 与解析 controller。** 当前解析运动明显改善 arm-home，却因
   transient twist/orientation、settled velocity 与 action cost 使总 reward 仍略低于
   zero action；这里比较的是 400/600 steps 的 partial rollout return。在目标排序
   合理以前不启动长 PPO。
3. **标称平地过拟合。** directional gate 与 reward 排序合理后，先用 512
   environments 跑 100--300 iterations；每 25--50 iterations 固定 seed 回放。只有
   arm-home、root-goal、成功率和 TCP 精度共同改善时才继续。
4. **建立一般 6D FK catalog。** 从安全 Z1 关节状态采样并过滤碰撞、lidar
   clearance、joint margin 和 manipulability，再逐渐加入平移/姿态难度。
5. **扩大并行度。** 512 环境确认学习方向后再到 1,024/2,048；先 profile 显存和
   steps/s，不直接假定 4,096 最快。
6. **补安全信号。** 当前组合 USD 关闭 self-collision，任务也还没有可靠的
   arm-body/lidar collision cost。进入宽目标或真机前必须补上距离/碰撞监控，并把
   wheel slip、wheel lift、非法接触和 actuator saturation 纳入验收。
7. **最后加入 sim-to-real randomization。** 标称 held-out 指标通过后，先加入小
   范围摩擦、质量/惯量、COM、PD、delay、state noise 和实测 payload；保留约 20%
   nominal environments 作为精度锚点。
8. **terrain 最后加入。** 当前 state-only、室内平地 TCP 任务不需要照搬 MLM 的
   rough-terrain mix。只有真实部署包含坡度或轻微地面不平时才加入对应的小范围
   terrain；超过约 1--2 cm 前先提供真机可获得的 lidar/height observation。

关键验收指标是 TCP position/orientation RMS、P95 与终端 dwell、arm-home/joint
margin、底盘位移、成功与存活率、碰撞、wheel slip/lift、clamp/saturation 和多
seed 稳定性。terrain 与宽域随机化都不能用来掩盖标称平地上的 reward 或控制错误。
