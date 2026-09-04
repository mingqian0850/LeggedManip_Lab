# B2-W + Z1：真实底盘运动下的世界 TCP 保持门槛

最后更新：2026-09-04（Europe/Berlin）

## 结论

冻结的公开 B2-W locomotion policy 与 Z1 6D Differential IK 已经在同一个
Isaac Lab physics loop 中接通。最终 1、16、64 个并行环境测试都通过：底盘由
腿/轮接触真实移动约 5 cm 并旋转约 5°，Z1 同时保持初始化后只记录一次的世界
坐标 TCP pose。

这个 gate 没有写入 root pose 或 root velocity，gravity 保持开启。它证明现有
两个子系统可以形成最小可用闭环；它还不是 PPO task，也没有证明 sim-to-real。

## 控制结构

[`b2w_z1_robot_lab_tcp_hold_gate.py`](../scripts/standalone/b2w_z1_robot_lab_tcp_hold_gate.py)
使用：

```text
200 Hz physics
  ├── 50 Hz frozen robot_lab B2-W policy
  │     input:  [base state, vx/wz command, 16 joint state, previous action]
  │     output: 12 leg position targets + 4 wheel velocity targets
  └── 200 Hz Z1 Differential IK
        input:  immutable world TCP target transformed into current root frame
        output: 6 arm joint position targets + arm-only gravity feed-forward
```

底盘位置小闭环只使用 `vx` 和 `wz`；`vy` 始终为零。脚本绝不直接写 root。

一个关键索引问题已经修正：floating-base PhysX gravity tensor 的前六列是基座
自由度，所以 Z1 gravity compensation 必须使用 `arm_joint_id + 6`，也就是与
floating Jacobian 相同的 generalized-coordinate 列。修正后 deployed/home arm
的最大静态关节偏差从诊断 run 中约 0.456 rad 降到 0.0353--0.0372 rad。

## Nominal 试验为什么失败

不做任何底盘补偿时，TCP controller 本身通过，但 base goal 失败：

- 2 秒 zero-command settle 已产生约 `+0.1008 m` 前移、`0.0356 m` 横移和
  `-3.60 deg` yaw；
- 最终 base x 为 `0.0751 m`，而目标是 `0.050 m`；
- 最终 yaw 为 `-1.835 deg`，而目标是 `+5 deg`；
- world target 没有改变，TCP moving/terminal checks、姿态、高度、关节限位和
  NaN checks 都通过。

机器可读 nominal 报告：
[`b2w_z1_robot_lab_tcp_hold_nominal_1env_gpu.json`](validation/b2w_z1_robot_lab_tcp_hold_nominal_1env_gpu.json)。

这与独立 locomotion gate 的结果一致：公开 policy 在当前组合资产上有约
`+0.05 m/s` 的零指令前向偏置，并且小正 yaw command 位于明显的左转死区。

## 显式补偿 A/B

为了判断问题是否只在低层命令映射，而不是 TCP controller，A/B 只增加两个
可见、可记录的补偿：

- `--forward_command_bias 0.05`：从 position-loop `vx` command 中减去
  `0.05 m/s`；
- `--min_yaw_command 0.30`：yaw error 超过 0.25° 时，给 policy 至少
  `0.30 rad/s` 的同向命令，以越过已测得的低速 yaw dead zone。

这些数值只是当前 nominal Isaac 模型上的系统辨识起点，不是真机标定值，不能
静默写死到 sim-to-real controller。正式任务应把它们作为配置参数，并用多速度
测试、真机低速数据及后续随机化替换。

复现 64 环境通过结果：

```bash
timeout --signal=TERM --kill-after=5s 120s \
  /home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/standalone/b2w_z1_robot_lab_tcp_hold_gate.py \
  --headless --device cuda:0 --num_envs 64 \
  --forward_command_bias 0.05 --min_yaw_command 0.30 \
  --report /tmp/b2w_z1_robot_lab_tcp_hold_64env.json
```

本机 Isaac Sim 5.1 偶尔会在 `simulation_app.close()` 后卡住，因此命令保留外层
`timeout`；JSON 会在关闭 app 前落盘。一次只运行一个 Kit 进程。

## 实测结果

所有结果均为 RTX 4090、Isaac Sim 5.1、Isaac Lab `main` commit
`b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`（仓库 `VERSION` 为 2.3.2，
不是 v2.3.2 release tag）、200 Hz physics、50 Hz frozen locomotion、
200 Hz DIK、平地、摩擦 0.8、deployed/home arm。

| 并行环境 | moving TCP 最坏环境 RMS | terminal TCP 最坏环境 RMS | base x 最坏终差 | base yaw 最坏终差 | 结果 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 12.160 mm / 1.045° | 1.198 mm / 0.154° | 2.305 mm | 0.295° | PASS |
| 16 | 12.454 mm / 1.070° | 1.449 mm / 0.194° | 3.186 mm | 0.387° | PASS |
| 64 | 12.415 mm / 1.084° | 1.554 mm / 0.196° | 3.373 mm | 0.381° | PASS |

64 环境 run 的最低机身高度为 0.608 m，最大绝对 roll/pitch 为 3.411°，最大
Z1 joint speed 为 1.757 rad/s；没有 NaN、joint-limit clamp 或 target mutation，
报告中的 root state write count 为零。

机器可读通过报告：

- [`1 env`](validation/b2w_z1_robot_lab_tcp_hold_bias_1env_gpu.json)；
- [`16 env`](validation/b2w_z1_robot_lab_tcp_hold_bias_16env_gpu.json)；
- [`64 env`](validation/b2w_z1_robot_lab_tcp_hold_bias_64env_gpu.json)。

## 这意味着什么与后续状态

现在可以结束“是否必须从零重做 B2-W locomotion”的讨论：第一版不需要。
本页验证的接口随后已放入正式 `B2W-Z1-TCP` manager-based task；task-level
1/16/64 environment smoke、PPO checkpoint save/load 和导出均已通过。完整实现、
20-iteration 非收敛 pilot 和下一阶段计划见
[`B2W_Z1_TCP_TASK_MVP_ZH.md`](B2W_Z1_TCP_TASK_MVP_ZH.md)。

冻结 locomotion 与 200 Hz DIK 仍保持不变。首个解析 coordinator 诊断已经完成：
它改善了 arm-home，但暴露出明显的 lateral/yaw plant asymmetry，尚未通过底盘
relocation accuracy gate。接下来先固定目标集合、门槛并校准 reward/controller，
再训练慢速 2 维 `[v_x, yaw_rate]` coordinator。碰撞过滤的一般 6D FK 目标、
测量后的 domain randomization 和 deployment-matched terrain 依次后置。
