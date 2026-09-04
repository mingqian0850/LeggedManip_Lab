# B2-W + Z1：复用 robot_lab 低层运动策略的验证记录

最后更新：2026-09-04（Europe/Berlin）

## 结论

第一版 TCP tracking 不再从零训练腿和轮子的 locomotion。当前路线是：

```text
固定世界坐标 6D TCP 目标
  -> Z1 DIK/OSC 快速内环
  -> 慢速可达性/姿态协调器
  -> [B2-W 机身 vx、yaw-rate 命令]
  -> 冻结的 robot_lab B2-W locomotion policy
  -> 12 个腿位置目标 + 4 个轮速度目标
```

公开的 B2-W checkpoint 已经能在本项目的 B2-W + Z1 组合 USD 上站立、前进、
后退，并证明两个方向都能产生物理 yaw。因此，没有必要先重做一个完整的轮足
locomotion policy。严格测试中的左转接触率/yaw-rate 和部分停车窗口仍未全部通过；
这是可用的低层控制基础，不是所有 locomotion margin 或 TCP 全身协调已经通过。

## 固定来源和接口

可复现下载与许可证说明位于
[`third_party/rl_sar_b2w`](../third_party/rl_sar_b2w/README.md)。当前固定：

- `fan-ziqi/rl_sar` commit：
  `376d42c9b128f963ab08579762d5a216a976ce39`；
- `policy.pt` SHA-256：
  `38155076408e8eccb22690c6c5be14bd1dcb9149245ca5e493308a9f6ff93b34`；
- TorchScript 接口：57 维 observation -> 16 维 action；
- physics：200 Hz；policy：50 Hz（decimation 4）；
- policy joints：FR、FL、RR、RL，每条腿 hip/thigh/calf，随后同顺序的四个 wheel。

组合模型包含 Z1，PhysX 全局 joint 顺序并不是 policy 的 16 关节顺序。
[`robot_lab_b2w_policy.py`](../source/LeggedManip_Lab/LeggedManip_Lab/controllers/robot_lab_b2w_policy.py)
因此按名字建立显式映射，并独立维护 locomotion policy 的 16 维上一动作；不能
截取组合 articulation 的“前 16 个关节”。轮子位置在 observation 中按上游契约
清零，轮速仍保留。

[`B2W_Z1_ROBOT_LAB_POLICY_CFG`](../source/LeggedManip_Lab/LeggedManip_Lab/assets/b2w_z1/b2w_z1_articulation_cfg.py)
是单独的 checkpoint 兼容配置，不会改变原有 B2-W + Z1 配置。它复现上游名义
腿姿、腿部 DCMotor 限幅/PD 参数和轮子 velocity actuator 参数，并继续使用当前
Z1 actuator。

## 可复现测试

先下载并校验 checkpoint：

```bash
./third_party/rl_sar_b2w/fetch.sh
```

单环境完整门槛：

```bash
/home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/standalone/b2w_z1_robot_lab_policy_gate.py \
  --headless --device cuda:0 --num_envs 1
```

16 环境批量站立：

```bash
/home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/standalone/b2w_z1_robot_lab_policy_gate.py \
  --headless --device cuda:0 --num_envs 16 --phases stand
```

## 当前测量结果

单环境、Z1 安全折叠、平地、无随机化结果：

| 相位 | 指令 | 测量结果 | 严格门槛 |
| --- | --- | --- | --- |
| stand | `0, 0, 0` | 4 s 漂移 0.194 m，姿态最大 0.49 deg | 通过 |
| forward | `vx=+0.3` | 1.253 m，稳态 0.338 m/s，RMSE 0.039 m/s | 通过 |
| reverse | `vx=-0.3` | 0.894 m，稳态 0.258 m/s，RMSE 0.046 m/s | 通过 |
| turn left | `wz=+0.3` | 14.85 deg，稳态 0.082 rad/s | 未通过 |
| turn right | `wz=-0.3` | 60.52 deg，稳态 0.274 rad/s | 通过 |

`wz=0.5 rad/s` 的诊断试验得到：

| 相位 | 最终 yaw | 稳态同向 yaw rate | 平移 | 解释 |
| --- | ---: | ---: | ---: | --- |
| turn left | 70.27 deg | 0.349 rad/s | 0.036 m | 确认真正左转；RMSE 0.164 略高于严格 0.15 门槛 |
| turn right | 90.70 deg | 0.418 rad/s | 0.078 m | 通过 |

16 个并行环境的零指令站立也通过：最低机身高度 0.615 m，最大横滚/俯仰
0.65 deg，4 秒最坏平移 0.189 m，最小单轮接触率 99.875%，非轮部件接触率
为零。

16 环境 locomotion 批量结果进一步表明：

- forward 的最坏行程 1.242 m、最坏稳态速度 0.335 m/s、最大速度 RMSE
  0.0416 m/s，所有门槛通过；
- reverse 的最坏行程 0.888 m、最坏稳态同向速度 0.256 m/s、最大速度
  RMSE 0.0485 m/s；运动/姿态/接触都通过，但至少一个环境没有在严格的 2 秒
  连续低速窗口内完成停车；
- `wz=+0.5` 左转最坏仍达到 66.56 deg，但最小单轮接触率 0.82 且最大 yaw-rate
  RMSE 0.1789 rad/s，严格门槛未通过；
- `wz=-0.5` 右转最坏仍达到 90.55 deg，yaw-rate RMSE 0.1064 rad/s，但部分
  环境没有在 2 秒内完成严格停车，严格门槛未通过。
- Z1 从 stow 改为 TCP 训练用的 deployed/home pose 后，16 环境 stand 仍通过：
  最低机身高度 0.615 m、最大横滚/俯仰 0.522 deg、最小单轮接触率 99.875%、
  无非轮接触；但 4 秒约前爬 0.200 m（约 0.052 m/s），已使用 80% 的 smoke
  漂移预算。后续 TCP hold 必须把它作为真实扰动进行补偿。

原始机器可读报告：

- [`stand, 16 env`](validation/b2w_z1_robot_lab_stand_16env_gpu.json)；
- [`deployed/home arm stand, 16 env`](validation/b2w_z1_robot_lab_home_stand_16env_gpu.json)；
- [`forward/reverse, 16 env`](validation/b2w_z1_robot_lab_straight_16env_gpu.json)；
- [`left/right yaw 0.5, 16 env`](validation/b2w_z1_robot_lab_yaw_05_16env_gpu.json)。

冻结 locomotion 与世界坐标 TCP DIK 的下一层组合 gate 也已经在 1/16/64 环境
通过，完整方法与精度见
[`B2W_Z1_ROBOT_LAB_TCP_HOLD_GATE_ZH.md`](B2W_Z1_ROBOT_LAB_TCP_HOLD_GATE_ZH.md)。

这些结果证明：

1. checkpoint、关节顺序、action scaling 和 actuator 配置已经接对；
2. 当前组合 USD 的轮子确实能够在 locomotion policy 控制下转向；
3. `wz=0.3` 左转存在明显低速死区或左右不对称，不能把 yaw 指令当成精确执行器；
4. 高层协调器需要用实际 base/TCP 状态闭环，而不能只做开环 command mixing；
5. checkpoint 是外部基础模型，不是本项目训练出来的最终策略。

## `wb-mpc-locoman` 在本项目中的位置

`lukasmolnar/wb-mpc-locoman` 是很有价值的 Whole-Body Inverse Dynamics MPC
参考，但当前公开示例是**足式 B2 + Z1**，使用足端接触 gait schedule，不包含
B2-W 的四个滚动轮、轮速动作和非完整滚动约束。默认目标也是 base 6D velocity、
相对 base 的 arm EE linear velocity 和世界系 EE force，而不是直接给定一个固定
世界坐标的 6D TCP pose。

本次审计固定到 upstream commit
[`80e906d35d91783e85e1ef994023ca9082dc40c3`](https://github.com/lukasmolnar/wb-mpc-locoman/commit/80e906d35d91783e85e1ef994023ca9082dc40c3)，
代码许可证为 MIT。默认
[`main.py`](https://github.com/lukasmolnar/wb-mpc-locoman/blob/80e906d35d91783e85e1ef994023ca9082dc40c3/main.py#L12-L40)
只启用 Z1 前四轴，并没有一般 6D TCP orientation tracking。公开程序用优化器预测
的下一状态继续滚动，再用 Meshcat 回放；它没有包含 Isaac/ODE plant、Unitree SDK
或真机状态反馈闭环。因此，应区分论文展示的系统和这个公开最小代码仓库。

因此它不是当前 `robot_lab B2-W policy + Z1 DIK` 的即插即用替代品。后续可以
复用或对照它的内容包括：

- whole-body RNEA/centroidal dynamics 建模；
- torque、contact、friction 和 joint limit 约束；
- receding-horizon、warm start 和 solver benchmark；
- 接触力任务或把 MPC 当作 teacher 生成示范。
- TCP 速度合成关系
  `v_tcp^W = v_base^W + omega_base^W x r^W + R_WB v_arm^B`，以及用 position
  feedback 生成 velocity command 的思路。

若要直接用于 B2-W，必须新增 wheel joint、轮地接触几何、滚动/侧滑约束、轮扭矩
以及与 50 Hz learned locomotion policy 的接口。这是独立的研究工作，不应阻塞
当前 TCP tracking 最小闭环。

它声明 Python 3.12、Pinocchio 3.3、CasADi/Meshcat/OSQP 等依赖；当前 Isaac Lab
环境是 Python 3.11/Pinocchio 2.7。若以后运行原仓库对照实验，应建立独立环境，
不要把这些求解器依赖直接灌进 `env_isaaclab51`。

## 后续状态与下一门槛

上述两个部件已经连入同一物理闭环，且 1/16/64 环境组合 gate 已通过。随后
`B2W-Z1-TCP` 正式 task、57 维 observation、2 维 coordinator action、PPO
checkpoint 保存/加载与导出也已实现。完整结果见
[`B2W_Z1_TCP_TASK_MVP_ZH.md`](B2W_Z1_TCP_TASK_MVP_ZH.md)。

20-iteration pilot 只证明训练管线可用；相同 seed 评估尚未超过 zero-action。
首个解析 coordinator 对照已经加入并完成 64 环境诊断：它降低了总体 arm-home
误差，但 left/lateral 与 yaw relocation 仍不可靠，总 reward 也没有超过 zero
action。下一门槛是固定目标集合与 accuracy gate、校准 reward/controller，再让
learned coordinator 同时改善 TCP held-out 与 arm-home/joint-margin 指标。之后才
扩展为碰撞过滤的一般 6D FK 目标；名义任务收敛以前仍不加入随机地形或宽域随机化。
