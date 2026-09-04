# B2-W + Z1 固定本体 TCP 微分 IK 验收

最后更新：2026-09-03（Europe/Berlin）

## 结论

固定 B2-W 本体后，Z1 的 6D TCP 微分 IK 已经在 RTX 4090 上通过 64 个
并行环境的标称物理验收。五组目标都满足以下门槛：

- 终端位置 RMS 不高于 5 mm；
- 终端姿态 RMS 不高于 2 度；
- 终端位置抖动不高于 2 mm；
- 无关节限位夹紧、NaN/Inf 或本体漂移。

这说明当前官方派生模型中的 Z1 关节映射、`tcp_frame`、固定本体 Jacobian、
四元数顺序和微分 IK 数据通路是正确的。后续“固定世界目标 + 浮动本体”独立
harness 也已经通过，见
[`B2W_Z1_WORLD_FRAME_GATE_ZH.md`](B2W_Z1_WORLD_FRAME_GATE_ZH.md) 和
[`validation/b2w_z1_world_frame_gate_64env_gpu.json`](validation/b2w_z1_world_frame_gate_64env_gpu.json)。
它仍不能说明轮子、全身协调、门接触或 sim-to-real 已经完成。

## 测试配置

| 项目 | 配置 |
| --- | --- |
| 设备 | NVIDIA GeForce RTX 4090 24 GB |
| 并行环境 | 64 |
| 物理频率 | 200 Hz，`dt=0.005 s` |
| 每个目标时间 | 800 步，约 4 s |
| 本体 | 固定 |
| 重力 | 开启 |
| 重力处理 | Z1 六关节显式重力前馈 |
| 执行器延迟 | 标称测试中固定为 0 步 |
| IK | Damped Least Squares，阻尼 0.03 |
| IK 每步增益 | 0.2 |
| 关节目标变化上限 | 2 rad/s |
| 评估窗口 | 最后 100 步，约 0.5 s |

目标由稳定后的 ready pose 生成。偶数与奇数环境使用相反符号，因此并行测试
同时覆盖正、负方向：

- X 前后移动 40 mm；
- Y 左右移动 35 mm；
- Z 上下移动 35 mm；
- 绕 Z 轴旋转 8 度；
- 平移与 roll/pitch/yaw 组合目标。

## 64 环境实测结果

| 目标 | 位置 RMS | 姿态 RMS | 最大终端抖动 | TCP 线速度 RMS | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 前后 | 2.080 mm | 0.217° | 0.823 mm | 0.0068 m/s | 通过 |
| 左右 | 0.199 mm | 0.021° | 0.100 mm | 0.0047 m/s | 通过 |
| 上下 | 1.808 mm | 0.194° | 1.634 mm | 0.0100 m/s | 通过 |
| yaw | 0.149 mm | 0.054° | 0.047 mm | 0.0047 m/s | 通过 |
| 组合 | 2.187 mm | 0.236° | 0.216 mm | 0.0053 m/s | 通过 |

全部目标的关节限位夹紧次数为 0，本体位置漂移为 0。完整机器可读结果保存在
[`validation/b2w_z1_tcp_dik_gate_64env_gpu.json`](validation/b2w_z1_tcp_dik_gate_64env_gpu.json)。

## 为什么最初会失败

最初直接把 DIK 位置目标发送给低增益显式 PD，同时开启重力，却没有重力
前馈；关节目标还相对测量位置逐步限速。这样 PD 无法建立支撑机械臂所需的
位置误差和力矩，机械臂在重力下下垂并最终向限位移动，出现约 0.6 m 和 44 度
的系统性误差。

修正方式不是放宽验收阈值，而是：

1. 首先把执行器随机延迟固定为 0，建立可重复的标称基线；
2. 给 Z1 六关节加入 PhysX 计算的重力前馈；
3. 对 DLS 关节修正使用明确的小步增益和速度限制；
4. 仍然保留原来的 5 mm / 2 度严格验收门槛。

随机延迟、动力学误差和负载将作为后续鲁棒性测试逐项恢复，不能与第一轮
坐标/Jacobian 验证混在一起。

## 如何复现

在实验机的 `LeggedManip_Lab` 根目录执行：

```bash
export PYTHONPATH="$PWD/source/LeggedManip_Lab${PYTHONPATH:+:$PYTHONPATH}"
ACCEPT_EULA=Y /home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/standalone/b2w_z1_tcp_dik_gate.py \
  --headless --device cuda:0 --num_envs 64 \
  --report docs/validation/b2w_z1_tcp_dik_gate_64env_gpu.json
```

脚本失败时返回非零状态，不会把未达标结果伪装成成功。

## 后续门槛当前进度

修复多环境 reset 时漏加 `scene.env_origins` 后，64 环境 world-frame harness
已经通过。七类正负本体激励的最坏动态 RMS 为 8.712 mm / 2.017°，最坏
位移端保持 RMS 为 1.149 mm / 0.136°，最大保持抖动为 0.654 mm。目标没有
随本体移动、降低或倾斜而漂移。

这个结果是 standalone kinematic DLS harness：重力关闭，root pose/velocity
由脚本直接写入，机械臂只使用六个关节列。它不是轮子闭环、动态全身控制或
生产 WBC。

后续已经采用冻结的 B2-W locomotion policy 取代固定腿开环轮速作为生产基础，
并完成真实 wheel/contact 运动下的 TCP 保持、正式 `B2W-Z1-TCP` task、
1/16/64 环境 smoke 和 PPO 保存/加载/导出。最新状态见
[`B2W_Z1_TCP_TASK_MVP_ZH.md`](B2W_Z1_TCP_TASK_MVP_ZH.md)。本页的固定基座
结果仍是修改 FK、Jacobian 或 actuator 后必须保留的回归基线，不代表碰撞安全或
最终全身策略已经通过。
