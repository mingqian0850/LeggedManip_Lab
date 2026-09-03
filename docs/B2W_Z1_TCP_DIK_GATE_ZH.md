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
四元数顺序和微分 IK 数据通路是正确的，可以进入“固定世界目标 + 浮动本体”
测试。它还不能说明轮子、全身协调、门接触或 sim-to-real 已经完成。

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

## 下一道门槛

下一步不是立即启动 PPO，而是解除固定本体，验证固定世界坐标 TCP 目标：

1. 浮动本体 Jacobian 使用 `tcp_frame` body row，并给六个 Z1 joint column
   加上 floating-base 的 6 列偏移；
2. 每一步把同一个 `T_world_tcp_target` 重新表达进当前 root/heading frame，
   禁止用“当前本体 + 偏移”重新生成目标；
3. 先让本体做小幅前后运动、升降和俯仰，检查机械臂是否同步反向补偿；
4. 再验证四轮正负方向、有效半径、制动和侧滑；
5. 上述测试通过后，才加入“目标超出舒适工作空间时底盘后退/转向”的
   reachability coordinator；
6. 最后才把 coordinator 替换或扩展为 PPO Actor。

浮动本体阶段的初始验收建议为：运动期间不高于 10 mm / 5 度，停止后的
终端误差恢复到 5 mm / 2 度，并且目标不随本体移动、降低或倾斜而漂移。
