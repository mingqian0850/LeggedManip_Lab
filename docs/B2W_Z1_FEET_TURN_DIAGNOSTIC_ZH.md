# B2-W + Z1 腿部参与转向诊断

最后更新：2026-09-04（Europe/Berlin）

## 结论

本轮没有得到可用的“只靠腿部运动掉头”控制器。

短时扫描一度出现约 `+9.56 deg / -8.77 deg` 的偏航，但相同参数从 4 秒延长
到 8 秒后变成 `-5.27 deg / +7.14 deg`，方向反转。完整周期复核证明该结果是
起止相位与机身摇摆造成的假阳性，不是可持续累积的转向。

对角步态、三轮支撑 crawl、带重心预移的 crawl 均未让指定摆动轮可靠离地。
摆动阶段的轮地接触率仍接近 100%，实际轮心最大抬升远低于指令值。因此当前
开环动作主要造成载荷变化、擦地和小幅姿态摆动。

这不代表腿部对真实 B2-W 转向没有作用。更准确的结论是：腿部可能负责姿态
和轮载协调，但当前证据不支持“普通平地转向由抬轮踏步取代轮电机差速”。官方
结构和视频证据边界见
[`research/B2W_REAL_TURNING_OFFICIAL_ZH.md`](research/B2W_REAL_TURNING_OFFICIAL_ZH.md)。

## 为什么早期结果不能采用

早期参数扫描把非整数个步态周期、启动 ramp 和最终不一致的步态相位一起计入
总偏航。最佳 4 秒候选使用 `0.6 Hz`，结束于第 2.4 个周期；终止时的机身摆动
被误计为转向。延长时间后净偏航改变符号，直接否定了该候选。

另一个容易混淆的地方是 `wheel_speed=0`：

- 当 `wheel_velocity_gain=10` 时，零轮速是主动速度环刹车/保持，不是自由轮；
- 当 `wheel_velocity_gain=0` 时，轮子才是不施加驱动力矩的被动自由轮。

后续报告明确区分这两种模式。

## 严格验收方法

诊断脚本：
[`scripts/standalone/b2w_z1_leg_wheel_turn_sweep.py`](../scripts/standalone/b2w_z1_leg_wheel_turn_sweep.py)

脚本现在额外记录：

- 每个完整步态周期的净 yaw 和方向同号率；
- 每周期平移；
- 指令摆动期和支撑期的逐轮接触率；
- 实际轮心相对测量起点的抬升；
- base 高度下降、最大 roll/pitch 和非轮部件接触；
- 腿关节目标跟踪误差；
- 实际轮速、轮旋转量、正功、制动功和 20 Nm 饱和占比；
- 积分 yaw 与四元数 yaw 的一致性。

“持续足驱动转向”要求连续 4 个完整周期均沿命令方向、每周期至少 2° yaw、
每周期平移不超过 2 cm，同时满足姿态、离地、接触和执行器门槛。所有本轮候选
在该严格门槛下均为失败，这是预期且正确的报告结果。

## 实验结果

### 1. 对角两轮交替

固定频率 `0.625 Hz`，先运行 2 个完整周期（包含 ramp）但不计分，再测量 4 个
完整周期。全因子测试了：

- 起始对角相位：`0.0 / 0.5`；
- yaw stride：`-0.12 ... +0.12 rad`，包含零 stride 对照；
- 抬升指令：`0.06 / 0.09 m`；
- 轮速命令：0，轮速度环主动保持。

结果：

- 最大总偏航绝对值为 `2.81°`，但四个周期不保持同向；
- 零 stride 对照也出现最高约 `1.72°` 总偏航，证明摇摆/接触不对称会制造假
  yaw；
- 摆动期接触率最低仍为 `96.9%`；
- 所有轮中最大的实际轮心抬升也只有 `1.28 cm`；
- 最差 base 高度下降 `8.74 cm`。

结论：两条对角腿形成线支撑，机身下沉/滚转后摆动轮继续压地，不能作为转向
步态。

### 2. 单轮依次摆动的 crawl

将步态改为每次仅摆动一只轮足，其余三只支撑；随后增加周期性重心预移，先把
载荷移向另外三足，再抬起目标轮足。

主动轮刹车下的代表结果：

| 候选 | 每周期 yaw | 最大每周期平移 | 最大 roll/pitch | base 下降 | 摆动接触 | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 右转 `418` | `-0.45, -0.45, -0.39, -0.38°` | `0.30 cm` | `4.35°` | `2.96 cm` | `100%` | 失败 |
| 左转 `478` | `+0.58, +0.49, +0.42, +1.33°` | `1.41 cm` | `6.20°` | `3.55 cm` | `100%` | 失败 |

这两个候选的方向能连续保持，但幅度太小，而且四轮全程接触。它们是腿部改变
轮载与接触几何后产生的小幅擦地偏航，不是实际抬轮转向。

### 3. 自由轮对照

关闭轮速度环后，代表候选的四周期 yaw 不再稳定同向：

- 候选 `418`：`-1.53, +0.04, -1.32, +0.48°`；
- 零 stride 对照 `445`：`-0.67, +0.97, +0.31, +0.45°`；
- 候选 `478`：`+1.23, +1.63, -0.21, +0.87°`。

自由轮模式还带来明显更大的平移和约 `1.0--1.2 rad/s` 的被动轮速。说明主动
锁轮能抑制漂移，但当前腿部轨迹仍不足以形成可靠掉头。

## 对训练设计的影响

首版训练不应把“只靠腿转向”硬编码为约束，也不应奖励开环抬轮动作。推荐让
locomotion policy 同时输出 12 个腿关节目标偏移和 4 个轮速目标，并通过任务
奖励让它学习协同：

- 主要任务：`vx / vy / wz` 跟踪，尤其单独覆盖 `vx=vy=0, wz!=0`；
- 稳定性：roll/pitch、base 高度、非轮接触和关节限位；
- 转向质量：yaw-rate 误差、横向漂移、轮胎 slip 和轮载不均衡；
- 执行器：腿/轮扭矩、轮扭矩饱和、动作变化率和能耗；
- 接触：允许腿部做小幅载荷协调，但不直接规定必须抬哪只轮。

Z1 在这个底盘预训练阶段保持当前安全折叠位。底盘 yaw 控制通过后，再加入手臂
home pose、TCP tracking 和全身协调奖励。

## 复现命令

```bash
# 对角步态真假鉴别：主动锁轮
ACCEPT_EULA=Y /home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/standalone/b2w_z1_leg_wheel_turn_sweep.py \
  --headless --device cuda:0 --strategies diagonal_factorial \
  --ramp_steps 320 --warmup_steps 640 --drive_steps 1280 \
  --wheel_velocity_gain 10 --report /tmp/b2w_z1_diagonal_factorial_active_brake.json

# 三轮支撑、带重心预移：主动锁轮
ACCEPT_EULA=Y /home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/standalone/b2w_z1_leg_wheel_turn_sweep.py \
  --headless --device cuda:0 --strategies crawl_weight_shift \
  --ramp_steps 400 --warmup_steps 800 --drive_steps 1600 \
  --wheel_velocity_gain 10 --report /tmp/b2w_z1_crawl_weight_shift_active_brake.json

# 最佳左右候选和零 stride 的自由轮对照
ACCEPT_EULA=Y /home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/standalone/b2w_z1_leg_wheel_turn_sweep.py \
  --headless --device cuda:0 --candidate_indices 418 445 478 \
  --ramp_steps 400 --warmup_steps 800 --drive_steps 1600 \
  --wheel_velocity_gain 0 --report /tmp/b2w_z1_crawl_selected_passive_wheels.json
```

这些是仿真诊断，不是可直接下发真机的控制命令。若要确认实机机制，应使用
Unitree 高层低速 yaw 命令同步记录四个轮电机、12 个腿关节、IMU 和轮载；仅凭
外观视频不能判断轮电机是否在输出差速力矩。
