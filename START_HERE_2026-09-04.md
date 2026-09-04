# 2026-09-04：B2-W + Z1 在 RTX 4090 上从这里开始

> **历史执行计划，已被后续结果取代。** 本页以下内容保留当天早期“固定腿开环
> 转向诊断”的上下文，不再是当前待办。冻结的 `robot_lab` B2-W policy 已经产生
> 双向物理转向，组合 TCP gate 与正式 `B2W-Z1-TCP` task 的 1/16/64 环境测试
> 已通过，20-iteration PPO pilot 也已完成但未收敛。当前入口是
> [`docs/B2W_Z1_TCP_TASK_MVP_ZH.md`](docs/B2W_Z1_TCP_TASK_MVP_ZH.md)。

这份文件是今天在实验机 `/home/mingqian/LeggedManip_Lab` 的直接执行入口。
今天只解决 **B2-W 轮式转向模型**，不启动 PPO、不加入门、不加入相机。

## 今天结束时应该得到什么

至少留下以下产物：

1. 一份现有 `in_place` 模式的原地正/负 yaw 基线 JSON；
2. 当前轮胎碰撞体、诊断圆柱和圆冠代理的受控 A/B 报告；
3. 固定关节参考与阻抗/upright 稳定腿姿的 A/B 报告；
4. 一张对比表，能回答 yaw 为什么失败、哪个模型最可信；
5. 更新 wheel gate、主 handoff 和 Git commit。

如果今天只能完成第一项，也应保存失败报告和判断，不能为了显示通过而随意降低
所有摩擦或提高到不真实的轮端力矩。

## 当前起点

| 项目 | 当前状态 |
| --- | --- |
| 分支 | `codex/b2w-z1-tcp-foundation-20260903` |
| 开始今天工作前的远端 commit | `f0d1d624941b59d75a1e6693ef0029301ca0be46` |
| B2-W + Z1 资产 | 通过：23 个可动关节 |
| 64-env 固定本体 DIK | 通过 |
| 64-env 固定世界 TCP harness | 通过，但 root 由脚本直接驱动 |
| 16-env 前进/倒车 | 通过 |
| 1-env 左右圆弧 | 失败：左 `1.960 deg`，右 `2.026 deg` |
| 原地 `in_place` 模式 | 代码已经存在，尚无正式保存的验证报告 |
| PPO / 正式 `B2W-Z1-TCP` task | 尚未开始，今天仍为 No-Go |

真实 B2-W 没有转向舵机，官方普通转向演示为四轮着地的差速滑移转向。详细
证据见 [`docs/research/B2W_REAL_TURNING_OFFICIAL_ZH.md`](docs/research/B2W_REAL_TURNING_OFFICIAL_ZH.md)。

## 先保护现场

仓库已有与本任务无关的 GO1/GO2 USD 修改，以及未跟踪的遥操作脚本和 handoff
指针。它们属于用户已有工作：不要 reset、clean、stash、checkout 覆盖，也不要
使用 `git add -A`。只精确暂存今天创建或明确修改的 B2-W 文件。

在 4090 桌面终端中执行：

```bash
cd /home/mingqian/LeggedManip_Lab
source /home/mingqian/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab51

git status --short --branch
git rev-parse HEAD
nvidia-smi

export PYTHONPATH="$PWD/source/LeggedManip_Lab${PYTHONPATH:+:$PYTHONPATH}"
```

必须看到当前分支为 `codex/b2w-z1-tcp-foundation-20260903`。如果 HEAD 或工作区
与上表不同，先记录差异，不要自动 reset 或 pull。

## A0：先运行已经存在的原地转向基线

`scripts/standalone/b2w_z1_wheel_gate.py` 已经支持：

```text
--turn_mode in_place
```

它会使用本 gate 的关节顺序 `[FL, FR, RL, RR]`：

```text
正 yaw / 左转：[-1, +1, -1, +1]
负 yaw / 右转：[+1, -1, +1, -1]
```

这与官方电机索引顺序 `[FR, FL, RR, RL]` 不同；必须按关节名映射。

运行：

```bash
/home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  scripts/standalone/b2w_z1_wheel_gate.py \
  --headless --device cuda:0 --num_envs 1 \
  --turn_mode in_place \
  --phases turn_left turn_right \
  --report docs/validation/b2w_z1_wheel_in_place_baseline_1env_gpu.json
```

如果 gate 未通过，程序会在写完 JSON 后以非零状态退出，这是预期的诊断结果，
不是文件没有生成。确认报告存在并查看：

```bash
test -s docs/validation/b2w_z1_wheel_in_place_baseline_1env_gpu.json
/home/mingqian/miniforge3/envs/env_isaaclab51/bin/python -m json.tool \
  docs/validation/b2w_z1_wheel_in_place_baseline_1env_gpu.json | less
```

需要保存的关键量：

- 左/右 `signed_yaw_min_deg` 和 yaw 方向；
- `translation_max_m`；
- 四轮速度误差和最大轮端力矩；
- 四轮接触率、估计 contact-patch slip；
- 最大 roll/pitch、最低 base height、非轮部件触地；
- 制动时间、制动距离和最终轮速。

若要在本机 GUI 中观看，先完成 headless 报告，再去掉 `--headless` 运行同一命令。

## A1：轮胎碰撞几何 A/B

不要覆盖 canonical `b2w_z1.usd` 或官方视觉 mesh。创建独立的诊断资产/layer
或一个明确的运行时选项，建议接口为：

```text
--wheel_collision_mode current_hull | diagnostic_cylinder | crowned_proxy
```

三组只允许改变碰撞几何：

| 组 | 碰撞体 | 用途 |
| --- | --- | --- |
| A | 当前 `convexHull` | 已有基线 |
| B | 轴沿局部 `+Y`、半径 `0.113 m`、宽约 `0.050 m` 的解析圆柱 | 仅判断是否为接触几何问题 |
| C | 24--32 段、保留圆冠剖面的低面数凸代理 | 候选真实训练模型 |

保持视觉、质量、惯量、轮轴、增益、命令、摩擦、20 Nm effort limit 和
50 rad/s velocity limit 完全一致。解析平底圆柱的锐边不真实，B 组即使通过也
不能直接成为最终模型。

每组都运行同一套：

```text
turn-in-place left/right
arc left/right
forward/reverse regression
```

报告建议保存为：

```text
docs/validation/b2w_z1_turn_current_hull_1env_gpu.json
docs/validation/b2w_z1_turn_diagnostic_cylinder_1env_gpu.json
docs/validation/b2w_z1_turn_crowned_proxy_1env_gpu.json
```

## A2：腿部稳定控制 A/B

先在最佳且物理上可信的碰撞候选上比较：

- A：当前固定腿关节位置参考；
- B：保持四轮着地的阻抗/upright 稳定器，只用小幅腿动作抑制 roll、pitch 和
  高度误差。

不要一开始就抬轮或强制大幅卸载内侧轮。官方普通 turn-in-place 视频没有显示
这种动作。只有法向载荷和 yaw 数据明确表明需要时，才增加“小幅载荷重分配”
作为第三组实验。

## A3：接触参数只在几何和腿姿之后测试

当前简单材料使用 `static_friction=0.8`、`dynamic_friction=0.8`，属于各向同性
接触，不能真实表达橡胶胎纵向滚动和横向 scrub 的差异。

可以使用 `0.6 / 0.8 / 1.0` 的成对摩擦值做敏感性分析，但必须同时复测直行、
制动和坡度稳定。不能因为更低摩擦让 yaw 变大，就把它称为已校准轮胎。若
PhysX 简单材料无法同时复现纵向抓地和横向擦滑，应记录这个限制，并考虑独立
轮胎力模型或由真实低速日志校准的等效模型。

## 今天采用的通过标准

沿用现有 gate，而不是看视频主观判断：

| 指标 | 原地转向门槛 |
| --- | ---: |
| 左右有符号 yaw | 均至少 `25 deg` |
| 原地转向平移 | 不超过 `0.20 m` |
| 最差单轮接触率 | 至少 `95%` |
| 最大 roll/pitch | 不超过 `10 deg` |
| 非轮部件接触率 | 不超过 `1%` |
| 转向轮速误差 | 不超过 `50%` |
| 最低 base height | 至少 `0.35 m` |
| 制动时间 | 不超过 `1.2 s` |
| 制动距离 | 不超过 `0.25 m` |
| 最终轮速 RMS | 不超过 `0.75 rad/s` |

候选模型还必须继续通过 16-env 前进/倒车回归；只让 yaw 变大但破坏纵向速度、
制动或稳定性，不算通过。

## 立即停止并先诊断的情况

- 左右方向反了或数组顺序不确定；
- NaN/Inf、CUDA OOM、仿真崩溃或报告没有写完；
- 机身、Z1 或腿部碰地，base height 低于门槛；
- 轮端长期打满 20 Nm，或通过依赖提高官方限制；
- 只有把所有摩擦调得很低才能转，且直行/制动回归失败；
- 同时修改几何、摩擦、腿姿和控制增益，导致无法判断原因。

## 今天不做的事情

- 不开始 PPO 或 4096 环境训练；
- 不创建门把手接触和视觉任务；
- 不修改固定世界目标或已通过的 DIK/world-frame 控制器来掩盖转向问题；
- 不把官方未公开的 `wheeled_sport` 内部算法写成已知事实；
- 不在真机使用未经验证的裸四轮高速命令。

## 完成后的记录与提交

建立一张 `docs/validation/B2W_Z1_TURN_AB_2026-09-04_ZH.md`，逐组列出：

```text
commit / asset hash / collision mode / stance mode / friction
yaw left/right / translation / wheel tracking / torque
contact / slip / roll-pitch / braking / pass-fail / interpretation
```

随后更新：

- `docs/B2W_Z1_WHEEL_GATE_ZH.md`；
- `docs/B2W_Z1_TCP_TRAINING_PLAN.md`；
- `LAB_MACHINE_HANDOFF_2026-09-03.md`。

只精确暂存本次 B2-W 文件。完整 wheel gate 可信通过后，下一阶段才是正式
`B2W-Z1-TCP` task 的 1/16/64 环境冒烟和 checkpoint 保存/重载。

## 可以直接交给 4090 上 Codex 的任务说明

> 先阅读 `/home/mingqian/LeggedManip_Lab/START_HERE_2026-09-04.md`、
> `LAB_MACHINE_HANDOFF_2026-09-03.md` 和 `docs/B2W_Z1_WHEEL_GATE_ZH.md`。
> 保护现有 dirty/untracked 用户文件。先运行脚本已经支持的
> `--turn_mode in_place` 左右转基线并保存 JSON，不要重复实现 pure-yaw。
> 然后一次只改变一个因素，依次完成当前 convex hull、诊断圆柱、圆冠代理，
> 再做固定腿参考与四轮着地阻抗稳定姿态 A/B。保留官方 20 Nm/50 rad/s 限制
> 和现有纵向回归。用机器可读报告与统一指标决定模型；不要调 PPO reward，
> 不要靠整体降摩擦制造通过结果。完成后更新 A/B 对比、wheel gate 和 handoff。
