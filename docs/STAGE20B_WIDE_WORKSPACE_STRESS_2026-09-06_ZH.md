# Stage20b：大范围 EE/TCP tracking 压力测试

日期：2026-09-06  
策略：Stage18 `model_725.pt` + Stage19e reach-aware adaptive mirrored Filter35 action

## 为什么需要 Stage20b

Stage20 nominal 录像的最大位移只有前向 0.60 m、后向 0.35 m、侧向 0.32 m，主要用于检查训练分布附近的姿态质量。Stage20b 将位置和姿态显著放大，用来回答当前 actor 是否真的具备大范围 whole-body locomotion，而不是制作成功演示。

模型原训练的平面 waypoint 半径上限约 0.70 m。Stage20b 同时包含训练边界内、边界附近和明显超出训练范围的目标：

- 前向 1.00 m，三档后最大 1.25 m；
- 后向 0.80 m，最大 1.00 m；
- 纯左右 0.75 m，范围 0.5625–0.9375 m；
- 左右对角基准半径约 0.99 m，最大约 1.24 m；
- 高低偏移 ±0.25 m，最大 ±0.3125 m；
- 大 6D pose：位置 `(0.45, 0.20, 0.12)` m，姿态 `(25°, −20°, 45°)`，并使用三档幅度。

所有独立目标使用 4.0 s minimum-jerk 过渡和 3.0 s 保持，避免把目标太远与目标速度过快混为一谈。

## 33 环境独立目标结果

每个命名目标分别使用 0.75、1.00、1.25 三档幅度。结果为：

- 首回合存活 **28/33 = 84.85%**；
- 5 次失败全部是 `undesired_contact`；
- 无 `low_base`、`bad_orientation` 或 time-out；
- 前向、后向、高位、低位、右前对角和独立 large 6D pose 的三档目标全部完成；
- 纯右侧三档全部失败；
- 左侧与左前对角仅最大 1.25 倍目标失败。

| 目标 | 存活 | 位置均值 mm（仅幸存者） | 最大本体 roll ° | 轮位镜像误差均值 cm | 载荷 CV 均值 | 最终底盘平移均值 m |
|---|---:|---:|---:|---:|---:|---:|
| front 1.0 m | 3/3 | 2.8 | 1.0 | 8.7 | 0.16 | 1.03 |
| rear 0.8 m | 3/3 | 3.1 | 0.3 | 6.9 | 0.24 | 0.76 |
| high 0.25 m | 3/3 | 10.9 | 0.9 | 4.2 | 0.24 | 0.49 |
| low 0.25 m | 3/3 | 8.3 | 1.8 | 12.9 | 0.27 | 0.48 |
| diagonal right | 3/3 | 9.6 | 11.2 | 15.2 | 0.81 | 0.98 |
| large 6D pose | 3/3 | 5.5 | 8.9 | 9.6 | 0.35 | 0.61 |
| diagonal left | 2/3 | 4.6 | 9.1 | 12.9 | 0.69 | 0.48 |
| left 0.75 m | 2/3 | 6.0 | 14.8 | 14.7 | 0.81 | 0.25 |
| right 0.75 m | **0/3** | — | — | — | — | — |

具体终止：

- right 0.75 m ×0.75：3.84 s 触发异常接触；
- right 0.75 m ×1.00：3.58 s；
- right 0.75 m ×1.25：3.44 s；
- diagonal left ×1.25：5.24 s；
- left 0.75 m ×1.25：4.64 s。

右侧目标越大，反而越早触发接触；这符合 actor 用更激进、不自然的腿部侧倾补偿目标，而不是形成稳定侧向 locomotion 的表现。

## 连续轨迹与路径依赖

大范围目标独立成功不代表能够任意连续拼接：

1. 原顺序 `diagonal_right → large_6d_pose` 在 **42.02 s** 触发异常接触；
2. 在 large 6D pose 前后插入 home 后，该段能够通过，但之后 `home → diagonal_left` 仍在 **54.40 s** 触发异常接触。

这证明当前策略存在明显路径依赖。其观察虽然包含 previous action 和短时轨迹 preview，但训练数据没有充分覆盖跨工作空间的大幅连续切换；回到 TCP home 也不等价于恢复统一、对称的本体内部姿态。

## 有效审阅视频

为了避免自动 reset 后字幕与真实命令错位，最终审阅版裁到第一次失败：

- 本地：`implementation/e2e_ee_wbc/artifacts/stage20/stage20b_wide_valid_until_failure_model725.mp4`
- 远端：`/home/mingqian/LeggedManip_Lab-e2e-ee-wbc/docs/validation/e2e_ee_wbc/stage20b/stage20b_wide_valid_until_failure_model725.mp4`
- 接触表：`implementation/e2e_ee_wbc/artifacts/stage20/stage20b_wide_valid_contact_sheet.jpg`

视频为 42.5 s、1280×720、50 fps，完整展示 1.0 m 前进、0.8 m 后退、±0.25 m 高低、约 1 m 右前对角，并在 42.02 s 明确标出 large 6D transition 的异常接触与自动 reset。MP4 继续保存在本地和试验机，不加入 public fork 的 Git LFS 历史。

## 结论与下一步

用户指出“移动范围太小”是正确的。扩大评估后得到的结论不是“模型在更大范围也很好”，而是：

- 当前 actor 已可靠学会约 1 m 的前进和后退协同；
- 大范围左右移动明显不可靠，并伴随本体 roll、四轮几何与载荷失衡；
- 连续跨区域目标比独立 waypoint 更困难，存在路径依赖和姿态记忆；
- 仅扩大测试命令无法修复，必须扩大训练课程并调整动作结构/姿态约束。

下一轮训练应分开处理：先建立前后/转向 locomotion 稳定基线，再逐步加入对角与侧向 EE 目标；对纯侧向不可直接假设轮式 B2W 能像全向底盘平移，应允许策略通过转向—行驶—再对准完成。验收继续使用 Stage20/20b，不再只看小范围周期轨迹的平均 TCP 误差。
