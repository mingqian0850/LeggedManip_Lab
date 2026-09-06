# Stage 21：扩大 EE tracking 移动范围与轮式航向课程

## 目标

Stage 20b 证明 `model_725` 能稳定完成前向 1.25 m、后向 1.0 m 和约 1.24 m 的部分对角目标，但纯侧向目标明显不对称：左侧 0.75 m 尚可，右侧 0.56 m 就会碰撞。Stage 21 的目标是在不破坏已有毫米级 TCP tracking 的前提下，让 B2-W 对大侧向目标更多使用轮子移动，而不是只靠身体侧倾和机械臂极限伸展。

当前仍是一个统一 actor：22 维输出同时控制 12 个腿关节、6 个机械臂关节和 4 个轮子。这里没有把机械臂与底盘拆成两个独立控制器。

## Stage 21a：全航向奖励（拒绝）

第一版增加了 turn--travel--realign 航向奖励：

1. 大目标初段让底盘朝目标方向转向；
2. 中段沿轮子滚动方向移动；
3. 末段恢复初始 yaw，以满足最终 EE orientation；
4. 后方目标采用倒车等价航向，避免转 180°。

训练从 `model_725` 开始，在 RTX 4090 上用 1024 个环境微调 50 次迭代，共 1,638,400 个环境步，耗时 126.84 s。结果没有通过验收：

| checkpoint | Stage20b 宽范围存活 | 结论 |
|---|---:|---|
| 基线 `model_725` | 28/33 = 84.85% | 对照 |
| Stage21a `model_750` | 27/33 = 81.82% | 拒绝 |
| Stage21a `model_774` | 27/33 = 81.82% | 拒绝 |

全角度转向虽然改善了左侧极限，却让原本稳定的右斜向目标发生 30° 以上 roll 和侧翻。这说明在 TCP 参考同时移动时，仅用奖励要求大角度转身过于激进。

## Stage 21b：限角、慢速、侧向重点采样

第二版重新从未受 Stage21a 影响的 `model_725` 开始，做了四项改变：

- 将中间航向限制为 ±35°，不要求纯侧向目标瞬间转 90°；
- 将大范围 waypoint 的 minimum-jerk 时间从 4--5 s 放慢到 6 s；
- 65% 的平面 waypoint 在左右侧向轴附近采样，同时保留 50% 的 0.03--0.60 m 旧范围 replay；
- 加强 root roll/roll-rate 约束，并加入四轮接触力变异系数惩罚。

PPO 使用保守参数：学习率 `2.5e-5`、clip `0.06`、3 epochs。1024 个环境微调 50 次迭代同样是 1,638,400 个环境步，耗时 128.60 s。

## 宽范围独立目标结果

| 模型 | 存活 | 平均位置误差 | 平均姿态误差 | 轮位形对称误差 | 平均轮载 CV |
|---|---:|---:|---:|---:|---:|
| `model_725` | 28/33 (84.85%) | 5.82 mm | 0.964° | 92.85 mm | 0.379 |
| Stage21b `model_750` | 30/33 (90.91%) | 6.88 mm | 0.841° | 85.51 mm | 0.400 |
| Stage21b `model_774` | **31/33 (93.94%)** | 7.16 mm | **0.852°** | **81.68 mm** | 0.390 |

`model_774` 的关键边界变化：

- 右侧目标从 0/3 提升到 2/3：0.5625 m 和 0.75 m 均存活并到达；
- 左斜向最大尺度约 1.24 m 从失败变为成功；
- 左侧和右侧 0.9375 m 极限目标仍失败；
- 三个右斜向目标仍全部成功，但平均位置误差约 11 mm，root roll 最大约 13--15°；
- 左右 0.75 m 目标虽然成功，最大 root roll 仍分别约 17.3° 和 20.7°，因此还不能称为自然姿态。

宽范围平均位置误差比基线增加约 1.33 mm，平均 roll 也从 4.05° 增加到 5.84°。所以 Stage21b 的主要收益是覆盖率与左右对称性，不是所有姿态指标都改善。

## 原能力回归

| 基准 | 模型 | 存活 | 位置误差 | 姿态误差 | 平均 roll | 轮位形对称误差 | 轮载 CV |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage20 小范围 39 点 | `725` | 39/39 | 4.59 mm | 1.05° | 2.70° | 83.56 mm | 0.296 |
| Stage20 小范围 39 点 | `774` | 39/39 | 5.22 mm | **0.77°** | 2.92° | **60.44 mm** | **0.243** |
| 0.2 Hz 连续轨迹 128 env | `725` | 128/128 | 5.49 mm | 1.33° | 1.38° | 56.30 mm | 0.217 |
| 0.2 Hz 连续轨迹 128 env | `774` | 128/128 | 5.87 mm | **1.00°** | **0.97°** | **35.75 mm** | **0.196** |

结论：小范围位置精度有约 0.4--0.6 mm 的轻微退化，但存活率不变，orientation、连续轨迹 roll、左右腿几何对称和轮载均得到改善。因此 `model_774` 可作为新的宽范围候选，不替换或删除可回退的 `model_725`。

## 路径依赖与视频边界

单环境连续顺序测试在第 10 个目标 `left_0p75m` 于 59.94 s 因 `undesired_contact` 终止。此前已经完成：

- 前向 1.0 m；
- 后向 0.8 m；
- 高/低 ±0.25 m；
- 右斜向约 0.99 m；
- 大 6D pose；
- 左斜向约 0.99 m。

这表示独立初始化下的 31/33 不等于任意目标序列都稳定。录像刻意截到 59.60 s，在首次失败前结束：

- 远端：`docs/validation/e2e_ee_wbc/stage21/stage21b_wide_valid_model774_labeled.mp4`
- 本地镜像：`artifacts/stage21/stage21b_wide_valid_model774_labeled.mp4`
- 接触图：`artifacts/stage21/stage21b_wide_valid_model774_contact_sheet.jpg`

视频不提交 Git，因为当前公开 fork 无法写入 Git LFS；JSON、接触图、脚本和本报告提交远端。

## checkpoint

- 远端训练模型：`logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_15-26-22_stage21b_limited_heading_from725_50/model_774.pt`
- 本地备份：`artifacts/stage21/model774_wide_limited_heading.pt`
- 保留基线：`logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_11-49-23_stage18a_filter35_from700_50/model_725.pt`

## 复现注意事项

远端 Conda 环境当前的 editable install 默认指向 `/home/mingqian/LeggedManip_Lab`。运行本分支时必须优先设置：

```bash
export PYTHONPATH=/home/mingqian/LeggedManip_Lab-e2e-ee-wbc/source/LeggedManip_Lab${PYTHONPATH:+:$PYTHONPATH}
```

训练任务名：

```text
B2W-Z1-EE-WBC-Stage21b-Limited-Heading-Adaptive-Mirrored-Filter35-v0
```

宽范围验收任务名：

```text
B2W-Z1-EE-WBC-Stage20b-Wide-Grid-Adaptive-Mirrored-Filter35-Play-v0
```

## 下一步：Stage 22

不建议继续单纯增大奖励或训练半径。当前失败表明 actor 没有显式看到“底盘应该往哪里转、目前处于转向/滚动/回正哪一阶段”，只能从 EE error 间接猜测。Stage 22 应：

1. 在 actor observation 追加期望底盘 heading、沿轮向距离和机动 phase；
2. 将旧 216 维网络第一层扩展，新列初始化为零，从而在训练开始时严格复现 `model_774` 行为；
3. 参考生成器采用 turn-in-place、roll、final-align 的分段 minimum-jerk 轨迹，避免一边直线移动 TCP、一边要求底盘大角度旋转；
4. actor 仍统一输出腿、臂和轮，不拆成两个策略；可用 privileged base subgoal 训练 teacher，再蒸馏到仅 EE target 的 student；
5. 验收同时使用 33 点独立测试和连续顺序测试，并将左右 0.9375 m、连续 `left_0p75m` 作为明确失败用例。

只有 Stage 22 同时实现 33/33、连续序列无接触、侧向最大 roll 明显低于 15°，才继续把训练半径扩大到 1 m 以上。
