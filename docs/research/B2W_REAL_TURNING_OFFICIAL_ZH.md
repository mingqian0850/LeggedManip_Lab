# 真实 Unitree B2-W 转向方式：官方资料核对

最后更新：2026-09-03（Europe/Berlin）

## 一句话结论

真实 B2-W 没有独立的车轮转向舵机。普通平地偏航依靠左右轮纵向速度不同，
同时允许固定轮胎发生横向擦滑，即从机械运动学上属于 differential/skid
steering。官方教学视频显示常规原地旋转时四轮保持着地，没有抬轮或明显大幅
倾斜；腿可能执行小幅稳定或载荷调节，但 Unitree 没有公开生产控制器内部的
轮速混合、滑移补偿或腿部协同算法。

## 证据边界

| 结论 | 证据等级 |
| --- | --- |
| B2-W 没有 steering/swivel joint | 官方 URDF 与官方 MuJoCo 模型直接确认 |
| 高层接口使用 `vx, vy, vyaw`，支持 turn in place | 官方 SDK 示例与遥控手册直接确认 |
| 普通常规旋转时四轮着地、没有明显抬腿 | 官方教学视频逐帧可见 |
| 转向需要差速和横向轮胎擦滑 | 由固定轮结构与四轮着地原地偏航必然推出；不是官方采用的术语 |
| 内置控制器是否重分配轮载、怎样抑制滑移 | 未公开，不能宣称已经知道 |
| 精确最小转弯半径和生产 yaw-rate 性能 | 官方未公布；只能说支持名义上的原地旋转 |

## 1. 机械结构：四轮只有滚动自由度

官方 `unitree_ros` 快照
`7d6075f7f58588b189b940130e3edab3c839b2df` 的 B2-W URDF 中，每条腿只有
hip、thigh、calf 和连续 wheel joint；四个 wheel joint 的轴都为 `0 1 0`，
不存在绕竖直方向旋转轮面的关节：

- [FL wheel joint](https://github.com/unitreerobotics/unitree_ros/blob/7d6075f7f58588b189b940130e3edab3c839b2df/robots/b2w_description/urdf/b2w_description.urdf#L216-L221)
- [FR wheel joint](https://github.com/unitreerobotics/unitree_ros/blob/7d6075f7f58588b189b940130e3edab3c839b2df/robots/b2w_description/urdf/b2w_description.urdf#L320-L325)
- [RL wheel joint](https://github.com/unitreerobotics/unitree_ros/blob/7d6075f7f58588b189b940130e3edab3c839b2df/robots/b2w_description/urdf/b2w_description.urdf#L424-L429)
- [RR wheel joint](https://github.com/unitreerobotics/unitree_ros/blob/7d6075f7f58588b189b940130e3edab3c839b2df/robots/b2w_description/urdf/b2w_description.urdf#L528-L533)

官方 `unitree_mujoco` 快照
`4134cb5dc7ff1ba7f484deda48b5274b58694519` 独立确认了相同结构：

- [FL wheel](https://github.com/unitreerobotics/unitree_mujoco/blob/4134cb5dc7ff1ba7f484deda48b5274b58694519/unitree_robots/b2w/b2w.xml#L129-L136)
- [FR wheel](https://github.com/unitreerobotics/unitree_mujoco/blob/4134cb5dc7ff1ba7f484deda48b5274b58694519/unitree_robots/b2w/b2w.xml#L175-L182)
- [RL wheel](https://github.com/unitreerobotics/unitree_mujoco/blob/4134cb5dc7ff1ba7f484deda48b5274b58694519/unitree_robots/b2w/b2w.xml#L222-L229)
- [RR wheel](https://github.com/unitreerobotics/unitree_mujoco/blob/4134cb5dc7ff1ba7f484deda48b5274b58694519/unitree_robots/b2w/b2w.xml#L268-L275)

因此 B2-W 不是 Ackermann 转向，也没有四轮独立转舵。四个轮面始终基本随
小腿方向固定。

## 2. 官方用户接口与演示

官方 SDK2 快照
`9754cd153af3da471b0fe5f3aa535e426fb11db3` 提供的高层接口是：

```cpp
Move(float vx, float vy, float vyaw)
```

见 [`sport_client.hpp`](https://github.com/unitreerobotics/unitree_sdk2/blob/9754cd153af3da471b0fe5f3aa535e426fb11db3/include/unitree/robot/b2/sport/sport_client.hpp#L51)。
B2-W 官方示例分别使用：

```text
Move(0.3, 0, 0)   forward
Move(0, 0.3, 0)   lateral/panning
Move(0, 0, 0.5)   turn in place
```

见 [`b2w_sport_client.cpp`](https://github.com/unitreerobotics/unitree_sdk2/blob/9754cd153af3da471b0fe5f3aa535e426fb11db3/example/b2w/b2w_sport_client.cpp#L122-L138)。
公开 ROS2 客户端只把三个速度量传给 API 1008，没有公开内部逐轮混控：
[`ros2_b2_sport_client.cpp`](https://github.com/unitreerobotics/unitree_ros2/blob/668d1ec5a05d1c38d3306bdca7d59f2ba3581a88/example/src/src/common/ros2_b2_sport_client.cpp#L37-L46)。

官方遥控资料与原始视频：

- [B2-W Remote Control manual](https://marketing.unitree.com/article/en/B2-W/Remote_Control.html)
- [遥控手册第 7 页：右摇杆左右控制 rotate](https://doc-cdn.unitree.com/static/2025/4/18/d34e574c26b24e8c804f735716928b32_2481x3508.png)
- [官方 B2/B2-W 教学页](https://www.unitree.com/app/b2/)
- [官方英文 wheel-motion 教学视频](https://www.unitree.com/images/c36289843432492a972016894a63eabe.mp4)
- [官方 B2-W 产品页](https://www.unitree.com/b2-w/)

英文教学视频约 `01.8--05.9 s` 是普通 turn-in-place：四轮始终接地，轮面没有
转舵，机身围绕近似固定中心持续偏航；未观察到踏步、抬轮或刻意大幅俯仰/
侧倾。腿有轻微调姿，但视频不能量化法向载荷，也不能反推出闭源控制律。

## 3. 电机顺序与初始差速公式

官方 MuJoCo actuator 顺序和 DDS bridge 给出：

| 官方电机索引 | 轮关节 |
| ---: | --- |
| 12 | FR wheel |
| 13 | FL wheel |
| 14 | RR wheel |
| 15 | RL wheel |

来源：

- [官方 actuator 列表](https://github.com/unitreerobotics/unitree_mujoco/blob/4134cb5dc7ff1ba7f484deda48b5274b58694519/unitree_robots/b2w/b2w.xml#L283-L300)
- [DDS `motor_cmd[i] -> ctrl[i]`](https://github.com/unitreerobotics/unitree_mujoco/blob/4134cb5dc7ff1ba7f484deda48b5274b58694519/simulate/src/unitree_sdk2_bridge.h#L181-L185)
- [README：仿真电机编号与真机一致](https://github.com/unitreerobotics/unitree_mujoco/blob/4134cb5dc7ff1ba7f484deda48b5274b58694519/readme.md#L20-L22)

在官方模型的 `+X` 前、`+Y` 左、`+Z` 上约定下，若有效轮距为 `b`、加载后
滚动半径为 `r`，可用作仿真起点的理想差速关系为：

```text
dq_left  = (vx - b*wz/2) / r
dq_right = (vx + b*wz/2) / r
```

所以纯正 yaw 的初始符号是左轮负、右轮正。按官方电机索引为
`[12:+, 13:-, 14:+, 15:-]`；按当前 Isaac gate 的局部列表
`[FL, FR, RL, RR]` 则是 `[-, +, -, +]`。这只是结构推导出的名义模型，不能
写成 Unitree 已公开的生产混控公式。

## 4. 为什么真机能转，而当前 Isaac gate 几乎不能

当前完整 gate 的左右圆弧方向率均为 100%，轮端 drive 阶段没有 20 Nm 饱和，
但只得到 `1.960 deg` 和 `2.026 deg` yaw。问题不是再次左右翻转，也不是简单
增加轮速增益就能解决。

真实机器人具有：

- 窄而有圆冠的橡胶轮胎；
- 有限接触斑和橡胶形变；
- 纵向滚动与横向 scrub 不同的有效行为；
- 闭环腿部姿态稳定；
- 内置但未公开的轮式运动控制器。

当前 Isaac 资产则使用刚性 `convexHull` 轮胎、近似各向同性接触摩擦和固定腿
目标。四个刚性接触可能互相约束，差速主要被局部 slip 吸收，而没有形成足够
的机身偏航。官方 DAE 视觉网格虽然较密，但当前碰撞已经是 convex hull，不能
把失败简单归因于逐三角形碰撞。

## 5. 下一步实验顺序

所有实验都保留相同质量、轮端力矩限制和命令时长，每次只改变一个因素：

1. 增加纯 `+/- yaw` 原地旋转测试，按关节名生成左右轮命令；同时保留现有
   左右圆弧测试。
2. 以当前 convex hull 为 A；轴向正确的解析圆柱仅作诊断 B；低面数圆冠凸包
   作为更合理候选 C。全宽平底圆柱不是最终真实轮胎。
3. 对比固定腿目标与阻抗/upright 稳定站姿。先验证普通四轮着地转向，不先
   假设必须卸载或抬起内侧轮。
4. 之后再做受控接触参数实验；不能同时降低纵向与横向摩擦来制造表面通过。
5. 每次记录实际 yaw/yaw-rate、轨迹半径、四轮速度误差、接触率、纵向/横向
   slip、轮端力矩、roll/pitch、非轮部件触地和停止距离。
6. 如果可以安全访问真机，在空旷平地用高层 `Move(0,0,+/-wz)` 做低速短时
   标定，记录电机 12--15、12 个腿关节、IMU、本体姿态与轨迹。Z1 保持收拢，
   准备急停，不把未经验证的裸四轮底层命令直接用于高速测试。

若可信的轮胎/contact A/B 仍无法通过，应把首版 TCP tracking 限制为已验证的
前进/倒车 relocation，并将 yaw 委托给单独验证过的 locomotion controller；
不能让 PPO 学习补偿错误物理模型。
