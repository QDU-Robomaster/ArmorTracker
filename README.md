# ArmorTracker

`ArmorTracker` 只负责装甲板观测关联、EKF 状态估计和目标主题发布，不再承担弹道解算。

## Runtime Role

- 输入: `armor_detector/results` 和相机图像
- 输出: `tracker/info`、`tracker/metrics`、`tracker/target`
- 下游: `Aimer` 订阅 `tracker/target`，完成瞄准点选择和弹道迭代

## Debug

- 打开预览: 在 `User/xrobot.yaml` 或 `User/xrobot_demo.yaml` 中设置 `armor_tracker.debug.preview: true`
- 关键面板: 状态机、`jumped` 标志、NIS、目标 EKF 状态、候选装甲板匹配结果

## Legacy Files

- `SolveTrajectory.*` 和 `TableGenerator.cpp` 仅保留作历史参考
- 当前运行时构建不会链接这些旧弹道文件
