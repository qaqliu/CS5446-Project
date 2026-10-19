# CS5446 Project — Multi-Agent Reinforcement Learning for Railway Traffic Coordination

[中文](#chinese) | [English](#english)

## Demo video

https://github.com/user-attachments/assets/624e2828-ca60-426b-a941-bf6e2033a724

<a id="chinese"></a>

## 项目介绍

这是 CS5446 课程项目的团队仓库。项目使用 Flatland 铁路仿真环境，研究多列车共享轨道时的交通协调问题，重点关注到站率、通行效率、延误和死锁。仓库提供配置驱动的训练管线，让队友按现有接口实现模型、奖励或训练方法，再通过 YAML 组合实验，减少重复编写环境交互和训练流程的工作。

**当前已完成第一版 PPO demo 的完整代码管线**：配置加载、环境与 agent 构建、rollout 采集、TD/GAE 优势估计、PPO 更新、训练日志、checkpoint 保存，以及确定性策略视频导出。当前 demo 使用树深度 3 的共享 PPO 和 champion 奖励，80 epoch checkpoint 在固定 30×30 地图上以确定性策略让 8 辆列车全部到站，整局在第 100 步完成。

项目背景参考 [Flatland 官方竞赛页面](https://www.aicrowd.com/challenges/flatland)、[Flatland Competition 2020](https://proceedings.mlr.press/v133/laurent21a.html) 和 [Flatland 官方文档](https://flatland-association.github.io/flatland-book/)。

## 快速开始

### 安装环境

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate cs5446-project
```

环境使用 Python 3.11、Flatland 4.3.0、PyTorch 2.8.0、TorchRL 0.9.2 和 TensorDict 0.9.1。Linux/Windows 使用 CUDA 12.8 版 PyTorch，macOS 使用无 CUDA 版本；GPU 训练需要兼容的 NVIDIA 驱动。当前主要在 WSL/Linux 下开发和运行，PPO 的采样与优化流程由项目代码实现，动作分布使用 TorchRL 的 `MaskedCategorical`。

```bash
# 已有环境同步依赖
conda env update -n cs5446-project -f environment.yml

# 检查主要依赖和 CUDA
python -c "import torch, torchrl, tensordict, flatland; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 训练与可视化

以下命令均从仓库根目录运行。不传 `--config` 时，三个入口都使用 `config/policy/single_map/ppo_demo_no_intermediate_stops.yaml`。

```bash
# 训练默认 PPO demo
python train_policy.py

# 指定实验配置
python train_policy.py --config config/policy/single_map/ppo_demo_no_intermediate_stops.yaml

# 生成地图预览，无需 checkpoint
python visualize_map.py

# 加载 checkpoint，以确定性策略运行完整 episode 并导出 MP4
python visualize_result.py --fps 7
```

`visualize_map.py` 将地图保存为 `visualize_dir/map_preview.png`。`visualize_result.py` 优先读取 `save_dir/<total_epochs>.pt`，找不到时读取目录中数值最大的 epoch checkpoint，并输出 `visualize_dir/<epoch>.mp4`。可视化使用 checkpoint 中的模型结构和权重、当前 YAML 中的地图与任务配置；改变观测维度时需要保证它与模型输入兼容。

启用 W&B 前，需要将 YAML 中 `wandb.api_key`、`wandb.project` 和 `wandb.entity` 的 `[YOUR_...]` 占位符替换成自己的 API key、项目名称和用户名或团队名称。替换时连同方括号一起去掉，填写为字符串；未加引号的 `[YOUR_...]` 在 YAML 中会被解析为列表，不能直接用于运行。`wandb.name` 是这次实验的显示名称，`wandb.interval` 是记录间隔，单位为外层 epoch。例如：

```yaml
wandb:
  enabled: true
  interval: 10
  api_key: "YOUR_WANDB_API_KEY"
  project: "YOUR_WANDB_PROJECT_NAME"
  entity: "YOUR_WANDB_ENTITY_NAME"
  name: ppo_demo_no_intermediate_stops
```

也可以先执行 `wandb login`，再将 `api_key` 设为 `null`，使用已有登录信息。API key 仅用于本地认证，不要提交到公开仓库。不使用在线日志时设置 `wandb.enabled: false`；没有 CUDA 的机器需要把 `trainer.device` 改成 `cpu`。

## 仓库结构

代码按职责分为模型、agent、奖励和训练四层：模型负责神经网络计算，agent 负责组织各列车的模型与动作选择，奖励组件定义环境反馈，trainer 负责采样和参数更新。配置文件决定各层采用的实现和参数，入口脚本将它们组装成一次实验。

```text
.
├── config/                     # 实验配置：地图、列车、观测、模型、训练与输出参数
├── src/
│   ├── agent/                 # 封装模型，统一独立、共享和集中式 agent 的调用方式
│   │   ├── actor.py           # 组织策略网络，应用动作掩码并采样或确定性选择动作
│   │   ├── critic.py          # 组织状态价值网络，提供各列车的 V(s) 估计
│   │   └── value.py           # 组织动作价值网络，根据 Q(s,a) 进行 epsilon-greedy 动作选择
│   ├── model/                 # 定义网络结构和前向计算，不负责环境交互与训练循环
│   │   ├── registry.py        # 将模型名称映射到模型类，自动发现并按名称构造模型
│   │   ├── policy_model.py    # 定义从观测到动作 logits 的策略网络
│   │   ├── critic_model.py    # 定义从观测到状态价值 V(s) 的评论家网络
│   │   └── value_model.py     # 用于定义从观测到各动作价值 Q(s,a) 的网络
│   ├── reward/                # 定义并注册奖励规则，向训练提供每辆车的标量反馈
│   │   ├── registry.py        # 按名称自动发现并构造奖励类
│   │   ├── rewards.py         # 实现奖励计算，通过统一配置接口读取参数
│   │   └── deadlock.py        # 根据轨道出口和列车占用依赖判断死锁
│   ├── trainer/               # 管理环境采样、学习目标、损失计算和参数更新
│   │   ├── registry.py        # 将训练方法名称映射到 trainer 类并构造实例
│   │   ├── policy_trainer.py  # 定义策略梯度训练流程，组织 rollout、优势估计与策略/价值更新
│   │   └── value_trainer.py   # 用于定义基于 Q 值的采样、学习目标与网络更新流程
│   └── utils/                 # 各模块共用的构建与辅助函数
│       ├── environment.py     # 按配置创建 Flatland 环境，管理地图与任务的 reset 行为
│       ├── agents.py          # 按配置构造网络，并封装为 actor 和 critic
│       ├── action_mask.py     # 根据轨道与列车状态计算合法动作掩码
│       └── seed.py            # 统一设置 Python、NumPy 和 PyTorch 的训练随机种子
├── technical_guide/           # 说明环境概念、观测与动作定义，以及模块接口约定
├── train_policy.py            # 读取配置，组装环境、agent 和 trainer，启动策略训练
├── visualize_map.py           # 根据配置生成地图和列车初始状态的预览图
├── visualize_result.py        # 加载模型权重，运行确定性策略并导出完整 episode 视频
├── environment.yml            # 定义 Conda 环境、CUDA 版 PyTorch 与项目依赖
└── readme.md                  # 项目介绍、使用入口、结构说明与开发计划
```

训练生成的 `checkpoints/`、`wandb/`、`output/` 等目录默认不纳入 Git。更详细的背景和接口说明见 [technical_guide](technical_guide/)，阅读时以当前代码实现为准。

## 当前 demo 与配置说明

当前默认 demo 使用固定的 **30×30 地图、6 个城市、8 列车**，所有列车速度为 1，不设经停站。`map_seed: 42` 控制地图与任务生成，`trainer.training_seed: 42` 控制模型初始化、动作采样及训练随机过程；两者独立设置。`random_map: false` 时，后续 reset 保留同一套轨道、列车任务和时刻表；设为 `true` 时，后续 episode 沿环境随机数序列生成新场景。

观测使用官方归一化展平树，深度为 3，输入维度为 1020。每个树节点包含 12 个特征，深度为 d 时输入维度为 `12 × (1 + 4 + ... + 4^d)`，因此深度 3 和 4 分别对应 1020 和 4092；更改深度时要同步修改 actor、critic 的 `input_dim`。`observation_radius` 用于部分距离特征的归一化，不是树深度或地图裁剪半径。

Actor 和 critic 各自采用 `[64, 64]` 的 ReLU MLP。Actor 输出 5 个动作的 logits，critic 输出一个状态价值。两者是分别实例化、分别优化的网络；配置中的 `shared` 指不同列车共用同一个 actor，以及共用另一个 critic，并不表示 actor 与 critic 共用参数。动作编号为 0 不操作、1 左转、2 直行、3 右转、4 停车；动作掩码屏蔽轨道不允许的移动方向，不负责规划路线或避免所有占用冲突和死锁。

奖励目前使用 `champion`：`0.01 × 距离进展 − 5 × 首次死锁 + 10 × 首次到站`，替代 Flatland 默认评分。奖励参数放在 `reward.config` 下；改为 `reward.type: default`、`reward.config: null` 可使用原生默认奖励。`max_episode_steps: null` 使用 Flatland 时刻表生成的默认局长，当前配置上限为 138 步，也可以在 YAML 中指定正整数。

| 配置项 | 含义 |
|---|---|
| `actor.type` / `critic.type` | `shared` 为列车间共享参数，`independent` 为每辆列车单独持有网络 |
| `actor.model` / `critic.model` | 注册名称、输入维度、隐藏层等模型构造参数 |
| `rollout_steps` | 每轮采集的环境步数，当前为 1024 |
| `total_epochs` | 外层训练轮数，当前为 80；每轮重新采样并优化 |
| `update_epochs` | 同一批 rollout 最多重复优化的轮数，当前为 4 |
| `batch_size` | 每个 minibatch 的有效“时间步、列车”样本数量，当前为 512 |
| `gamma` / `gae_lambda` | 折扣因子与 GAE 参数；当前 lambda 为 0，即一步 TD 风格优势估计 |
| `lr_schedule` | `constant`、`linear`、`cosine`，按外层 epoch 调度，无 warmup |
| `entropy_schedule` | `constant`、`linear`、`cosine`，控制熵奖励系数随训练变化 |
| `target_kl` | 可选 KL 提前停止阈值；`null` 关闭，仅停止当前 rollout 剩余更新 |
| `log_interval` / `wandb.interval` | 分别控制终端和 W&B 的记录间隔，单位为外层 epoch |
| `save_dir` / `checkpoint_interval` | checkpoint 目录与保存间隔，当前每 20 个外层 epoch 保存一次 |
| `visualize_dir` | 地图图片和运行视频的输出目录 |

## 当前 demo 验证结果

以下结果来自同一次 80 epoch 训练的 checkpoint，在其保存的固定地图与任务配置上逐步选择概率最大的合法动作。该结果验证当前场景中的确定性表现，不代表跨地图或列车数量变化后的泛化能力。

| Checkpoint | 确定性到站率 | 完成情况 |
|---|---|---|
| `20.pt` | 7/8（87.5%） | Agent 6 在 138 步时限结束时仍未到站 |
| `40.pt` | 8/8（100%） | 第 100 步全部到站 |
| `60.pt` | 8/8（100%） | 第 100 步全部到站 |
| `80.pt` | 8/8（100%） | 第 100 步全部到站 |

`80.pt` 各列车的到站步数为：Agent 0：38、1：65、2：100、3：87、4：93、5：75、6：35、7：97。训练阶段仍使用随机动作采样，结果应与训练曲线中的窗口平均到站率区分。

## 训练流程与指标

每个外层 epoch 依次执行 `collect_rollout()`、`compute_advantages()` 和最多 `update_epochs` 次 `update_one_epoch()`，随后推进学习率调度、记录日志并按间隔保存 checkpoint。一次 rollout 可以跨越多个 episode，也可以在 episode 中间结束；未结束的 episode 会在下一轮采样中继续。环境推进依赖前一步动作，按时间顺序执行；demo 网络支持批量前向，critic 推理和更新阶段会尽量合并计算。

Actor 使用带 clipping 的 PPO 策略损失减去熵奖励，critic 使用预测状态价值与固定 return target 的均方误差。更新时固定这批 rollout 的旧动作 log probability、优势和 return target，并对梯度进行裁剪。终端和 W&B 的训练指标定义如下：

| 指标 | 统计含义 |
|---|---|
| `episodes` | 从训练开始到当前累计结束的 episode 数，包括因时限结束的 episode |
| `reward` | 统计窗口内已结束 episode 的平均回报；每局先把所有列车的累计奖励相加，再除以列车数 |
| `completion_rate` | 每局实际处于 `TrainState.DONE` 的列车比例，再对窗口内已结束 episode 求平均 |
| `actor_loss` | 包含熵项的 actor loss，按评估的 actor 样本数加权平均 |
| `critic_loss` | 价值预测 MSE，按评估的 critic 样本数加权平均 |
| `entropy` | 有效动作分布的平均熵，按 actor 样本数加权平均 |
| `kl` | 新旧策略的近似 KL，按 actor 样本数加权平均 |
| `actor_lr` / `critic_lr` | W&B 记录当前 epoch 更新时实际使用的 actor / critic 学习率 |
| `entropy_coef` | W&B 记录当前 epoch 实际使用的熵奖励权重 |

训练阶段从动作分布采样，视频阶段选择合法动作中概率最大的动作，因此训练曲线的到站率与确定性视频的到站率可能不同。W&B 初始化时保存除认证密钥外的完整实验配置；终端与 W&B 分别按各自窗口统计。KL 检查触发时，该 minibatch 的指标仍纳入统计，但不执行该 minibatch 的参数更新。

Checkpoint 使用 `<epoch>.pt` 命名，包含 `epoch`、`actor`、`critic` 和完整 `config`，在指定间隔及最后一轮保存。当前不保存 optimizer、scheduler 等恢复状态，也没有接入断点续训入口，因此这些文件主要用于模型加载和结果可视化。

## 如何扩展

新增模型放在 `src/model/`，使用 `@register(name="your_model")` 注册，再在 YAML 的 `actor.model.name` 或 `critic.model.name` 中引用；其余模型字段作为构造参数传入，不需要逐个维护 import。Policy 模型需符合 actor 的观测与信息输入接口并输出 5 个 logits，critic 模型需符合 critic 接口并输出状态价值；可以参考两个 MLP demo，提供 `forward_batch` 可启用对应的批量前向路径。

`src/agent/` 已提供独立、共享与集中式包装接口，当前 PPO demo 主要用于独立和共享网络。集中式方式需要匹配联合输入的模型，不能直接把本地观测 MLP 切换为集中式模型；单个团队价值的 centralized critic 暂不属于当前 PPO demo 的支持范围。`value.py` 已有 Q 值和 epsilon-greedy 接口，但对应模型和 trainer 尚未实现。

奖励放在 `src/reward/`，使用 `@register(name="your_reward")` 注册，并通过 YAML 的 `reward.type` 选择；`reward.config` 整体传给 `__init__(self, reward_config)`，设为 `null` 时使用奖励类的默认参数。现有 `default` 使用 Flatland 默认评分，`champion` 使用 `0.01 × 距离进展 − 5 × 首次死锁 + 10 × 首次到站`，替代默认评分；三项系数分别可通过 `reward.config` 下的 `progress_coef`、`deadlock_penalty`、`finish_bonus` 配置。`champion` 基于 JBR_HSE 团队后续开源奖励组件适配，来源与实现边界见 [ChampionRewards 类说明](src/reward/rewards.py)。环境层通过 `RewardRailEnv` 在 reset 时清理奖励历史、在整步结束后统一检查死锁，避免读取部分更新的列车状态；PPO trainer 接收的仍是每辆车一个标量奖励。

## 当前进度与 TODO

- [x] 搭建配置驱动的环境、模型、agent 和 trainer 构建流程。
- [x] 实现 Actor、Critic 与 ValueAgent 接口，以及模型和 trainer 注册器。
- [x] 完成共享 MLP actor/critic 的 PPO demo，包括 TD/GAE、minibatch 更新、动作掩码、调度器与可选 KL 提前停止。
- [x] 接入 tqdm、终端日志、W&B、checkpoint 保存、地图预览和确定性策略视频导出。
- [ ] **设计更好的奖励**：探索距离进展、到站奖励、死锁惩罚等可能的奖励设计，评估其对学习效果与列车行为的影响。
- [ ] **比较独立网络与共享网络**：在相同地图、列车数和环境交互预算下，比较完成率、通行效率、学习速度和稳定性，并观察冲突加剧时的差异。
- [ ] **研究泛化能力**：测试共享策略在不微调的情况下适应列车数量变化和新地图的能力，并比较固定地图训练与随机地图训练。
- [ ] **探索基于价值的方法**：补齐 Q 网络和训练管线，在相同交互预算下比较 epsilon-greedy 方法与 PPO 的完成率、效率、稳定性和探索行为。

---

<a id="english"></a>

## About the project

This CS5446 team project studies railway traffic coordination with multiple trains sharing tracks in Flatland. The repository provides configurable environment construction, agent interfaces, registered models, and training pipelines so that teammates can implement components without rebuilding the entire interaction loop.

**The first PPO demo pipeline is implemented**, including rollout collection, TD/GAE advantages, PPO optimization, logging, checkpoints, map previews, and deterministic video export. The current demo uses shared PPO, depth-3 observations, and champion-style rewards. Its 80-epoch checkpoint reaches 100% completion (8/8 trains) on the fixed 30×30 map in 100 environment steps with deterministic actions.

## Setup and usage

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate cs5446-project
python train_policy.py
python visualize_map.py
python visualize_result.py --fps 7
```

All three scripts accept `--config` and default to `config/policy/single_map/ppo_demo_no_intermediate_stops.yaml`. The environment uses Python 3.11, Flatland 4.3.0, PyTorch 2.8.0, TorchRL 0.9.2, and TensorDict 0.9.1. Linux/Windows use CUDA 12.8 PyTorch wheels; set `trainer.device: cpu` for CPU execution. Before enabling W&B, replace the `[YOUR_...]` placeholders in `wandb.api_key`, `wandb.project`, and `wandb.entity` with your API key, project name, and username or team name. Remove the square brackets and supply string values, as shown in the YAML example above: unquoted bracketed placeholders are parsed as lists. `wandb.name` sets the run display name, and `wandb.interval` sets the logging interval in outer epochs. Alternatively, run `wandb login` and set `api_key: null` to use existing credentials. Keep real API keys out of public commits. Set `wandb.enabled: false` to disable online logging.

Map previews are saved to `visualize_dir/map_preview.png`. Policy visualization loads `<total_epochs>.pt`, falling back to the largest numeric checkpoint filename, and writes `visualize_dir/<epoch>.mp4`. It uses the saved model architecture and weights with the current YAML scenario, which must provide compatible observations.

## Repository and design

The repository separates network computation, agent organization, reward definition, and training. In `src/model/`, `policy_model.py` maps observations to action logits, `critic_model.py` maps observations to state values, and `value_model.py` is the location for action-value networks. In `src/agent/`, the corresponding wrappers organize independent, shared, or centralized models and expose value estimation or action selection. `src/reward/` is the location for scalar reward rules and their composition. In `src/trainer/`, `policy_trainer.py` organizes rollout collection, advantage estimation, and policy/value updates; `value_trainer.py` is the location for Q-learning targets and updates. Registries construct models and trainers by name. `src/utils/` handles environment construction/reset, agent construction, action masks, and random seeds. YAML files select experiment components and parameters; entry scripts assemble training or visualization runs. Further background is available in [technical_guide](technical_guide/).

The default demo uses a fixed 30×30 map, six cities, eight unit-speed trains, and no intermediate stops. A depth-3 normalized flattened tree supplies 1020 features to separate policy and critic MLPs with hidden widths `[64, 64]` and ReLU activations. The policy produces five action logits; the critic produces one state value. Shared networks share parameters across trains, while actor and critic retain separate parameters. The legal-action mask does not provide conflict-free routing or deadlock prevention.

Map generation and training use separate seeds. Fixed-map resets preserve rails, train tasks, and timetables; random-map resets generate subsequent scenarios from the seeded RNG sequence. A null episode limit uses Flatland's generated limit, currently 138 steps for this configuration. The demo selects `champion` rewards: `0.01 * distance_progress - 5 * first_deadlock + 10 * first_arrival`, replacing default scoring. Parameters are nested under `reward.config`; use `reward.type: default` and `reward.config: null` to select native default scoring.

## Demo evaluation

Deterministic evaluation on the saved fixed scenario gives 7/8 arrivals for `20.pt` and 8/8 arrivals for `40.pt`, `60.pt`, and `80.pt`. All three successful checkpoints finish the episode in 100 steps, within the 138-step limit. With `80.pt`, arrival steps for agents 0–7 are `[38, 65, 100, 87, 93, 75, 35, 97]`. These are fixed-scenario results; transfer across maps and train counts remains future work.

## Training and outputs

Each outer epoch collects `rollout_steps` environment transitions, computes fixed advantages and return targets, then reuses the rollout for up to `update_epochs` optimization passes. A minibatch contains individual valid time–train pairs. Rollouts may span multiple episodes, and unfinished episodes continue across rollout boundaries. The default `gae_lambda: 0` gives one-step TD advantages; larger values enable multi-step GAE. The default run uses 80 outer epochs, 1024 environment steps per rollout, four update passes, and minibatches of 512 time–train samples.

The actor uses the clipped PPO objective with an entropy bonus; the critic uses value MSE. Gradient clipping, optional KL early stopping, and constant/linear/cosine learning-rate and entropy schedules are available. Learning-rate schedules advance once per outer epoch, without warmup.

Logs report cumulative ended episodes, mean per-train episode return, completion rate, actor and critic losses, entropy, and approximate KL. Reward and completion statistics average ended episodes in the logging window; losses, entropy, and KL use their respective evaluated sample counts. Completion requires `TrainState.DONE`; reaching the time limit alone does not count as arrival. Training samples actions, while video evaluation selects the highest-probability legal action, so their completion rates can differ. W&B also records `actor_lr`, `critic_lr`, and `entropy_coef` as used in the logged epoch, before the learning-rate schedulers advance. It records the experiment configuration excluding its authentication key.

Checkpoints contain `epoch`, actor weights, critic weights, and the full configuration, saved as `<epoch>.pt` every 20 outer epochs in the default configuration and at the final epoch. Optimizer/scheduler restoration and training resume are not implemented.

## Extending the demo

Add models under `src/model/`, register them with `@register(name="your_model")`, and reference their names and constructor parameters in YAML. Follow the existing actor/critic model interfaces; an optional `forward_batch` enables batched inference. The current PPO demo focuses on independent and shared networks; centralized models require joint-input implementations, and a single team-value critic is not supported by this demo.

Register scalar rewards under `src/reward/` with `@register(name="your_reward")` and select them through `reward.type`; the complete `reward.config` mapping is passed to `__init__(self, reward_config)`, with `null` selecting the reward class defaults. `default` uses Flatland scoring, while `champion` replaces it with distance progress (0.01), a one-time deadlock penalty (-5), and a one-time arrival bonus (+10). Coefficients are configurable as `progress_coef`, `deadlock_penalty`, and `finish_bonus` under `reward.config`. This adapts the JBR_HSE team's later open-source components; see [the source and adaptation notes on ChampionRewards](src/reward/rewards.py). `RewardRailEnv` resets reward history and runs global shaping after the complete environment step. PPO remains unchanged.

## Progress and TODO

- [x] Config-driven construction, model/trainer registries, and actor/critic/value-agent interfaces.
- [x] End-to-end PPO demo with logging, checkpoints, map rendering, and deterministic video export.
- [ ] Investigate reward designs, such as distance progress, arrival bonuses, and deadlock penalties, and evaluate their effects on learning and train behavior.
- [ ] Compare independent and shared networks under matched scenarios and environment interaction budgets.
- [ ] Study zero-shot transfer across train counts and maps, including fixed-map versus random-map training.
- [ ] Implement and evaluate value-based learning with epsilon-greedy exploration against PPO under the same interaction budget.
