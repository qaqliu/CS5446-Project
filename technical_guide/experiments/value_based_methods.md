# Value-based methods / 价值方法

[中文](#中文) | [English](#english)

## 中文

本模块实现最终实验使用的共享参数 PQN，及其等预算 PPO 对照。
`value_pqn_scale20.yaml` 与 `value_ppo_scale20.yaml` 是最终等预算实验配方：
Flatland 4.3、20 车、32×32、200 张种子专属训练地图、300 万联合环境步，
深度 3 树观测（1020 维），距离特征归一化尺度 20，无 predictor。
奖励为进展 0.01、首次死锁 −0.5、首次到站 +10。
`observation_radius` 控制距离归一化，不增加树深度或物理观测范围。

### 实现位置

| 位置 | 职责 |
|---|---|
| `src/model/value_model.py` | 注册 `mlp_q`；64/64 MLP，可选 LayerNorm |
| `src/trainer/value_trainer.py` | 注册 `pqn`，环境采样与网络更新 |
| `src/trainer/value_targets.py` | 合法动作约束下的 Q(λ) 目标 |
| `src/trainer/exact_ppo.py` | 精确步数的 PPO 对照，跨 checkpoint 保留 episode 片段 |
| `src/eval/value.py` | checkpoint 加载、贪心执行、预设 ε=0.05 附加执行 |
| `scripts/train_value.py` | 配置、种子、训练/验证、checkpoint 和来源记录 |

PQN 在 4 个同步环境上收集新轨迹，以冻结的 Q(λ) 目标重复回归；没有目标网络或
历史经验回放。使用 LayerNorm、λ=0.65，训练 ε 在 150 万步内从 1 降到 0.05。
不在网络或 agent 中实现奖励、训练循环或验证逻辑。

### 训练

使用仓库 `environment.yml` 创建 Python 3.11 / Flatland 4.3 环境。
从仓库根目录运行；PPO 原有入口继续可用，价值方法使用以下入口。

```sh
python -m scripts.train_value --config config/protocol/value_pqn_scale20.yaml \
  --seed 42 --out results/runs/value_pqn_scale20_seed42
python -m scripts.train_value --config config/protocol/value_ppo_scale20.yaml \
  --seed 42 --out results/runs/value_ppo_scale20_seed42
```

完整对照还需使用种子 43、44，并相应修改输出目录名。两个方法在同一训练种子下
使用相同 200 张地图池。PQN 多环境分配不同的 episode 流；环境步数为所有环境
`env.step` 的总数，列车 transition 和 optimizer update 单独计数。
训练期间的评估只使用验证种子 2000–2019，不能读取测试图调参。
完整比较应先完成六个模型的预算并冻结模型与执行设置，再统一运行测试。
本入口中 `checkpoint_interval` 的单位是联合环境步；配方设为 100000，产生
0、100000、…、3000000 共 31 个候选 checkpoint。

短工程检查可追加 `--steps 64 --checkpoint-interval 64 --smoke-validation`。
它使用完整模型进行更新、checkpoint 往返和两张验证图；短跑不用于衡量学习性能。
输出目录必须不存在。`run_metadata.json` 记录源码哈希、Git 状态、配置哈希和版本。
W&B API key 不写入配置快照或 checkpoint。

### 评测

使用上游评测入口与同一任务/评分协议。以下命令先在验证集选择模型，再评测标准图。

```sh
python -m scripts.evaluate results/runs/value_pqn_scale20_seed42 \
  --loader src.eval.value:load_checkpoint --require-env-steps 3000000 --jobs 1
```

PPO 可使用同一 loader，它会识别 actor checkpoint。Q 主执行始终贪心；标准评测器
的三条 sampled-mode Q 记录是确定性重复，不能当作独立训练种子或 ε 探索样本。
`best_val` 与 `final` 分开报告。文件名中的数字在此训练入口表示环境步数，
评测器还检查 checkpoint 内部计数，不能用重命名文件绕过预算检查。

复现本次确认集时，可在上述命令中追加：

```sh
--test-seed-start 9300000 --test-seed-count 20 --output-name evaluation_fresh.csv
```

该范围已用于发表结果；新调参实验应另行预留确认集。测试种子必须与验证集、
checkpoint 中的训练池分离。可选 `--include-100` 不属于本次已完成的必需评测。

预设 ε=0.05 附加实验使用同一贪心验证选出的 checkpoint，只改变测试执行：

```sh
python -m scripts.evaluate results/runs/value_pqn_scale20_seed42 \
  --loader src.eval.value:load_checkpoint --test-loader src.eval.value:load_epsilon_checkpoint \
  --require-env-steps 3000000 --test-seed-start 9300000 --test-seed-count 20 \
  --output-name evaluation_fresh_epsilon.csv --jobs 1
```

输出记录 `policy_loader` 与 `execution_epsilon`。附加执行、标准集和确认集使用不同
CSV，不能混成一个主结果。上游报告器可从默认 `evaluation.csv` 生成 Task 4 对照；
结果目录名应与配方名一致，例如 `value_pqn_scale20_seed42`。

```sh
python -m scripts.html_report results/runs --out results/value_report.md
```

checkpoint 仅供评测，未保存 optimizer、回放池、环境或 RNG 的精确恢复状态。
本入口要求 fresh initialization，拒绝 `init_from`；上游原有课程学习入口保持独立。

### 验证与结果

```sh
python -m pip install -r requirements-dev.txt
make check
```

`PYTHON=/path/to/python` 可指定现有解释器。测试覆盖终端/到站 padding、合法动作、
Q 目标、精确预算、PPO 片段、验证 RNG 隔离、Q/PPO loader 和附加执行
不改变 checkpoint 选择。训练/评测产物留在忽略的 `results/`、`output/` 中。

最终研究结果与数据来源见 [独立评测报告](../../docs/report/value_based/README.md)。
历史诊断脚本、旧 Flatland 版本和 Slurm 私有路径未纳入日常接口；冻结原始运行仍保留
用于审计。集成整理没有重新运行 18M 步正式训练。

### 完整复算已报告的评测表

使用单独保存的原始包 `completed_evaluation_20261105.tar.gz`，其 SHA-256 在
`docs/report/value_based/provenance.json` 中。原始包约 9.3 MB，包含评测记录和
冻结 checkpoint，不随源码提交。

```sh
python -m scripts.verify_value_results --bundle /path/to/completed_evaluation_20261105.tar.gz \
  --out results/value_reproduction
```

该命令验证包和模型哈希、数据划分、checkpoint 角色、场景一致性、零干预和零更新，
从逐列车到站时刻复算 7824 个回合，按地图及训练种子重新汇总全部 64 组结果，
并要求输出 CSV 与已报告数据逐字节一致。成功后生成 `summary.csv` 和 `audit.json`。
它不执行神经网络推理或重训，也不从训练曲线重新选择模型，而是核对测试前已冻结的
角色。这与“重新执行全部策略评测”和“从头重训 18M 步”是三个不同的验证层次；
当前集成仅对后两者做了案例执行与短程训练回归，没有完成全量重跑。

## English

Use `scripts.train_value` for the registered PQN trainer and the
exact-budget PPO control. The paired scale20 recipes specify the same 20-train
task, reward, 200-map pool and 3M interaction budget; repeat with training seeds
42/43/44. `mlp_q` uses local flattened tree observations, a shared 64/64 MLP and
optional LayerNorm. PQN fits frozen Q(lambda) targets from fresh synchronous
trajectories without replay or a target network.

The commands above cover training, standard testing, the recorded confirmation
cohort and the prespecified test-time epsilon ablation. Keep the primary loader
for validation and pass the exploratory loader only through `--test-loader`;
otherwise checkpoint selection would change. Greedy Q duplicates in sampled
mode are not independent stochastic evaluations. Keep datasets, checkpoint
roles and execution settings in separate files/tables. Confirmation seeds
9300000–9300019 are now exposed and should not validate new tuning decisions.

Exact budgets count joint environment ticks, not per-train transitions. PPO
episode fragments survive intermediate checkpoints and are joined before the
normal episode update. Finite-horizon endings are terminal for both methods.
Validation restores training RNG state. Checkpoints are evaluation artifacts;
the matched driver rejects warm starts and does not promise exact resumption.

Run `make check` after installing the optional development requirements. The
runtime environment remains `environment.yml`. Extend focused regressions when
changing terminal masks, targets, budget boundaries or evaluation
selection. Preserve upstream reward attribution. Source/config hashes identify
the actual implementation even when the checkout has uncommitted changes.

`scripts.verify_value_results` regenerates the reported held-out evaluation table from the
checksum-pinned raw bundle, with exact CSV comparison and independent arrival
score checks. This verifies recorded results; it does not rerun neural policy
inference, reselect checkpoints from training curves or retrain models. Full
18M-step retraining and full inference replay have not been completed on this
integration; short training and archived-case parity are narrower checks.
