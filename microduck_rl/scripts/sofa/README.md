# 沙发任务（ModelScope 运行手册）

用魔搭（ModelScope）的 GPU 实例训练 **MicroDuck 在沙发上行走** 的策略。

官方仓库的训练入口是 `uv run train`，跟 HuggingFace 没有耦合——HF 只负责「托管 GPU + 存 checkpoint」。本目录提供等价的魔搭路径：

| 官方 HF 路径 | 本目录的魔搭路径 |
|---|---|
| `--hf-jobs` 提交托管 Job | 在魔搭 Notebook 的终端里跑 `scripts/sofa/modelscope_run.sh` |
| checkpoint → `<ns>/<run>` 模型仓库 | `scripts/sofa/ms_upload.py` → `cartman/microduck-sofa-walk` |
| 源码打包上传 | 直接上传本仓库（约 20 MB，主要是 STL） |
| wandb 曲线 | `--agent.logger tensorboard`，日志在 `logs/`（不需要 wandb 账号） |

任务 id：`Mjlab-Velocity-Sofa-MicroDuck`
观测/动作契约与官方一致：`obs[1,61] → actions[1,14]`，训出来的 ONNX 可以直接热切换进 `robotd`。

---

## 1. GPU 怎么选

**只用单卡**（rsl_rl 不做多卡并行，多卡实例的其余卡纯属浪费）。

| 配置 | 建议 | 说明 |
|---|---|---|
| **A100 80GB ×1** | **首选** | 可跑 `NUM_ENVS=8192`，4096 envs/4000 iter 大约 40–60 分钟 |
| A100 40GB ×1 | 推荐 | `NUM_ENVS=4096`，约 1 小时 |
| A10 24GB ×1 | 可用（官方 l4x1 同档） | `NUM_ENVS=2048`，约 1.5–2.5 小时 |
| 多卡实例 | 不要 | 训练不会变快 |

其余：CPU ≥ 8 核、内存 ≥ 32 GB（4096 envs 建议 64 GB）、镜像选 **CUDA 12.8 + PyTorch 2.9** 那档（和仓库 pin 的 `torch==2.9.1` 对齐，系统 torch 不会被用到，但驱动版本够新很重要）。

> 显存不够的典型症状：启动即 `CUDA out of memory` 或 warp 报 `nconmax` 相关错误。直接 `NUM_ENVS=1024` 重跑，不要改别的。

## 2. 把代码放上去

任选其一：

- **上传压缩包**（最稳）：把 `microduck_sofa_bundle.tar.gz` 拖进 Notebook 文件浏览器，然后
  ```bash
  mkdir -p /work && tar -xzf microduck_sofa_bundle.tar.gz -C /work && cd /work/microduck_rl
  ```
- **直接 git clone**（魔搭机器在阿里云上，GitHub 通常可达；不通就用上面的包）
  ```bash
  cd /work && git clone -b develop https://github.com/pollen-robotics/microduck_rl.git
  ```
  注意：clone 下来的是**官方版**，没有沙发任务；必须用压缩包或把本目录的改动同步过去。

## 3. 开跑

```bash
cd /work/microduck_rl
export MODELSCOPE_API_TOKEN=ms-...        # 只在 upload 阶段需要

bash scripts/sofa/modelscope_run.sh check   # 环境自检 + CPU 沙发物理验证（约 1 分钟）
bash scripts/sofa/modelscope_run.sh smoke   # 64 envs × 5 iter（几分钟，抓 cfg 错误）
bash scripts/sofa/modelscope_run.sh full    # 全量训练
bash scripts/sofa/modelscope_run.sh verify  # ★ 在训练环境里扫指令，确认它真的会走
bash scripts/sofa/modelscope_run.sh export  # 最新 checkpoint → policy.onnx（含形状校验 + 指令敏感性探测）
bash scripts/sofa/modelscope_run.sh upload  # checkpoint + onnx → 模型仓库（探测不过会拒绝上传）
```

> `verify` 和 `probe` 是 2026-11-12 那次“训出站桩策略”事故之后加的两道闸门，详见第 8 节。**不要跳过 `verify`**：它能在导出前、用训练环境本身判定「这个 checkpoint 到底会不会走」。`upload` 还会再跑一次纯 ONNX 的 `probe`，不过就拒传（`FORCE_UPLOAD=1` 可强行覆盖）。

常用环境变量：

```bash
NUM_ENVS=4096 ITERS=4000 bash scripts/sofa/modelscope_run.sh full
GH_PROXY=https://ghfast.top/ bash scripts/sofa/modelscope_run.sh check   # bam 依赖拉不动时
```

**`smoke` 一定要先跑**：官方 `AGENTS.md` 说这一步能抓住 95% 的配置错误，成本几分钟。

## 4. 各阶段做什么

- **check**：打印 GPU/CPU/磁盘 → 装 `uv` → 保证 Python 3.12（镜像是 3.11，uv 走南大镜像下载）→ `uv sync`（默认**阿里云源**——魔搭实例在阿里云机房，内网最快；可用 `UV_INDEX_URL` 换成清华源等；GitHub 拉 `bam` 失败时自动换代理重试）→ 跑 `scripts/sofa/validate_sofa_physics.py`（真机模型 + 真沙发，CPU 上验证软接触、穿模、接触预算）
- **smoke / full**：`train Mjlab-Velocity-Sofa-MicroDuck --env.scene.num-envs N --agent.max_iterations M --agent.logger tensorboard`（实际执行 `.venv/bin/train`，见第 12 节：不经过 `uv run`）
- **verify**：先跑 `ckpt_report.py` 溯源（旧配置训练的直接 exit 3 中止），再跑 `scripts/sofa/verify_mjlab.py`——在**训练环境本身**里跑一串指令（默认 ±0.20 m/s，2.5 s/条），测实际达成的车体速度，逐条打印 `achieved/commanded` 与躯干高度；任一指令达不到 55% 或掉下座面即判失败
- **report**（`watch` 同义）：`scripts/sofa/ckpt_report.py`——从 run 目录的 `params/env.yaml` 读出**训练时的真实配置**（track std / head_pose 权重 / 指令保持时长 / `rel_forward_envs` / 接触传感器 secondary），再从 tensorboard 事件文件读出 `air_time`、`fell_off_sofa`、tracking 奖励的首末值。退出码 3 = 该 checkpoint 是 pre-fix 配置训的
- **export**：`scripts/export.py --checkpoint-file <pt> --num-envs 1 --onnx-file policy.onnx`，断言 `obs[1,61]/actions[1,14]`，随后自动跑 `probe`（不通过只告警）
- **probe**：`scripts/sofa/probe_policy.py`——纯 ONNX、不用 GPU/仿真：只改观测里的指令槽（`obs[48:51]`），看动作变化幅度。官方 walker 的敏感性 ≈0.91，站桩策略 ≈0.08
- **upload**：模型仓库 `cartman/microduck-sofa-walk`，checkpoint 在 `run/`，ONNX 在根目录；上传前再跑一次 `probe` 作为硬闸门

产物：`logs/rsl_rl/velocity_sofa/<时间戳>_v2_bounded/model_*.pt`、`policy.onnx`。

## 5. 沙发任务改了什么

只改场景 + 少量任务旋钮，奖励/DR/课程/网络完全是官方 velocity 配方（详见
`src/mjlab_microduck/tasks/microduck_velocity_sofa_env_cfg.py` 的 docstring）：

- 沙发作为 fixed-base entity（mjlab 自动包成 mocap），每个 env 一份，按 env origin 摆放
- 座面 `solref` 0.06（资产默认 0.02，家具级硬接触）→ 模拟泡沫下陷
- 座面碰撞 geom 移到 group 0（否则足高射线穿过座面打到 0.5 m 下的地板）
- 脚部接触传感器的 secondary 从 `terrain` 改指向沙发体（`entity="sofa"` + `^odg_sofa_concept_a$`，mjlab 会在 sofa 实体内解析正则并写成 `sofa/odg_sofa_concept_a`）——原来指向地板，站在座面上永远不接触，`air_time`（权重 3.0）直接失效
- 出生点：座面**居中**（y ±0.20，x -0.18~-0.08）、朝沙发长边（±y），`z = 座面 0.518 + 站立高度`
- 指令范围 ±0.20 m/s、±0.10 m/s 横向、±0.8 rad/s 转向（1.56 × 0.62 m 的可用座面），推力减半到 ±0.15 m/s
- **指令保持时长 1.2–2.4 s**（stock 是 3–8 s）：0.20 m/s × 2.4 s = 0.48 m，才装得进 0.58 m 的余量（第 11 节）
- `rel_forward_envs = 0`：stock 的 0.2 会把 20% 的 env 命令成 0.30 m/s，超出本任务范围
- **防“站桩”配方**（第 8 节）：跟踪奖励 std 0.316→0.12、head_pose 权重 2.0→0.3、站立 env 课程上限 25%→5%
- 新增终止条件：躯干低于座面 0.2 m（走出座面即结束，算失败）

## 6. 训完怎么用

1. 把 `policy.onnx`（约 8 MB）从 Notebook 下载下来，或从模型仓库拉
2. 本机装进仿真鸭子（和装 flamingo 一样的流程）：
   ```bash
   scripts/duck-sim ctl policy --file ~/.cache/duck-sim/robotd-duck-a.toml \
       add sofa-walk /path/to/policy.onnx
   ```
3. 想让它成为默认步态：把 `~/.cache/duck-sim/robotd-duck-a.toml` 里的 `walk` 指向这个 onnx，重启 `duck-sim`
4. 真机：先扶着试，`robotd` 的 fall predictor 会在倾倒时自动瘫软

## 7. 常见问题

| 症状 | 处理 |
|---|---|
| `uv sync` 卡在 `Rhoban/bam` | `GH_PROXY=https://ghfast.top/`（或 `https://ghproxy.net/`）重跑 |
| 大包（torch/cudnn）中断后重新下 | uv **没有字节级断点续传**：半个 wheel 直接丢弃。终极方案：`bash scripts/sofa/modelscope_run.sh prefetch` —— 按锁文件清单（`wheels_linux_x86_64.txt`，140 个 wheel 全部）用 curl `-C -` 续传到 `.wheelhouse`，每个包 sha256 校验，阿里云镜像优先、官方源兜底 |
| `.wheelhouse` 下完后 `uv sync` 仍要联网下 torch | **`--frozen` 按 URL 取包**：`--find-links`（只在解析阶段用）和 `--offline`（按 URL 查缓存）都无法喂本地 wheel。脚本的 `localize_lock` 会把 lock 里 140 条 URL 改写成 `file://$WHEELHOUSE/<file>` 再 `uv sync --frozen`（sha256 仍校验）；原始 lock 备份在 `.wheelhouse/uv.lock.remote`。注意：**不要运行 `uv lock`**，若需恢复网络模式，把该备份拷回 `uv.lock` 即可 |
| 下载太慢 | `UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/`（魔搭在阿里云机房，走自家镜像最快，新版脚本已默认） |
| 卡在 `Downloading cpython-3.12...` | uv 的 Python 解释器走的是 GitHub，脚本已默认改用南大镜像（`UV_PYTHON_INSTALL_MIRROR`）；若仍超时可 `conda create -n py312 python=3.12 && conda activate py312` 后重跑 |
| `no python 3.12` / uv 下载超时 | 脚本会尝试 conda；也可手动 `conda create -n py312 python=3.12` 后 `conda activate py312` 再跑 |
| `CUDA out of memory` | `NUM_ENVS=1024`（A10 建议 2048 起） |
| 训练 reward 不涨、原地不动 | 先看 `Episode_Reward/air_time` 是不是一直 0（说明根本没抬脚），再看 `Episode_Termination/fell_off_sofa` 是不是一直 0（说明它从没走过座面）——两者同时为 0 就是典型的"站桩最优解"，见第 8 节 |
| 训出来“能站不会走” | 跑 `modelscope_run.sh probe policy.onnx`：线性指令敏感性 < 0.35 就是站桩策略，重训时按第 8 节改奖励 |
| `export` 报找不到 checkpoint | 确认 `logs/rsl_rl/velocity_sofa/*/model_*.pt` 存在；smoke 只跑 5 iter，`save_interval=100` 可能还没存 |
| 上传报 401 | 令牌无效/过期：modelscope.cn → 个人中心 → 访问令牌 |

## 8. 事故复盘：2026-11-12 那次训出的是「站桩策略」

**现象**：4000 iter 正常跑完，曲线也漂亮（`track_linear_velocity` 1.56/2.0、`upright` 1.85），
但装进仿真鸭子后：

- 平地上任何指令（0.08 / 0.3 / **1.0** m/s）实测位移都是 **0.00 m/s**，躯干高度恒定 115 mm
- 沙发上能站 60 s 不动，但一给指令就往一侧倒（不是因为走，是因为它压根不迈步）
- `probe_policy.py` 测出线性指令敏感性 **0.077**（官方 `alpha_walking` 是 **0.914**，差 12 倍）

**排除了什么**（这些都没问题，别再怀疑）：

| 环节 | 检查方式 | 结论 |
|---|---|---|
| ONNX 导出 | ONNX 元数据里的 `checkpoint_sha256` 与 `model_3999.pt` 实际 sha256 | 完全一致 |
| 观测布局 | 元数据 `observation_names` = base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command, head_command, body_command | 61 维，与部署端逐段对齐 |
| 默认关节角 | 元数据 `default_joint_pos` vs `infer_policy.py` 的 `DEFAULT_POSE` | 14 个关节逐个相同 |
| 动作尺度 | 元数据 `action_scale = 1.0` vs 训练侧 `joint_pos_action.scale` | 一致 |
| 部署脚本 | 用官方 `alpha_walking.onnx` 跑同一脚本 | 会走（0.3 指令 → 0.11 m/s），脚本没坏 |

**根因**：任务被写成了「站着最划算」。

1. 指令范围收到 ±0.12 m/s，而跟踪奖励是高斯型（权重 2.0、`std=sqrt(0.1)=0.316`）——
   **原地不动也能拿到的跟踪奖励是 exp(-(0.12/0.316)²)=87%**；
2. 同一份奖励里还有 `head_pose_tracking` 权重 2.0，而"站着一动不动"恰好是这个项的最优解；
3. 站立 env 课程把 25% 的并行环境指令置零，等于用四分之一的样本反复教它"冻结"；
4. 脚部接触传感器还指着地板 `terrain`，站在座面上永远不触发 → `air_time`（权重 3.0）全程 0.000，
   **唯一鼓励抬脚走路的那一项是死的**。

于是 PPO 找到了全局最优：蹲站不动。曲线好看是因为它确实把能拿的分都拿到了。

**修复**（已在 `microduck_velocity_sofa_env_cfg.py` 里落地）：

| 改动 | 值 | 作用 |
|---|---|---|
| 指令范围 | ±0.12 → **±0.20** m/s（横向 ±0.10，转向 ±0.8） | 站桩能拿的跟踪奖励从 87% 降到 exp(-(0.2/0.12)²)=6% |
| 跟踪 std | 0.316 → **0.12** | 拉大"站着"和"走着"的奖励差 |
| `head_pose_tracking` 权重 | 2.0 → **0.3** | 别让它和行走抢同一组动作 |
| 站立 env 课程 | 25% → **5%** 上限 | 少教它"冻结" |
| 足部接触传感器 | `terrain` → **沙发体** | 让 `air_time` 活过来，抬脚才有分 |

**新增的两道闸门**（就是为了这类问题不再流到部署端）：

- `verify`（训练环境内）：扫 ±0.20 m/s 等指令，逐条打印 `achieved/commanded`，<55% 直接判失败
- `probe`（纯 ONNX，秒级）：只扰动 `obs[48:51]` 的指令槽，看动作是否跟着变；`upload` 前硬性拦截

**下次重训**：

```bash
bash scripts/sofa/modelscope_run.sh full    # full 内置 CPU 构建闸门（build_check）
bash scripts/sofa/modelscope_run.sh verify  # 必须 PASS
bash scripts/sofa/modelscope_run.sh export
bash scripts/sofa/modelscope_run.sh upload
```

如果 `verify` 里仍然偏低但明显在动（比如达到 40~50%），说明是"走得慢"而不是"不走"，
可以先把 `--min-ratio` 放到 0.4 观察，再决定是否继续调奖励。

## 9. 事故复盘：2026-11-13 训练命令直接崩溃（ContactMatch 正则）

**现象**：`train` 在构建环境时立即崩溃（没浪费 GPU 时间）：

```
ValueError: Error: unrecognized name '.*odg_sofa_concept_a.*' of object
Element name 'feet_ground_contact_left_foot_collision_found', id 6
```

**根因**：读了 mjlab 1.3.0 源码（`sensor/contact_sensor.py`）确认——
`ContactMatch` 的 `pattern` 是否按正则解析，**取决于有没有填 `entity`**：

| 写法 | 行为 |
|---|---|
| `ContactMatch(mode="body", pattern=r".*odg_sofa_concept_a.*")`（**无 entity**） | pattern 被当成**字面 MuJoCo 名**原样写进 sensor 的 `refname`，正则符号根本不会被展开 → `MjSpec.compile` 找不到这个 body，报上面的错 |
| `ContactMatch(mode="body", pattern=r"^odg_sofa_concept_a$", entity="sofa")`（**有 entity**） | 正则先在 sofa 实体内部用 `ent.find_bodies()` 解析，得到具体名后再自动加 `sofa/` 前缀写进 spec → 编译通过 |

`body_names` 属性会剥掉 `entity/` 前缀（`split("/")[-1]`），所以正则匹配的是**不带前缀**的名字；
而场景合并用 `spec.attach(ent.spec, prefix="sofa/")`，编译后的 body 名是 `sofa/odg_sofa_concept_a`。

另外当时还写了个非法值 `secondary_policy = "all"`（合法值只有 `first`/`any`/`error`），
"all" 会被静默当成字面直通路径，掩盖了问题。

**验证**：`mjlab_src/test_contact_refname.py`（本地 MuJoCo 复现）证明——
① 无 entity 的正则写法在 `MjSpec.compile` 复现出一模一样的报错；
② `attach(prefix="sofa/")` 后用解析好的字面名 `sofa/odg_sofa_concept_a` 做 refname 能编译，
且 `sensor_refid` 精确指向该 body。

**修复**：

1. `microduck_velocity_sofa_env_cfg.py`：secondary 改为
   `ContactMatch(mode="body", pattern=r"^odg_sofa_concept_a$", entity="sofa")`，
   `secondary_policy="first"`
2. 新增 `scripts/sofa/build_check.py`（CPU 上 ~1 分钟）：构建完整训练环境，
   断言每条 `feet_ground_contact` CONTACT 传感器的 refname 都解析到了沙发体；
   `full` 阶段训练前自动跑（`SKIP_BUILD_CHECK=1` 可跳过）——以后任何环境配置错误
   都会在几秒内死掉，而不是等到 GPU 排队之后。

**教训**：smoke 阶段本来就能抓住这类错误（它会构建完整环境），但训练命令直接
跑 `full` 时没人挡。现在 `full` 自带构建闸门，这类问题无法再漏进训练。

## 10. 事故复盘：2026-11-13 verify 脚本本身崩了（mjlab 1.3.0 API 不匹配）

**现象**：训练跑通、checkpoint 有了，但 `verify` 在打印表头后立刻抛异常：

```
TypeError: CommandManager.reset() missing 1 required positional argument: 'env_ids'
```
脚本把它当成"策略不会走"报 FAIL —— 其实是**评估脚本自己的 bug**，与策略无关。

**四个 API 坑**（都在 mjlab 1.3.0 源码里核对过）：

| 项 | 脚本原来写的 | mjlab 1.3.0 实际 |
|---|---|---|
| 指令重置 | `command_manager.reset()` | `reset(env_ids)` **必须**传 tensor（内部 `assert isinstance(env_ids, torch.Tensor)`）。正确做法是根本不手动调——`env.reset()` 会自动 resample |
| 环境交互 | `obs, rew, term, trunc, extras = wrapped.step(a)` | `RslRlVecEnvWrapper.step()` 返回 **4 元组** `(obs, rew, dones, extras)`；`reset()` 返回 `(TensorDict obs, extras)`。且 raw env 的 obs 是 dict，rsl_rl 策略只吃 wrapper 的 TensorDict |
| 根状态 | `robot.root_state_w[0]`（`.pt` 里没有这个属性） | `robot.data.root_link_pose_w`（pos+quat）、`root_link_lin_vel_b`、`root_link_ang_vel_b`（机体坐标系，直接就是指令所在的坐标系） |
| 指令可变 | 只设了 `ranges` 就以为指令固定了 | `_resample_command` 还会按概率改指令：`rel_forward_envs` 把 `lin_vel_x` 抬到 ≥0.3 并清零其余、`rel_world_envs` 旋转指令、`rel_standing_envs` 清零、`init_velocity_prob` 改写机器人初始速度 |

最后一条最阴：单 env 扫描时这些开关等于**随机**——抽到零指令会把会走的策略判成站桩，
抽到 forward 改写会把站桩判成"在动"。所以脚本现在：

1. 把上述四个概率全部置 0（在创建 env **之前**改 cfg）；
2. 每段开始时只改 `ranges`，靠 `wrapped.reset()` 重新采样；
3. **断言**实际指令等于期望值（`torch.allclose`），不一致就打印 `BAD COMMAND` 并计入失败，
   绝不拿一条不是自己想测的指令去下结论；
4. 速度直接用 `root_link_lin_vel_b[0,0]` / `root_link_ang_vel_b[0,2]`，不再手写四元数旋转。

**第二层坑**（第一次修完才浮出来，三次迭代才对）：

| 现象 | 原因 | 正确做法 |
|---|---|---|
| `ValueError: ranges.heading is set but heading_command=False` | 为了"关掉"heading 而设 `heading_command=False`，但 cfg 里 `ranges.heading` 非 None；命令项构造函数**双向**校验这两者必须一致 | 别动 `heading_command`；只把 `rel_heading_envs=0`（`is_heading_env` 全 False，heading 控制器就不会改写 `ang_vel_z`） |
| 0.20 m/s 的用例可能变成"原地转" | microduck 自有 `VelocityCommandCommandOnlyCfg` 多了 `rel_turn_in_place_envs`（沙发任务设 0.10）：按概率强制 `lin=0`、`|wz|≥0.4·max` | `twist.rel_turn_in_place_envs = 0.0`（用 `hasattr` 保护，基类没有这个字段） |
| 置 0 之后在扫描途中又被改回去 | 课程是**运行时的写入者**：`standing_envs` 课程会把 `rel_standing_envs` 从 0.02 一路抬到 0.05，直接覆盖我们的 0 | `env_cfg.curriculum = {}`（评估不需要任何课程） |

**结论**：`verify FAIL` 只有一种可信前提——表里每行都打印了实测速度。表头之后立刻
抛异常的情况，先怀疑脚本，再怀疑策略。修复后重跑：

```bash
bash scripts/sofa/modelscope_run.sh verify
```

如果输出里出现 `BAD COMMAND`，说明指令又被某个 resample 规则改写了，去检查
`UniformVelocityCommandCfg`/`VelocityCommandCommandOnlyCfg` 里还有哪个概率字段没归零。

## 11. 事故复盘：2026-11-13 verify 还是判「站桩」（指令时长 vs 坐垫尺寸）

verify 终于能跑了，结果是：

```
 cmd vx  cmd wyaw   ach vx  ach yaw  ratio  trunk z  verdict
0.20      0.00   -0.002   -0.028   0.00    0.627  NOT TRACKING
-0.20     0.00   -0.002   -0.009   0.06    0.634  NOT TRACKING
```

`trunk z` 0.627/0.634 ≈ spawn 高度（SEAT_TOP_Z 0.518 + 0.115~0.125），也就是它
**站在坐垫上原地不动**，和 2026-11-12 那次一模一样。

### 先分清两件事

1. **这个 checkpoint 是哪套配置训出来的？** 前一次是 `pre-fix` 的站桩模型，
   用它跑 verify 当然还是站桩——这不构成对修复的任何信息。
2. **如果确实是新配置训的，为什么还是站？** 那才需要改任务。

`scripts/sofa/ckpt_report.py`（新增）直接把答案从训练产物里读出来，不再靠回忆：
训练器在 run 目录写了 `params/env.yaml`（rank 0 启动时 dump 的**真实 cfg**）和
tensorboard 事件文件。它打印四个反站桩开关的当时取值 + 训练指标首末值，并给出

```
VERDICT: PRE-FIX config (4 knob(s) differ).
         This checkpoint predates the fix; a standing policy here is EXPECTED.
```

退出码 **3 = pre-fix**。`verify` 阶段现在**先**跑它，拿到 3 就直接中止，不再对着
旧 checkpoint 争论：

```bash
bash scripts/sofa/modelscope_run.sh report          # 最新 checkpoint 的溯源 + 指标
bash scripts/sofa/modelscope_run.sh report --all    # 所有 run，一行一个
```

run 目录名现在带 `_v2_bounded`，`ls logs/rsl_rl/velocity_sofa/` 就能一眼看出新旧。

### 真正的问题：指令时长根本装不进坐垫

这条和奖励权重无关，是**几何约束**，之前两次都漏了：

| 量 | 值 | 来源 |
|---|---|---|
| 可用坐垫长度（行走轴） | y ∈ [-0.78, 0.78]，共 1.56 m | `sofa_constants.py` |
| 旧的 spawn y 范围 | [-0.45, 0.45] | 同上 |
| 从最差 spawn 到扶手 | **0.33 m** | 0.78 − 0.45 |
| 指令保持时长 | **3–8 s**（mjlab stock 默认，从未被覆盖） | `velocity_env_cfg.py:180` |
| 0.20 m/s × 3 s | **0.60 m** ≫ 0.33 m | — |

也就是说：**一个完美跟踪指令的鸭子，在指令还没变之前就撞上扶手了**。而
`fell_off_sofa` 是终止项（`time_out=False`），掉下去就等于提前结束、未来奖励清零。
反过来站着不动能活满 20 s，还白拿 `pose`(1.0) + `upright`(2.0)。
把 track std 压到 0.12 也救不回来——最优解真的就是「别动」。

第二个隐藏 bug：`rel_forward_envs = 0.2` 也从 stock 继承了。它的实现是
`abs(lin_vel_x).clamp(min=0.3)`，于是 20% 的 env 被命令 **0.30 m/s**——既超出任务
声明的 ±0.20，又正好是最快冲出坐垫的指令。

### 修复

| 问题 | 修复 |
|---|---|
| 指令保持 3–8 s 装不进坐垫 | `command.resampling_time_range = (1.2, 2.4)`：0.20 m/s × 2.4 s = **0.48 m** < 0.58 m 余量 |
| spawn 铺得太开 | `SPAWN_Y = (-0.20, 0.20)`、`SPAWN_X = (-0.18, -0.08)`（居中，最差余量 0.58 m） |
| 20% env 被命令 0.30 m/s | `command.rel_forward_envs = 0.0` |
| 验证窗口比坐垫还长 | verify 默认 `--steps 100 --warmup 25`（2.5 s → 0.50 m），并按 `CUSHION_ROOM_M = 0.58` 自动截断并打印说明 |
| 分不清 checkpoint 新旧 | `ckpt_report.py` + verify 前置溯源闸门（exit 3） |

判据：**每条指令段都必须在坐垫上做得完**。只要「跟踪 = 必死」这个等式还在，
任何奖励调参都会被它压过去。

### 重训

```bash
bash scripts/sofa/modelscope_run.sh full     # 构建闸门 + 4096 envs × 4000 iters
bash scripts/sofa/modelscope_run.sh verify   # 先溯源，再扫 ±0.20 m/s
bash scripts/sofa/modelscope_run.sh export
bash scripts/sofa/modelscope_run.sh upload
```

训练中想早点知道会不会又站桩，随时跑 `report`：`Episode_Reward/air_time` 一直是
0.000、`Episode_Termination/fell_off_sofa` 一直是 0，就是又没走起来——不必等 4000 轮。


## 12. 事故复盘：2026-11-13 构建闸门崩在「下不到 networkx」

```
== build gate: constructing the training env on CPU (SKIP_BUILD_CHECK=1 to skip)
⠙ nvidia-cufft-cu12==11.4.1.4
error: Failed to fetch: https://mirrors.aliyun.com/pypi/packages/9e/c9/.../networkx-3.6.1-py3-none-any.whl
cause: HTTP status client error (403 Forbidden)
!! build gate failed -- fix the task spec before training
```

两个问题叠在一起。

### 一、闸门根本不该联网

注意报错 URL 的**主机**：`mirrors.aliyun.com/pypi/packages/...`，这是**索引推导出来**的
地址。如果走的是 lock（哪怕没本地化），主机应该是 `files.pythonhosted.org`；如果走的是
wheelhouse，应该是 `file://`。所以可以断定：**uv 做了重新解析，而不是按 lock 取包**。

原因是 `uv run` 本身——它在 exec 前会校验 lock 并做一次隐式 `uv sync`。解压 bundle 覆盖
了 `uv.lock`（本地化版本只存在于实例上，原件在 `.wheelhouse/uv.lock.remote`），这个隐式
sync 就重新解析、重新下载了。对一个**只是要跑个脚本**的闸门来说，这是完全没必要的风险。

> 修法不是"赌 uv 不会重新解析"，而是**让它没有机会**：所有阶段直接跑
> `$REPO_ROOT/.venv/bin/python`（训练跑 `.venv/bin/train`），不经过 uv，就没有 lock
> 校验、没有解析、没有网络。

### 二、阿里云镜像会对个别文件返回 403

`networkx-3.6.1` 在阿里云上是 403（它没有镜像到这个文件），而 uv 没有"逐文件换源"的
能力：一个 403 直接终止整个 sync。curl 有回退（prefetch 一直如此），uv 没有。

### 三、报错把人往错误方向带

403 是**依赖可用性**问题，却被 `die` 成「fix the task spec before training」——上一轮
我们就是这样跑去翻沙发 XML 的。现在闸门失败会明确区分：

```
build gate failed -- see the traceback above
  (a MuJoCo/sensor error means the task spec;
   an ImportError or a fetch error means the environment: run `... check`)
```

### 修复

| 问题 | 修复 |
|---|---|
| `uv run` 在闸门里触发隐式 sync | 新增 `run_py` / `run_train`，直接用 `.venv/bin/python` / `.venv/bin/train`；`ensure_env` **先**检查 venv 是否完整，完整就完全不碰 uv（连装都不需要） |
| 阿里云 403、uv 无逐文件回退 | `sync_deps` 按 **阿里云 → 清华 → 腾讯云** 轮换索引重试；`prefetch` / `fetch_wheels` 逐文件按 **阿里云 → 清华 → pythonhosted** 链式回退 |
| 镜像个别文件缺失 | `wheel_urls()` 为每个 wheel 生成候选链，任一源成功即停，`curl -C -` 保证可续传 |
| 截断的 wheel 能骗过"文件存在"检查 | 首次 sync 失败后 `repair_corrupt_wheels()` 按 lock 的 sha256 逐个校验，删掉坏的重下（截断文件只会在 uv 的 sha256 检查里失败，且报错不点文件名） |
| 旧代码只要 `.wheelhouse` 里有任意 `.whl` 就走离线 | `wheelhouse_ok()` 要求清单里**每个** wheel 都在；不全就先 repair，而不是直接 `uv sync --frozen` 撞 "Network connectivity is disabled" |

### 现在怎么跑

venv 已经完整的话，`check` / `smoke` / `full` / `verify` / `report` / `export` 全都不再联网，
第一行会打印：

```
== venv ready: Python 3.12.x -- stages run it directly (no uv, no network)
```

只有这两种情况才会下载：venv 不存在/不完整，或者你显式跑 `prefetch`。

```bash
tar -xzf microduck_sofa_bundle.tar.gz -C /mnt/workspace
cd /mnt/workspace/microduck_rl
bash scripts/sofa/modelscope_run.sh full      # 闸门 -> 训练
```

如果 venv 真的坏了，先 `prefetch`（可续传、多镜像），再 `check`：整条链路会装成全离线。

## 13. 事故复盘：v2_bounded 仍然站桩（奖励数学 + 软垫可行性实验）

v2 训完（`2026-11-13_21-35-25_v2_bounded`），verify 判 NOT TRACKING（ach vx ≈ 0，
trunk z ≈ spawn 高度）。`report` 的训练指标反而"看起来不错"：

```
Episode_Reward/track_linear_velocity   0.0045 -> 0.8240     <- 看起来在学跟踪
Episode_Reward/air_time                0.0000 -> 0.0000     <- 步态激励全程为 0
Metrics/air_time_mean                  0.0419 -> 0.0000
Episode_Termination/fell_off_sofa      0.0000 -> 0.0000
Train/mean_episode_length              23.76  -> 984.87     <- 活满 20 s，从不摔
Curriculum/action_rate_weight          -0.1   -> -1.0       <- 平滑惩罚被课程拉满
```

### 一、用奖励数学判定"训练里也在站"

从 `reward_manager.py` 源码确认：`Episode_Reward/<term> = weight × 高斯项的时间平均`
（episode 求和 × dt ÷ episode 时长）。给"站着不动"算理论值：

- 指令 |vx| 均匀分布在 ±0.2 → 高斯项对指令分布求平均 = 0.42（**小指令很常见，而
  小指令下站着=完美跟踪**）；再算上 10% turn-in-place（lin=0，直接拿满分）和 5%
  standing env → 站立理论值 ≈ 2.0 × 0.51 = **1.02**，叠加推搡晃动后 ≈ 0.82
- **日志 0.824 与站立预测几乎完全一致**；air_time ≡ 0、从不摔、episode 满长，全部吻合

结论：训练里鸭子也是站着的。verify 没有测量错误。

### 二、软垫上到底能不能走？（本地 A/B 实验）

用 duck-sim 的 `infer_policy`（真实 BAM 执行器 + 位姿 5Hz 遥测）把**已知策略**放到
我们的沙发场景上跑（指令 0.20 m/s）：

| 策略 / 场景 | 结果 |
|---|---|
| alpha_walking（官方平地 gait）/ 软垫 | 前冲脉冲 0.05–0.11 m/s，横向漂移，**21 s 后从侧边掉下沙发** |
| alpha_walking / 硬垫（solref 0.02 A/B 场景） | 同样 ~0.05 m/s，横漂，14 s 掉下 |
| alpha_walking / 平地 | 能走但本身只会绕大圈（比率 ~0.3–0.4） |
| 社区 sofa_walk / 软垫 | **完全不动**（38 s 位移 ≈ 0，站得很稳） |

结论：① 软垫上**物理上可以走**（官方 gait 有明确前冲），不是"垫子走不了"；
② 但官方 gait 自己的 tracking 比率也只有 0.25–0.55 且必然侧漂出界——
**verify 的 0.55 通过线定得比物理允许的还高**，已下调到 0.35（依然能以极大裕量抓住站桩）；
③ 训练收敛到站立是奖励结构问题，不是场景问题。

### 三、v3_gait 配方（本次修复）

| 问题 | 修复 |
|---|---|
| 指令均匀分布含大量近零指令，站着≈满分跟踪 | 新增 `*_abs_range` 采样（`VelocityCommandCommandOnlyCfg`）：`|vx| ~ U(0.08, 0.20)`、`|wz| ~ U(0.15, 0.80)`、随机符号——**任何 env 都被要求动** |
| 5% standing env + 课程会把它们加回来 | `rel_standing_envs = 0` 且 `curriculum.pop("standing_envs")`（课程是运行时写入者，不删会被改回去） |
| turn-in-place 桶的 lin=0 指令恰是"站桩形状"的指令 | `rel_turn_in_place_envs = 0`（转向由 |wz|≥0.15 覆盖） |
| `air_time` threshold_min=0.125 s（50Hz 下要腾空 6 步）在软垫碎步上永远达不到 → 3.0 权重的步态激励全程为 0 | `threshold_min = 0.04`（2 个控制步，任何真实抬脚即计奖） |
| action_rate 课程拉到 −1.0，惩罚步态起振所需的振荡 | 课程封顶 −0.3（平滑压力保留，不再压制步态） |
| 横向漂移把不转向的步态带出坐垫边 | `LIN_VEL_Y` ±0.10 → ±0.06 |

### 四、下次训练

```bash
bash scripts/sofa/modelscope_run.sh full     # run 目录名 v3_gait
bash scripts/sofa/modelscope_run.sh verify   # 先溯源（PRE-V3 直接中止），再扫指令
bash scripts/sofa/modelscope_run.sh export
bash scripts/sofa/modelscope_run.sh upload
```

判读要点（`report`）：
- `Episode_Reward/air_time` 首次 > 0 并增长 → 步态激励活了，这是 v3 生效的第一信号；
- `Metrics/air_time_mean` 稳定 > 0 → 真的在迈步；
- 若仍站桩：看 `Episode_Termination/fell_off_sofa`——若始终为 0 且 episode 满长，
  说明探索从未离开站立盆地，下一步考虑奖励比_gain_（提高 air_time 权重）或把
  track std 进一步收紧；若 fell_off 频发，说明在走但出界，需要回到几何约束
  （指令更短 / 出生更居中）。

---

## 14. 绕圈任务（c1_circle，2026-11-14 晚）

### 一、为什么不再调直行

v3（abs_range 指令 + air_time 0.04 + 课程封顶）在训练中仍未产生步态。复盘下来，
三份直行配方改的都是**奖励数值**，而真正的约束是**几何**：

```
坐垫可用区 1.56 m (y) × 0.62 m (x)
0.20 m/s 的直行指令 → 从居中出生点出发，2.9 s 后撞到扶手
episode 20 s，命令保持 1.2–2.4 s 后重采样 → 每次都朝一个新方向再撞一次
⇒ 「跟得准」= 更快出界，「不出界」= 不动。最优解真的是站着。
```

只要任务是直行，这个矛盾就无法用权重解决——把 weight 调到天上，也改变不了
「跟踪 = 死亡」的因果。所以换任务，而不是继续调参。

### 二、为什么是绕圈（而不是跳跃）

| 任务 | 软垫可行性 | 几何 | 抗站桩 | 改造量 |
|---|---|---|---|---|
| 直行 | 勉强 | 贴边，2.4 s 上限 | 差（v2/v3 都站） | 0（已在跑） |
| **绕圈 r≤0.20** | **可行**（不需长距离推进） | **正好用满坐垫中心** | **强** | 小 |
| 原地跳跃 | 最差（需瞬时大冲量，被泡沫吸收） | 原地不占地方 | 强 | 重写任务 + 相位奖励 |

跳跃把成功率押在「垫子刚度」这一个不可测参数上（solref 0.02 能走、0.06 侧漂，
A/B 已证一阶敏感），sim2real 风险最大，且 mjlab 无现成任务。绕圈相反：慢速、
小半径、每条指令都回到起点。

### 三、几何

> ⚠️ **本节记录的是 c1 当时实际用的数字，全部是错的**（圆心偏了 0.365 m、
> 半径上限大了 0.06 m、坐垫尺寸来自包围盒而非可用面）。2026-11-14 的几何事故
> 复盘见 **§16**，那里的数字是射线实测的，现在是唯一权威。

c1 当时的假设（错）：

```
圆心  = 坐垫可用区中心 (x=-0.14, y=0)      ← 实际嵌在靠背实体里
半径  r ~ U(0.14, 0.20) m
角速度 ω ~ ±U(0.35, 0.55) rad/s（每 episode 随机符号）
切向速度 v = ω·r ∈ [0.049, 0.110] m/s
一圈 2π/ω ≈ 11–18 s（20 s episode ≈ 一圈）

圆最右 x = -0.14+0.20 = +0.06  vs 坐垫边 +0.17  → 余 0.11 m
圆最左 x = -0.34               vs 坐垫边 -0.45  → 余 0.11 m
y 方向 ±0.20 vs ±0.78                          → 余 0.58 m
```

c2_geo 按实测重算后的几何（现在是代码里的值）：

```
圆心  = 可用口袋内接矩形中心 (x=+0.235, y=-0.010)
半径  r ~ U(0.08, 0.12) m        （几何硬上限 0.137 = 0.53/2 − 0.128）
角速度 ω ~ ±U(0.50, 0.90) rad/s
切向速度 v = ω·r ∈ [0.040, 0.108] m/s
一圈 2π/ω ≈ 7–13 s（12 s episode 至少一整圈）
```

### 四、核心机制：用「扫过的圆心角」打分，而不是 yaw rate

`circle_progress` 测的是**机器人位置绕圆心的极角变化**（几何量）：

```
Δθ = wrap(θ_t − θ_{t-1}),  θ = atan2(y−cy, x−cx)
ratio = Δθ / (|ω|·dt) · sign(ω)        # 1.0 = 完美
reward = exp(−((ratio−1)/0.5)²)
```

这一个选择同时消灭两种失败模式：

| 行为 | gyro ω_z | 扫过角 Δθ | circle_progress |
|---|---|---|---|
| 站着不动 | 0 | 0 | 0.018 |
| **原地打转** | 满值 | **0** | **0.018** |
| 沿圆走 | ω | ω·dt | 1.0 |
| 反向走 | −ω | <0 | 0.018 |

因此 **`track_angular_velocity` 被关掉（weight 0.0）**——它恰好奖励「原地打转」，
是唯一一处刻意偏离原配方的奖励项。`track_linear_velocity` 降到 1.0 保留，因为
它无法在不动的情况下被满足。

`circle_radius`（weight 1.0, std 0.04）负责「别走到另一个半径的圆上」：只看
角速度的话，任何半径的圆都能满分。注意它对站桩是**不敏感**的（站着的鸭子正好
在圆上），所以绝不能单独使用，权重也压在 progress 之下。

### 五、代码位置

| 文件 | 内容 |
|---|---|
| `tasks/mdp.py`（文件末尾） | `CircleTrackState` / `reset_on_circle` / `CircleVelocityCommand(Cfg)` / `circle_progress_reward` / `circle_radius_reward` |
| `tasks/microduck_circle_sofa_env_cfg.py` | 任务配方（几何常量、奖励权重、课程、run 名 `c1_circle`） |
| `tasks/__init__.py` | 注册 `Mjlab-Circle-Sofa-MicroDuck` |
| `scripts/sofa/build_check.py` | CPU 门禁：**新增 7 项圆几何断言**（在圆上 / 切向朝向 / 高度 / 指令=ωr / 范围内 / 双向 / 不出坐垫） |
| `scripts/sofa/verify_circle.py` | 训练后门禁：实测扫过角、半径误差、圈数 |
| `scripts/sofa/ckpt_report.py` | `--experiment` + CIRCLE 专属溯源检查 |
| `scripts/sofa/modelscope_run.sh` | `TASK` 可切换；verify/probe/report 自动按任务分流 |

obs 契约不变：仍是 61 维、cmd 通道仍是 3 维 `(v, 0, ω)`。部署时喂恒定的
`(ω·r, 0, ω)` 即可，ONNX 与 robotd 无需改动。

### 六、跑法

```bash
bash scripts/sofa/modelscope_run.sh check    # build gate 会先验证 7 项圆几何
bash scripts/sofa/modelscope_run.sh full     # run 目录名 c1_circle
bash scripts/sofa/modelscope_run.sh verify   # 自动选 verify_circle.py
bash scripts/sofa/modelscope_run.sh export
```

切回直行任务：`TASK=Mjlab-Velocity-Sofa-MicroDuck bash scripts/sofa/modelscope_run.sh full`

### 七、判读

训练中（`report`）：

| 曲线 | 站桩/打转 | 正常绕圈 |
|---|---|---|
| `Episode_Reward/circle_progress` | ≈0.02 | **≈0.6–1.0** |
| `Episode_Reward/circle_radius` | ≈1.0 | ≈0.8–1.0 |
| `Episode_Reward/air_time` | 0.000 | **首次 > 0 即为步态活了** |
| `Episode_Reward/track_angular_velocity` | 恒 0（weight 0，不计算） | 同左 |

验证（`verify_circle.py`）：`w ratio ≥ 0.40`、`r err ≤ 0.05 m`、未掉落。
站桩/打转的 ratio ≈ 0，与 0.40 有极大裕量。

### 八、已知风险（未在真机验证）

1. **yaw 漂移累积**：软垫上 yaw 漂移最严重，而绕圈对 yaw 误差是累积的。缓解手段
   是 radius 奖励（半径发散会被打分），但若训练后 `r err` 长期 >0.05 m，说明
   yaw 漂移压不住，下一步应把 `circle_progress` 的 sigma 从 0.5 收紧到 0.35。
2. **开环指令**：策略看不到自己在圆上的相位，只看到恒定 `(v,0,ω)`。这保持了
   61 维契约，代价是无法主动纠偏；纠偏完全依赖策略对「身体状态」的闭环，而不是
   对「位置误差」的闭环。若 ratio 上不去而半径也稳，说明它其实在走一个偏心的圆。
3. **双向泛化**：CW/CCW 各占一半，但机器人的腿不对称（bearing_roll 在右髋），
   可能出现只有一个方向学得好的情况。`verify_circle.py` 会分 env 打印 ω 的符号，
   注意对比。

### 九、2026-11-14 首次 full：build gate 的 2π 假阳性

首次 `full` 停在 build gate：

```
ok : spawn |dist - r| max = 0.000000 m (r in [0.142, 0.194])
FAIL: tangent heading max err = 6.283185 rad      <-- 恰好 2π
ok : spawn z max err = 0.0048 m      ok : command (v,0,w) err = (0,0,0)
ok : r/w 在范围内   ok : 双向 13 ccw / 11 cw   ok : 不出坐垫   ok : 24 次半径不同
```

**其余 7 项全过、误差 0，说明出生位姿是对的**，只有切向那一项报 2π。
2π 是恒等旋转，不可能是真实的朝向错误，是角度分支切割（branch cut）问题：

```python
theta0 ~ U(-π, π)              # 圆上的出生极角
yaw   = theta0 ± π/2           # 可以落在 [-3π/2, 3π/2] 之外
quat  = (cos(yaw/2), 0, 0, sin(yaw/2))    # 四元数只定义到 mod 2π，位姿正确
yaw_sim = atan2(...)           # 从四元数读回来，被折进 [-π, π]
delta = yaw_sim - theta0       # ← 未折回，|theta0| > π/2 时差出整整 2π
```

`|theta0| > π/2` 的概率是 50%（实测旧公式 25.2% 的样本直接判 FAIL，
最大误差 6.283185 rad，与日志一字不差）。所以这是**判据的 bug，不是环境的 bug**。

两处都改了，各管一件事：

| 位置 | 改动 | 作用 |
|---|---|---|
| `mdp.py` `reset_on_circle` / `_sample_unset` | `theta0`、`yaw` 用 `wrap_to_pi` 归一化后再存 | 存储的出生角与 `atan2` 读出的角同一规范，任何比较都不再跨越分支 |
| `build_check.py` 切向断言 | 改用**向量点积**（机体 x 轴 · 圆切线），角度差只作打印诊断 | 点积没有分支切割，即使将来又有人存了未归一化的角也不会假阳性 |

`wrap_to_pi(π) = π`（见 `mjlab/utils/lab_api/math.py:104`），所以 ±π 两个边界都稳定。

**教训**：`spawn |dist-r| = 0.000000` 与 `yaw` 相关的那一项必须**同源**比较——
位置量是欧氏的，角度量是模 2π 的，混用就会出现「几何完全正确但判据差 360°」。

### 十、2026-11-14 c1_circle 跑完后 report 判 PRE-CIRCLE（第二次判据 bug）

训练正常结束，report 却说：

```
air_time threshold_min     0.04   ok
circle_progress weight      2.0   ok
track_angular weight        0.0   ok
orbit radius max       0.14, 0.2   ok
reset_base func                ?   PRE-CIRCLE (want reset_on_circle)   <-- 唯一的错
contact sensor secondary     ? @ -
VERDICT: PRE-CIRCLE config (1 knob(s) differ).
```

四项旋钮全部读到且匹配，唯独 `func` 和传感器 `pattern/entity` 读不到。两个原因都在
`ckpt_report.py` 自己的解析器里，与环境无关：

| 现象 | 根因 |
|---|---|
| `reset_base func = ?` | PyYAML 写函数是 `func: !!python/name:…reset_on_circle ''`——**名字在 tag 里，payload 是空串**。解析器把所有 `!!` 开头的值当作 tag 丢掉，于是每个 callable 都变成 `None` |
| `contact secondary = ? @ -` | `ContactMatch` 是嵌套对象，深度取决于 `yaml.dump` 的 tag 选择，原来硬编码只找一层 |

同时修掉一类更危险的行为：**读不到 ≠ 不匹配**。这个 gate 已经两次因为自己解析不出而
否掉一个正常的 run（上一节 2π 也是同类）。现在三种结局分开：

```
读到 reset_on_circle  -> ok
读到别的函数名        -> 真的 PRE-CIRCLE（计入 stale）
什么都读不到          -> "? (unreadable, not counted)"，不计入 stale
```

再加一条**正向证据**：`circle_progress` / `reset_on_circle` / `CircleVelocityCommand`
这些标识符只可能来自圆周任务的 cfg，只要在 run 自己的 env.yaml 里出现，就是「训练
用的确实是这个任务」的硬证据，不依赖任何嵌套解析。四个 fixture 自测：正确 run → 0、
stock reset → 3、读不到 → 0（标 ?）、直行 run（track_ang=1.0）→ 3。

顺带补了两个盲点：

- `Episode_Reward/circle_progress` 之前**不在 report 的 tag 列表里**——report 是文档
  让用户在 gate 失败时运行的命令，却看不到这个任务唯一的决定性问题。现在会打印，
  并直接给一行结论（≥0.6 在绕圈 / 0.2–0.6 走不够 / <0.2 站着或原地打转）。
- `--all` 对圆周 run 之前会标成 `V3-CURRENT`（用它没有的直行旋钮打分），现在标 `CIRCLE`。

### 十一、c1_circle 的结果：它在原地踏步（2026-11-14 晚）

4000 iter 跑完，`verify_circle.py` 8 个 env 全部 NO ORBIT：

```
env  r cmd   w cmd   w ach  ratio   r err   laps      z  verdict
0    0.155  -0.462   0.002   0.00  0.0498  +0.00  0.632  NO ORBIT
1    0.192  -0.362   0.019  -0.05  0.0067  -0.03  0.625  NO ORBIT
...
4    0.149  +0.507   0.379   0.75  0.0430  +0.60  0.628  DIED
6    0.162  -0.511   0.078  -0.15  0.0283  -0.12  0.630  DIED
orbit pass rate: 0/8
```

读法：`w ach` 是实测扫过角速度（rad/s），指令是 0.36–0.51。**7 个 env 只走了
0.002–0.019 rad/s**，也就是 10 s 里几乎没动；`r err` 又极小，因为站着不动的鸭子
天生就在圆上。这不是测量 bug：训练里 `air_time` 0.37/step 说明它抬脚抬得很好，
`track_linear` 只有 0.34（站着不动的理论值是 0.64，因为它连瞬时速度误差都比站着
大），两者合起来只有一种解释——**原地踏步，不是走路**。

两个 `DIED` 的 env 恰好是唯一走起来的（laps +0.60 / −0.12），而且 `r err` 明显偏大
（0.043 / 0.028）：它能走，但走出的圆半径偏了，撞到坐垫边缘。训练里
`fell_off_sofa` 全程 0.0000，说明训练**从没见过边缘**，所以「走出去会掉」这件事
在价值函数里根本不存在。

#### 为什么奖励没能把它推出去

把训练末期的每步奖励按实测值摊开（`Episode_Reward/*` 是**加权后的每步均值**，
`Train/mean_reward ÷ 20 s = 3.31` 正好等于各项之和，可交叉验证）：

| 项 | 站立/踏步能拿多少 | 绕圈能拿多少 |
|---|---|---|
| upright 1.0 | ≈1.0 | ≈1.0 |
| pose 1.0 | ≈1.0 | ≈0.8 |
| **circle_radius 1.0** | **≈1.0（出生即在圆上，白送）** | ≈1.0 |
| **air_time 3.0** | **0.37（踏步就能满分）** | 0.37 |
| track_linear 1.0 | 0.34 | 0.6+ |
| **circle_progress 2.0（高斯 σ=0.5）** | **0.036** | 2.0 |
| action_rate | −0.63 | −0.63 |

站立侧稳拿 ≈2.4/步，而通往绕圈的唯一收益是 progress 的 2.0——但高斯的斜率在
ratio=0 处只有 `2·exp(−4)/σ = 0.147`，乘权重 2.0 也只有 **0.29/单位 ratio**。
换句话说，策略要放弃 2.4 的稳定收入，去换一条起步斜率 0.29 的坡。它选择了不走。

#### c2 配方（改动都对应上面某一行）

| 改动 | 对应证据 |
|---|---|
| `circle_progress` 改**线性** `clip(ratio,0,1.5)`，权重 2.5 | 高斯在站立点的斜率只有 0.29，比它要放弃的收入小 8 倍；线性项从 0 起就有恒定斜率 |
| `circle_radius` 权重 1.0→0.6，并乘 `clip(ratio/0.25,0,1)` 门控 | 站立白拿 1.0，是站着盆地最大的一笔收入 |
| `air_time` 权重 3.0→1.5 | 踏步就能拿满，它奖励的是"抬脚"而不是"走路" |
| 半径 0.14–0.20 → **0.10–0.15**，ω 0.35–0.55 → **0.45–0.70** | 两个 DIED 的 env 都是半径偏 3–4 cm 就撞边；r≤0.15 时同样的绝对误差仍有 >0.15 m 余量。切向速度仍锁在 0.045–0.105 m/s |
| episode 20 s → **12 s**（约一圈） | c1 的走法只能撑 0.6 圈；而且策略拿不到自己在圆上的位置（obs 契约不变），越长的 horizon 只是要求它"忍"更久的累积漂移。任务要求是一圈，那就把一圈当一集 |

`circle_arc_ratio()` 同时重写成**按 `common_step_counter` 缓存**：原来 `theta_prev`
的推进藏在 `circle_progress_reward` 里，谁先被 reward manager 调用谁就吃掉这一步的
弧长，后面的 term 一律读到 0。加半径门控后有两个消费方，这个隐式契约必然踩雷，
而且日志上看不出来。

#### build gate 新增的第 6 条断言

CPU 上 reset 一次（鸭子正好在圆上、还没动），直接调用奖励函数：

```
arc_rate  = 0.0000   （站立必须 ~0）
gated radius = 0.0000（站立必须 ~0，否则门控没接上）
ungated radius = 1.000（它确实在圆上）
```

这条断言 30 秒内能发现「gate_ratio 传错 / 传了高斯函数」，而不是花一整晚 GPU 之后
再靠 verify 反推。

#### 判读（c2）

| 曲线 | 站立/踏步 | 正常绕圈 |
|---|---|---|
| `circle_progress` | 0.00 | 0.6–1.0（线性，1.0 = 指令角速度） |
| `circle_radius` | **0.00（门控，正常）** | 0.5–0.6 |
| `air_time` | 0.1–0.2 | 0.2–0.4 |
| `fell_off_sofa` | 0 | 0（若 >0 说明半径压不住，下一步收窄 r） |

`report` 现在会按 env.yaml 里记录的**奖励函数名**自动识别 c1/c2 两套配方，各自用
各自的旋钮表核对（`CIRCLE_RECIPES`），所以老 c1 的 run 仍然可读、不会误报。

---

## 15. 上传阶段：去掉 uv 依赖 + 强制只传 onnx（2026-11-13 深夜）

### 一、报错

```bash
export FORCE_UPLOAD=1
bash scripts/sofa/modelscope_run.sh upload
# == uploading logs/rsl_rl/velocity_circle/2026-11-14_11-12-49_c1_circle + policy.onnx ...
# scripts/sofa/modelscope_run.sh: 行 606: uv: 未找到命令
```

`FORCE_UPLOAD=1` **本身是生效的**（闸门已跳过，已经到了 "uploading ..." 那一行），
死在它后面那三行：`.msvenv` 是用 `uv venv` + `uv pip install` 建的。

### 二、为什么这台机器上必然没有 uv

这是第 1 节那个改动的副作用，不是偶发：

```
ensure_env()
  ├── venv_ready ?  → 是 → 直接返回，uv 一次都不调用（连装都不装）
  └── 否            → ensure_uv / sync_deps
```

训练环境齐备的机器上，`uv` 从来没被安装过——**这是设计目标**（"gate 不许联网"）。
所以 upload 阶段只要碰 `uv`，就只会在"训练已经跑完"的机器上炸，而这恰好是唯一会
执行 upload 的机器。`check` / `full` 都过、只有 `upload` 挂，就是这个原因。

### 三、改法：标准库 venv + pip，镜像链复用

```bash
MSVENV=$REPO_ROOT/.msvenv
ensure_ms_sdk() {
  ms_importable $VENV_PY        && MS_RUN=($VENV_PY)     # 训练环境里已有就直接用
  msvenv_ready                  && MS_RUN=($MSVENV_PY)   # 已建好 → 复用
  rm -rf $MSVENV                                          # 半成品直接重建，不修
  $VENV_PY -m venv $MSVENV                                # ← 只依赖 stdlib（ensurepip 自带 pip）
  for idx in "${PYPI_INDEXES[@]}"; do
    $MSVENV_PY -m pip install -q --index-url $idx modelscope && MS_RUN=($MSVENV_PY) && return
  done
  # 最后兜底：--target .mslib + PYTHONPATH，绝不动被 lock 钉死的训练环境
}
```

三个刻意的选择：

| 选择 | 原因 |
|---|---|
| `$VENV_PY -m venv` 而不是 `uv venv` | uv 建的 venv 默认**不带 pip**，而 `-m venv` 走 ensurepip，pip 一定有 |
| 半成品 `.msvenv` 直接 `rm -rf` 重建 | 镜像重建后 `.msvenv/bin/python` 可能是指向已消失解释器的软链；修比删贵 |
| 兜底用 `--target .mslib` + `PYTHONPATH` | 训练环境被 uv.lock 钉死，modelscope 会带自己的 requests/urllib3，绝不能装进去 |

`msvenv_ready` 单独检查 `-m pip --version`：只判 `import modelscope` 会漏掉
"pip 被删掉但包还在"的状态，那时追加安装会失败在一个很难读的 traceback 上。

### 四、两个开关（这次补进脚本头部了）

| 开关 | 作用 |
|---|---|
| `FORCE_UPLOAD=1` | 跳过 probe 闸门。用于"明知它是站桩、但就是要拿到文件"的场景 |
| `ONNX_ONLY=1` | 只传 `policy.onnx`，不传 run 目录（新增 `ms_upload.py --only-onnx`） |

```bash
MODELSCOPE_API_TOKEN=ms-... FORCE_UPLOAD=1 ONNX_ONLY=1 \
  bash scripts/sofa/modelscope_run.sh upload
```

顺带修掉一个会静默丢文件的旧行为：`stage_upload` 以前无条件打印
`uploading ... + policy.onnx`，而 `ms_upload.py` 发现 onnx 不存在就**默默跳过**、
照样报告成功。现在：

- shell 侧：`policy.onnx` 不存在就先自动跑 `stage_export`（export 的 probe 只警告不拦）；
- python 侧：显式传了 `--onnx` 但文件不在 → `exit 2`，不再假装成功。

另外 `upload_folder` 现在带 `ignore_file_pattern=[".*\.tfevents$", "events\.out\..*$"]`：
旧代码把整个 run 目录（含几百 MB tfevents）都推上去，而 `SKIP_GLOBS` 只在逐文件
回退路径里生效。SDK 不认这个参数（`TypeError`）就落到逐文件路径，那里照样过滤。

### 五、自测（本地，无 GPU/无网络）

用**手工构造的最小 wheel + 本地 PEP 503 索引**（`file://`）跑了 `ensure_ms_sdk`：

```
cold（无 .msvenv）   → 用 stdlib venv 建环境 → 从索引装 modelscope → MS_RUN=.msvenv/bin/python
warm（已就绪）       → "ModelScope SDK ready in .msvenv"，不重装
half-built（删掉 pip）→ msvenv_ready 判否 → rm -rf 重建 → 恢复可用
```

`ms_upload.py` 用桩 SDK（记录每次调用）跑了四种组合：

| 场景 | 结果 |
|---|---|
| `--only-onnx` + 文件在 | 只调 `upload_file(policy.onnx)`，不碰 run 目录 |
| 正常 + onnx 在 | `upload_folder(ignore=tfevents)` 然后 `upload_file(policy.onnx)` |
| 正常 + onnx 不在 | `exit 2`，提示 "run the export stage first" |
| SDK 无 `upload_folder` | 逐文件回退，tfevents 被过滤掉（日志里可验证） |

## 16. 几何事故复盘：鸭子一直出生在靠背里（2026-11-14 深夜 ~ 11-15 凌晨）

这一节的结论推翻了此前**四个 run 的全部前提**，所以放在最后、也最该先读。

### 一、症状

新模型（ModelScope `cartman/microduck-sofa-walk` 上的 `policy.onnx`）拿回本地，
用 microduck 的 MuJoCo 环境跑，**2.5–3 s 内必翻倒**：

```
t≈3.2s  pitch −61°…−63°  roll 88°…104°  之后完全静止
trunk z 0.633 → 0.562
```

平地对照（`scene.xml`，同一份 ONNX、同一套观测拼装）完全正常：z 稳定
0.114–0.117，roll/pitch 都是 2°。所以仿真链路、ONNX 契约、61 维观测都没问题，
问题只在**软垫或机器人模型**这一侧。

### 二、排掉的假设（都留了数据，别再走一遍）

| 假设 | 检验 | 结论 |
|---|---|---|
| ONNX 契约不匹配 | `onnxruntime` 读 IO：`[1,61] → [1,14]` | 排除 |
| BAM 执行器 vs 位置执行器 | `--no-bam` 也倒 | 排除 |
| 舵机游隙（backlash） | 用不含 backlash 的 `robot_groundcontact.xml` 造临时场景，**仍然倒** | 排除 |
| 软垫有坡度 | 射线拟合平面，坡度 −11.95° 是**未排除地板**导致的假象；只扫沙发本体则为平面 | 排除 |
| 站姿本身不稳 | 冻结所有关节的开环测试四个场景全倒（pitch 82°） | **方法无效**，站姿本来就要闭环 |

最后一条值得单独说：冻结关节的开环「站姿测试」不能用来证明软垫有问题，
因为静止站姿本身就是一个需要闭环的平衡点。这个测试浪费了半小时，
保留此条以免下次再写一遍。

### 三、根因：可用面不是包围盒，常量是照着包围盒写的

用 `mj_ray` 以 1–2 cm 网格向下扫描**编译后**的沙发模型，得到的东西和代码里
声明的完全不是一回事。旧的 `sofa_constants.py` 写的是：

```
x in [-0.45, 0.17]      ← 这是两块座垫 box 的 AABB
圆心 x ~ -0.14          ← 于是 spawn_x = (-0.18, -0.08)，中点 -0.13
```

而实测（`build_check.py` 第 0 节，现在每次构建都重算）：

```
可用面在 z≈0.518 上是一个 T 形：
  深口袋  x [-0.03, +0.50]  y [-0.41, +0.39]   0.530 × 0.800 m
  前沿台  x [+0.32, +0.50]  y [-0.83, +0.82]   0.180 × 1.655 m（0.18 m 宽，走不了）
  口袋中心 (+0.235, −0.010)

挡路的东西：
  靠背 col_11/col_12  前脸 x = −0.007 / −0.026，顶 0.9797  → 整条后带都占住
  坐垫侧翼 col_3/col_4  顶 0.868 / 0.870                   → 把 1.66 m 砍成 0.80 m
```

所以：

* **spawn_x = −0.13 落在靠背实体里**（该处射线返回 z = 0.980），偏了 0.365 m。
  鸭子每个 episode 都是「从 0.98 m 高的墙里被弹出来」，四个 run 全部如此。
* 直行任务引以为据的「1.56 m 长轴」只存在于那条 0.18 m 宽的前沿台上，
  鸭子宽 0.162 m —— 左右各剩 1 cm。
* c1 的 r ≤ 0.20 也放不下：口袋只有 0.53 m 深，机体最远角离根 0.128 m，
  所以 **r ≤ 0.137** 是硬上限。

### 四、影响面

| run | 日期 | 是否受影响 |
|---|---|---|
| 直行 v1（无后缀） | 2026-11-12 | 是 |
| 直行 `v2_bounded` | 2026-11-13 | 是 |
| 直行 `v3_gait` | 2026-11-14 | 是 |
| 绕圈 `c1_circle` | 2026-11-14 | 是 |

四份 run 里所有「鸭子不会走」的结论都建立在一个鸭子被卡住的初始状态上，
**包括 c1 的「原地踏步」**。

### 五、修复清单

| 文件 | 改动 |
|---|---|
| `robot/microduck/sofa_constants.py` | 几何块整段改写：`SEAT_FREE_X/Y`、`SEAT_LEDGE_X/Y`、`SEAT_CENTER_XY`、`BODY_FOOT_X/Y`、`BODY_REACH`、`BODY_HALF_LEN` 全部换成实测值；`SPAWN_X/Y` 回到口袋里 |
| `tasks/microduck_circle_sofa_env_cfg.py` | `CIRCLE_RADIUS = (0.08, 0.12)`、`CIRCLE_OMEGA = (0.50, 0.90)`；`CIRCLE_RADIUS_MAX` 改为**推导**并在 import 时 assert；`RADIUS_STD` 0.04 → 0.02；`run_name` → `c2_geo` |
| `tasks/microduck_velocity_sofa_env_cfg.py` | 直行也自洽：`LIN_VEL_X` ±0.20 → ±0.12、`VX_ABS_RANGE` (0.06,0.12)、`CMD_HOLD_S` (1.2,2.4) → (0.8,1.45)；`run_name` → `v4_geo` |
| `tasks/mdp.py` | `reset_on_circle` / `CircleVelocityCommandCfg` 默认值同步；§几何 注释改正 |
| `scripts/sofa/build_check.py` | **新增第 0 节几何门禁**（见下） |
| `scripts/sofa/ckpt_report.py` | `POST_FIX` 的 `cmd_hold_max_s` 2.4→1.45、`vx_abs_range_max` 0.2→0.12；`CIRCLE_RECIPES["c2"]["radius_max"]` 0.15→0.12 |
| `robot/microduck/scene_sofa.xml` / `scene_sofa_hard.xml` | duck-sim 的 keyframe 出生点 (0,0) → **(0, −0.235)** |
| `scripts/infer_policy.py` | 新增 `--spawn-x` / `--spawn-y`（原来硬编码 0,0） |

`scene_sofa.xml` 那条单独解释一下：沙发 body 绕 z 转了 −90°，
所以**世界 y = −sofa_x**。口袋 sofa x ∈ [−0.03, +0.50] 对应世界
y ∈ [−0.50, +0.03]，中心 y = −0.235。此前出生在 y = 0，鸭子右肋
**插进靠背 5.8 cm** —— duck-sim 里那次「2.5 s 翻倒」不是摔倒，是被墙挤出来。

### 六、新增门禁：build_check.py 第 0 节

`scripts/sofa/build_check.py` 现在**每个任务**都先跑几何检查，全部 30 秒 CPU：

```
ok : seat top at pocket centre z = 0.5181 (declared SEAT_TOP_Z = 0.518)
     measured pocket: x[-0.030, +0.500] (0.530 m deep)   y[-0.410, +0.390] (0.800 m long)
     measured ledge : x~+0.41 y[-0.830, +0.825] (1.655 m long) -- documented only
ok : pocket x (depth) declared [...] vs measured [...]  err 0 mm
ok : pocket y (length) declared [...] vs measured [...]  err 0 mm
ok : pocket centre declared (+0.235, -0.010) vs measured (+0.235, -0.010)
ok : spawn footprint 5 yaws x 5 x 5 positions x 4 corners = 500 rays, 0 off the seat
     old spawn x=-0.13: z=0.98 -> OFF SEAT (this is what every run before 2026-11-14 spawned into)
     old spawn y=+0.45: z=0.861 -> OFF SEAT
ok : straight segment needs 0.367 m (1.45 s x 0.12 m/s + spawn + reach) vs 0.390 m to the nearer bolster
ok : orbit r <= 0.12 m + body reach 0.128 m = 0.248 m vs half-depth 0.265 m
ok : CIRCLE_RADIUS_MAX derived in the cfg agrees with the asset (0.137 vs 0.137)
ok : orbit footprint sweep (256 rays over 2 directions x 2 radii x 16 angles) stays on seat
```

三个实现坑，都写在代码注释里了：

1. **只扫碰撞体**。资产里有 6 个可视网格，座垫中缝那条 `vis_5` 比座面低 6 cm，
   射线打上去会报 z = 0.458，整个扫描塌成「y ≥ 0」。用 `geomgroup` 按
   `contype > 0` 过滤才是可靠的 —— 事后检查 `geomid` 没用，因为 `mj_ray`
   打在第一个命中点就停，永远看不到网格后面的东西。
2. **接缝要能穿过去**。`col_0`/`col_1` 在 y = +0.005 相接，缝宽为零：接触求解没问题，
   但正好落在缝上的射线会直接漏下去报「这里什么都没有」。取样点加 1 mm 四向微扰。
3. **量内接矩形，不量包围盒**。T 形的包围盒是 0.54 × 1.66 m，没有任何一个这么大的
   轴对齐矩形是空的 —— 拿它当可用面就是 §16.3 那个 bug 换个马甲。所以先在靠背
   范围内的一行（y = +0.20，不能取 y = 0，两块靠垫在 y≈0 处有缝）量深度，
   再在靠背前脸内侧量长度（最窄处），再回代收紧，两步不动点。

### 七、仿真复验：修正前后

同一份仓库里的 `policy.onnx`、同一套命令，只改出生点：

| 场景 | 命令 | 平均速度 | trunk z | roll / pitch | 结论 |
|---|---|---|---|---|---|
| 修正前（y=0，肋部插在靠背里） | 0 | 0.046 m/s | 0.562–0.633 | 111° / 52° | 被墙挤翻 |
| **修正后**（y=−0.235） | 0 | 0.013 m/s | **0.628–0.633** | **2° / 1°** | **站住了** |
| 修正后 | v=0.06, ω=+0.5 | 0.008 m/s | 0.628–0.633 | 3° / 1° | 站住，不走 |
| 修正后 | v=0.06, ω=−0.5 | 0.012 m/s | 0.628–0.633 | 2° / 1° | 站住，不走 |
| 修正后 | v=0.06, ω=0 | 0.010 m/s | 0.628–0.633 | 3° / 1° | 站住，不走 |
| 平地对照 | v=0.06, ω=0 | 0.020 m/s | 0.114–0.117 | 2° / 2° | 勉强走（33% 指令） |

复现命令：

```bash
SCENE=src/mjlab_microduck/robot/microduck/scene_sofa.xml
.venv/bin/mjpython scripts/infer_policy.py \
    --walking policy.onnx --new-cmd-obs \
    --scene $SCENE --spawn-x 0 --spawn-y -0.235 --spawn-z 0.638 \
    --lin-vel-x 0.06 --ang-vel-z 0.5
```

### 八、结论：仓库里传的是 c1，而 c1 的问题不是几何

`policy.onnx` 与 `run/2026-11-14_11-12-49_c1_circle.onnx` **输出逐位相同**
（5 组随机 obs，max|diff| = 0）；`run/params/env.yaml` 里是
`circle_progress_reward`（高斯）+ `episode_length_s: 20.0`，没有
`circle_progress_linear_reward` / `circle_arc_ratio` / `CircleVelocityCommand`。
按 §14 的定义这是 **c1**，不是 c2。

`scripts/sofa/probe_policy.py` 也印证了：

```
linear-command sensitivity = 0.861
yaw-command sensitivity    = 0.148      ← 绕圈任务的命门
FAIL: policy barely reacts to the velocity command
```

绕圈任务的指令永远是 (ω·r, 0, ω)，yaw 通道几乎没反应 = 不可能绕圈。
加上仿真里 0.008 m/s vs 指令 0.06 m/s，结论是**它是个站桩/原地踏步策略**，
和 §14.11 的判断一致。

**几何修复解决的是「鸭子能不能站住」，不解决「鸭子愿不愿意走」。**
后者是 §14.11 那个奖励数学问题，配方已经写好（`c2_geo`：线性 arc rate +
半径按行进量门控 + 12 s episode），但**还没训**。

### 九、下一步

1. 跑 `c2_geo`（`TASK=Mjlab-Circle-Sofa-MicroDuck bash scripts/sofa/modelscope_run.sh full`）。
   构建闸门现在会先把几何重新量一遍，几何不对根本进不到训练。
2. 训完用 `verify_circle.py` 判读 —— 它只看「位置有没有绕圆心扫过角度」，
   站桩和原地打转都是 0。
3. 直行任务（`v4_geo`）已经自洽但**不指望能训出走法**：口袋 0.80 m 长，
   一个 1.45 s、0.12 m/s 的指令段就从中心快走到侧翼了。它现在的定位是
   gait bootstrap 的诊断任务，交付物是绕圈。

## 17. 桌面绕圈：能不能和沙发一起训？（2026-11-14）

### 一、结论先说

**能，而且桌子比沙发好训。** 但「同时」不是改一个常量的事：现在整条链路是
按「只有一张沙发」写死的，要变成多表面至少要动四处（见 §17.4），其中一处
（foot 接触传感器）在笔记本上无法验证。这一节把**能验证的部分全部做完并钉成
闸门**，剩下的写清楚留给上机。

### 二、资产：`ondeskgen_static_v1` 里有两张桌子

已经把 `box/` 变体收进仓库（`.obj` 视觉 + box 碰撞，452 KB）：

| 资产 | 尺寸 深×宽×高 | 碰撞 box 数 | 用途 |
|---|---|---|---|
| `odg_desk_multiview_a` | 0.59 × 1.22 × 0.757 | **13** | **训练用**（比沙发 26 个还便宜一半） |
| `odg_desk_drawers_a` | 0.64 × 1.20 × 0.786 | 73 | 只做评估/演示，不要放进混合训练 |

`mesh/` 变体没有收：README 自己就推荐 box（GPU 最快），而桌面的碰撞本来全是
平面盒子，凸包不会更准。

### 三、三张表面的实测几何

`scripts/sofa/check_surfaces.py`（新增，纯 mujoco、无 mjlab、5 秒）：

```
surface          top z    内接矩形         中心               r_cap   余量(vs r=0.12)
sofa seat        0.5181   0.53 x 0.80      (+0.235, -0.010)   0.137   17 mm
desk_multiview   0.7566   0.59 x 1.21      ( 0.000,  0.000)   0.167   47 mm
desk_drawers     0.7863   0.60 x 1.19      (-0.015,  0.000)   0.172   52 mm
```

`r_cap = min(半深, 半长) - BODY_REACH(0.128)`。**关键结论：现在绕圈任务用的
`CIRCLE_RADIUS = (0.08, 0.12)` 在三张表面上都放得下，最紧的是沙发（17 mm）。**
也就是说「一条策略同时绕两种家具」在指令空间上是自洽的 —— 不需要按表面改
命令范围，这是可行性最硬的一条证据。

顺带两个「别照包围盒抄」的坑，都写进 `desk_constants.py` 注释了：

* 抽屉桌的 +x/+y 角是**切角**：射线在 (0.32, 0.45) 返回 0.696，在 (0.30, 0.50)
  返回「什么都没有」。可用深度到 +0.285 就停，而包围盒会给你 +0.3025 ——
  差了 1.8 cm，正好是鸭子掉下去的量。
* 多视图桌的顶板（col_5）和背后柜体（col_0）**同高**（0.7566 / 0.7560），
  中间只有 0.4 mm 台阶，所以正面 0.59 × 1.21 是一整块平面，没有缝。

### 四、要「同时训」，必须改这四处

| # | 现在 | 必须变成 | 难度 |
|---|---|---|---|
| 1 | `pose_range["z"] = SEAT_TOP_Z + 0.12`（标量，模块级） | 每 env 一个 `surface_z` buffer | 中 |
| 2 | `reset_on_circle(center_xy=SEAT_CENTER_XY)` | 每 env 一个 `center_xy` | 中 |
| 3 | `fell_off_sofa(minimum_height=SEAT_TOP_Z - 0.20)` | 每 env 阈值（桌面要 0.56，沙发 0.32） | 易 |
| 4 | `feet_ground_contact.secondary` 指向**一个** entity 的 body | 同时覆盖沙发 + 桌子两个 body | **难，且本地验证不了** |

第 4 条展开：mjlab 1.3.0 的 `ContactMatch` 在传了 `entity=` 时，只在该 entity
内部展开正则（见 build_check.py 顶部注释 —— 不传 `entity` 则是**字面名字**，正则
会直接让 `MjSpec.compile` 挂掉）。场景里有两套家具就有两个 body，单一 secondary
盖不住。两条退路：

* 正则覆盖多个名字（如果 `ContactMatch` 支持，需要上机试）；
* 建三个 sensor（`feet_sofa_contact` / `feet_desk_contact` …）+ 在 `mdp.py` 里
  写一个把三个 air-time 相加的 reward（约 30 行，但要真机验证）。

**这一条没验证过就不能开训** —— `air_time` 权重 3.0，接不上就等于没有步态激励，
而 2026-11-13 那次正是这么废掉的（air_time 全程 0.000）。

场景拼法也有两条路，推荐第二条：

* ~~三套家具各自 mocap，每 env 把不用的挪到别处~~ —— 要写自定义 reset，
  且 3 份碰撞体仍然在 broadphase 里，白花钱；
* **两种家具都放在同一个 env patch 里**（沙发在原点，桌子在 (2.0, 0)），
  `env_spacing` 3.0 → 4.0，reset 时按 env 抽一个表面去站。不需要 teleport，
  也不需要 parking，代价只是每 env 多 13 个碰撞体。

几何预算（每 env）：机器人 81 geom（其中 11 个碰）+ 沙发 26 + 桌子 13 = **120**。
塞 73 个盒子的抽屉桌会让它变成 180，纯亏。

### 五、桌面不是万灵药：实测同一份 ONNX 在桌上

新增 `scene_desk.xml`（可直接 `DUCK_SIM_SCENE=desk DUCK_SIM_KEYFRAME=DESK_STAND` 跑），
用仓库里那份 `policy.onnx`，只换表面：

| 表面 | 指令 | 平均速度 | trunk z | roll/pitch | yaw 漂移 |
|---|---|---|---|---|---|
| 沙发 | 0 | 0.013 | 0.628–0.633 | 2° / 1° | +0.21 rad/s |
| **桌面** | 0 | 0.017 | 0.870–0.873 | 2° / 2° | **+0.35 rad/s** |
| 桌面 | (0.06, ω=+0.5) | 0.018 | 0.870–0.873 | 3° / 2° | +0.38 rad/s |
| 桌面 | (0.06, ω=−0.5) | 0.014 | 0.870–0.873 | 2° / 1° | +0.25 rad/s |
| 平地 | (0.06, 0) | 0.020 | 0.114–0.117 | 2° / 2° | +0.13 rad/s |

三件事：

1. **桌面站得住**（2–3°、不滑落）—— 说明桌子这个平台是能走的，几何和接触都没问题；
2. **还是不走**（0.015 vs 指令 0.06）—— 和沙发一模一样。这不是表面问题，是 §14.11
   的奖励数学问题，换家具不解决；
3. **yaw 一直在正向漂，且与指令符号无关**（+0.5 和 −0.5 都往正向掉），硬表面
   漂得比软垫多。也就是说：**软垫在掩盖这个 policy 的 yaw 偏置**。这恰好是
   「一起训」的价值 —— 硬表面会把偏差暴露出来，逼它学真的 yaw 控制，而不是
   靠泡棉的阻尼混过去。

### 六、做到哪一步 / 还差什么

已完成且可本地复现：

* `assets/desk/`：两张桌子（box 变体，相对路径，无绝对路径，bundle 可跨机器）
* `desk_constants.py`：实测几何 + 硬接触（solref 0.02，**故意**不跟沙发一起软化）
* `scene_desk.xml`：duck-sim 可直接跑（内联而非 include，因为 `meshdir="assets"`
  会把 include 里的 mesh 路径双前缀 —— 沙发当初写绝对路径就是这个坑）
* `check_surfaces.py`：把「一条 orbit range 通不通用」变成 5 秒 CPU 判定，
  并已做变异测试（把 top_z 改成 0.777 → 3 项 FAIL、退出码 1）

还没做（需要上机验证，不能盲写）：

* 多表面 env cfg（`c3_multi`）：surface buffer、per-env 阈值、接触传感器接线
* `build_check.py` 第 0 节扩成多表面（现在仍只量沙发；`check_surfaces.py`
  已经覆盖桌面，等 cfg 落地后并进去）
* `verify_circle.py` 加 `--surface`

建议的顺序仍是先修「不愿意走」（§14.11 的 c2 线性 arc-rate 配方，还没训），
再把桌面并进去 —— 否则会得到一条在两种表面上都站桩的策略，白花一倍机时。
