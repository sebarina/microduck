# OnDeskGen：仿真就绪家具资产引擎

基于群核 Lux3D，从一段文字或一张图片生成**可以直接用于 MuJoCo 仿真训练**的家具资产，带碰撞代理、物理属性和可动部件。

| 目录 | 内容 |
|---|---|
| [`onDeskGen/`](onDeskGen/) | 3D 资产生成：调用 Lux3D（文生图、四视图、图生 3D、文生 3D、材质迁移、部件拆分、铰接、多格式导出），再生成碰撞凸包、质量与惯量、关节、MJCF，输出资产包和训练用的静态 MJCF。命令行 `ondeskgen`。 |
| [`WebUI/`](WebUI/) | 网页界面：演示视频、资产引擎（现场生成、3D 预览、打包下载）、仿真训练。依赖 `onDeskGen/`。 |
| [`microduck_rl/`](microduck_rl/) | 仿真训练：OnDeskGen 资产 → 静态 MJCF → mjlab 训练场景 → 魔搭 GPU 训练 → 本机仿真回放。依赖 `onDeskGen/`。 |

快速开始：

```bash
cd onDeskGen && uv sync
export AHOLO_API_KEY=...               # Lux3D
export ONDESKGEN_UPLOAD_URL=...        # 文件服务（见 onDeskGen/README.md）
uv run ondeskgen generate desk --text "浅色橡木书桌，右侧带两个抽屉"

cd ../WebUI && uv sync && uv run ondeskgen-webui     # http://127.0.0.1:8765


cd ../microduck_rl

# train the walking policy (uses your GPU; ~1-2 h for a usable gait at 4096 envs)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# watch a trained policy in the viewer
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# export to ONNX for deployment
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
uv run publish --onnx output.onnx --repo <user>/microduck-<name> --kind episodic --duration-s 4.0   # share it (see "Publishing a policy")

# drive the exported policy in CPU MuJoCo with the keyboard
uv run scripts/infer_policy.py --walking output.onnx
```

三部分各自的安装、配置和用法见各目录下的 README。
